//! Partial eigensolver: the `k` lowest non-zero modes, without ever forming a
//! dense `3N×3N` matrix.
//!
//! The Hessian is assembled sparse, then we run **shift-invert Lanczos**: Lanczos
//! on `(K + εI)⁻¹`, whose largest eigenvalues `μ = 1/(λ + ε)` are the smallest
//! `λ`. The small positive shift `ε` keeps `K + εI` positive-definite (so it has
//! a Cholesky factor) and sends the exact rigid-body modes to the largest finite
//! `μ = 1/ε`; they come out first and are dropped by their recovered `λ ≈ 0`.
//! Full reorthogonalization keeps the few wanted modes accurate.

use std::collections::HashMap;

use faer::col::Col;
use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use faer::Side;
use nalgebra::{DMatrix, DVector};

use crate::network::Contact;
use crate::Error;

/// Shift `ε`, relative to the largest diagonal entry: small enough not to blur
/// the soft modes, large enough to keep `K + εI` well-conditioned for Cholesky.
const SHIFT_FRACTION: f64 = 1e-6;

/// An eigenvalue below this fraction of the spectrum's top is treated as a
/// rigid-body (zero) mode and excluded.
const ZERO_FRACTION: f64 = 1e-7;

/// Compute the `k` lowest non-zero modes of the (optionally mass-weighted) ANM
/// Hessian. Returns their eigenvalues (ascending) and the matching eigenvectors
/// as columns of a `3N × k` matrix.
pub(crate) fn lowest_nonzero_modes(
    n_atoms: usize,
    gamma: f64,
    weights: &[f64],
    contacts: &[Contact],
    k: usize,
) -> Result<(Vec<f64>, DMatrix<f64>), Error> {
    let dof = 3 * n_atoms;
    let scale = crate::hessian::dof_scale(weights);

    let entries = hessian_entries(gamma, &scale, contacts);
    let max_diag = (0..dof)
        .map(|d| entries.get(&(d, d)).copied().unwrap_or(0.0))
        .fold(0.0_f64, f64::max);
    let shift = SHIFT_FRACTION * max_diag.max(f64::MIN_POSITIVE);

    // Cholesky reads the lower triangle, so emit only `row ≥ col`, with the shift
    // added on the diagonal to keep `K + εI` positive-definite.
    let mut triplets: Vec<Triplet<usize, usize, f64>> = entries
        .iter()
        .filter(|(&(r, c), _)| r >= c)
        .map(|(&(r, c), &v)| Triplet::new(r, c, if r == c { v + shift } else { v }))
        .collect();
    // Disconnected DOFs have no diagonal entry; the shift alone keeps them positive.
    for d in 0..dof {
        if !entries.contains_key(&(d, d)) {
            triplets.push(Triplet::new(d, d, shift));
        }
    }

    let a = SparseColMat::<usize, f64>::try_new_from_triplets(dof, dof, &triplets)
        .map_err(|_| Error::SparseSolverFailed)?;
    let llt = a
        .sp_cholesky(Side::Lower)
        .map_err(|_| Error::SparseSolverFailed)?;
    let apply_inverse = |v: &DVector<f64>| -> DVector<f64> {
        let rhs = Col::from_fn(dof, |i| v[i]);
        let y = llt.solve(&rhs);
        DVector::from_fn(dof, |i, _| y[i])
    };

    // Krylov dimension: generous margin over the wanted count plus the rigid-body
    // modes, capped at the problem size (where Lanczos becomes exact).
    let want = k + 6;
    let steps = (2 * want + 20).min(dof);
    let (mu, ritz) = lanczos(dof, steps, apply_inverse);

    // Convert μ back to λ, sort ascending, drop the rigid-body modes, keep k.
    let zero_tol = ZERO_FRACTION * max_diag.max(f64::MIN_POSITIVE);
    let mut modes: Vec<(f64, usize)> = mu
        .iter()
        .enumerate()
        .map(|(c, &m)| (1.0 / m - shift, c))
        .filter(|&(lambda, _)| lambda > zero_tol)
        .collect();
    modes.sort_by(|x, y| x.0.total_cmp(&y.0));
    modes.truncate(k);

    let eigenvalues: Vec<f64> = modes.iter().map(|&(l, _)| l).collect();
    let vectors = DMatrix::from_fn(dof, modes.len(), |r, c| ritz[(r, modes[c].1)]);
    Ok((eigenvalues, vectors))
}

/// The full symmetric mass-weighted Hessian as `(row, col) -> value`. The
/// shift-invert path keeps only the lower triangle; the matrix-free path needs
/// the whole matrix for its mat-vecs.
fn hessian_entries(
    gamma: f64,
    scale: &[f64],
    contacts: &[Contact],
) -> HashMap<(usize, usize), f64> {
    let mut acc: HashMap<(usize, usize), f64> = HashMap::new();
    for c in contacts {
        let s = -gamma * c.weight / c.dist2;
        for a in 0..3 {
            for b in 0..3 {
                let raw = s * c.delta[a] * c.delta[b];
                // Off-diagonal block and its symmetric counterpart.
                let (ia, jb) = (3 * c.i + a, 3 * c.j + b);
                let off = raw * scale[ia] * scale[jb];
                *acc.entry((ia, jb)).or_default() += off;
                *acc.entry((jb, ia)).or_default() += off;
                // Diagonal blocks accumulate the negated super-element.
                let (ii_a, ii_b) = (3 * c.i + a, 3 * c.i + b);
                let (jj_a, jj_b) = (3 * c.j + a, 3 * c.j + b);
                *acc.entry((ii_a, ii_b)).or_default() -= raw * scale[ii_a] * scale[ii_b];
                *acc.entry((jj_a, jj_b)).or_default() -= raw * scale[jj_a] * scale[jj_b];
            }
        }
    }
    acc
}

/// Gershgorin upper bound on the largest eigenvalue: the largest absolute row sum.
fn gershgorin_bound(acc: &HashMap<(usize, usize), f64>, dof: usize) -> f64 {
    let mut row_abs = vec![0.0_f64; dof];
    for (&(r, _), &v) in acc {
        row_abs[r] += v.abs();
    }
    row_abs.iter().copied().fold(0.0_f64, f64::max)
}

/// `(eigenvalues, lifted all-atom vectors, reduced block-space vectors)`. The
/// reduced vectors carry the per-block velocities the nonlinear extrapolation
/// needs; the lifted ones are the per-atom mode shapes.
pub(crate) type RtbModes = (Vec<f64>, DMatrix<f64>, DMatrix<f64>);

/// Matrix-free RTB partial solver: the `k` lowest non-zero modes of the reduced
/// operator `R = Pᵀ K P`, applied without ever forming `R`. Regular-mode Lanczos
/// on `c·I − R` (whose largest eigenvalues are `R`'s smallest, recovered as
/// `λ = c − μ`), with `c` a Gershgorin bound that holds for `R` because `PᵀP = I`.
pub(crate) fn lowest_rtb_modes(
    positions: &[[f64; 3]],
    weights: &[f64],
    blocks: &[usize],
    gamma: f64,
    contacts: &[Contact],
    k: usize,
) -> Result<RtbModes, Error> {
    let dof = 3 * positions.len();
    let scale = crate::hessian::dof_scale(weights);
    let acc = hessian_entries(gamma, &scale, contacts);
    let bound = gershgorin_bound(&acc, dof);

    let fail = |_| Error::SparseSolverFailed;
    let k_triplets: Vec<Triplet<usize, usize, f64>> = acc
        .iter()
        .map(|(&(r, c), &v)| Triplet::new(r, c, v))
        .collect();
    let k_mat = SparseColMat::try_new_from_triplets(dof, dof, &k_triplets).map_err(fail)?;

    // P (3N × nb6) and its transpose, both sparse, from the block projection.
    let (entries, nb6) = crate::rtb::projection_entries(positions, weights, blocks)?;
    let (p_triplets, pt_triplets): (Vec<_>, Vec<_>) = entries
        .iter()
        .map(|&(r, c, v)| (Triplet::new(r, c, v), Triplet::new(c, r, v)))
        .unzip();
    let p = SparseColMat::try_new_from_triplets(dof, nb6, &p_triplets).map_err(fail)?;
    let pt = SparseColMat::try_new_from_triplets(nb6, dof, &pt_triplets).map_err(fail)?;

    // op(y) = c·y − Pᵀ(K(P y)), all sparse mat-vecs — R is never formed.
    let op = |y: &DVector<f64>| -> DVector<f64> {
        let py = &p * &Col::from_fn(nb6, |i| y[i]);
        let r_y = &pt * &(&k_mat * &py);
        DVector::from_fn(nb6, |i, _| bound * y[i] - r_y[i])
    };

    // Regular-mode Lanczos converges more slowly to clustered soft modes than
    // the shift-invert path, so use a more generous Krylov dimension.
    let steps = (4 * (k + 6) + 40).min(nb6);
    let (mu, ritz) = lanczos(nb6, steps, op);

    let zero_tol = ZERO_FRACTION * bound.max(f64::MIN_POSITIVE);
    let mut modes: Vec<(f64, usize)> = mu
        .iter()
        .enumerate()
        .map(|(c, &m)| (bound - m, c))
        .filter(|&(lambda, _)| lambda > zero_tol)
        .collect();
    modes.sort_by(|x, y| x.0.total_cmp(&y.0));
    modes.truncate(k);

    let eigenvalues: Vec<f64> = modes.iter().map(|&(l, _)| l).collect();
    // Keep the reduced (block-space) eigenvectors and also lift them to all-atom
    // space with P. The reduced ones carry the per-block velocities the nonlinear
    // extrapolation needs; the lifted ones are the per-atom mode shapes.
    let mut reduced = DMatrix::zeros(nb6, modes.len());
    let mut vectors = DMatrix::zeros(dof, modes.len());
    for (out, &(_, ritz_col)) in modes.iter().enumerate() {
        let reduced_mode = Col::from_fn(nb6, |i| ritz[(i, ritz_col)]);
        let lifted = &p * &reduced_mode;
        for r in 0..nb6 {
            reduced[(r, out)] = reduced_mode[r];
        }
        for r in 0..dof {
            vectors[(r, out)] = lifted[r];
        }
    }
    Ok((eigenvalues, vectors, reduced))
}

/// Full-reorthogonalization Lanczos on the operator `op` (here `(K + εI)⁻¹`).
/// Returns the Ritz values (eigenvalues of `op`) and the Ritz vectors as columns.
fn lanczos(
    dof: usize,
    steps: usize,
    op: impl Fn(&DVector<f64>) -> DVector<f64>,
) -> (Vec<f64>, DMatrix<f64>) {
    // Deterministic, spectrum-covering start vector (no RNG dependency).
    let mut v = DVector::from_fn(dof, |i, _| ((i + 1) as f64).sin());
    v /= v.norm();

    let mut basis: Vec<DVector<f64>> = Vec::with_capacity(steps);
    let mut alpha = Vec::with_capacity(steps);
    let mut beta = Vec::with_capacity(steps);

    for _ in 0..steps {
        basis.push(v.clone());
        let mut w = op(&v);
        let a = w.dot(&v);
        alpha.push(a);

        // Subtract the Lanczos recurrence, then reorthogonalize against the whole
        // basis twice — cheap insurance against loss of orthogonality.
        w -= &v * a;
        if basis.len() >= 2 {
            w -= &basis[basis.len() - 2] * *beta.last().unwrap();
        }
        for _ in 0..2 {
            for q in &basis {
                let proj = w.dot(q);
                w -= q * proj;
            }
        }

        let b = w.norm();
        if b < 1e-12 {
            break; // invariant subspace found
        }
        beta.push(b);
        v = w / b;
    }

    // Diagonalize the small tridiagonal projection T = basisᵀ op basis.
    let m = basis.len();
    let mut t = DMatrix::zeros(m, m);
    for i in 0..m {
        t[(i, i)] = alpha[i];
        if i + 1 < m {
            t[(i, i + 1)] = beta[i];
            t[(i + 1, i)] = beta[i];
        }
    }
    let eig = t.symmetric_eigen();

    // Ritz vectors = basis · (eigenvectors of T).
    let basis_mat = DMatrix::from_fn(dof, m, |r, c| basis[c][r]);
    let ritz = basis_mat * eig.eigenvectors;
    (eig.eigenvalues.iter().copied().collect(), ritz)
}
