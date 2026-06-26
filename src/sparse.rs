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

    // Lower triangle of the mass-weighted Hessian, accumulated by (row, col).
    let mut entries: HashMap<(usize, usize), f64> = HashMap::new();
    for c in contacts {
        let s = -gamma / c.dist2;
        for a in 0..3 {
            for b in 0..3 {
                let block = s * c.delta[a] * c.delta[b];
                // Off-diagonal block (j, i) is entirely below the diagonal (j > i).
                *entries.entry((3 * c.j + a, 3 * c.i + b)).or_default() +=
                    block * scale[3 * c.j + a] * scale[3 * c.i + b];
                // Diagonal blocks accumulate the negated super-element; keep the
                // lower triangle (a ≥ b).
                if a >= b {
                    *entries.entry((3 * c.i + a, 3 * c.i + b)).or_default() -=
                        block * scale[3 * c.i + a] * scale[3 * c.i + b];
                    *entries.entry((3 * c.j + a, 3 * c.j + b)).or_default() -=
                        block * scale[3 * c.j + a] * scale[3 * c.j + b];
                }
            }
        }
    }

    let max_diag = (0..dof)
        .map(|d| entries.get(&(d, d)).copied().unwrap_or(0.0))
        .fold(0.0_f64, f64::max);
    let shift = SHIFT_FRACTION * max_diag.max(f64::MIN_POSITIVE);

    let mut triplets: Vec<Triplet<usize, usize, f64>> = entries
        .iter()
        .map(|(&(r, c), &v)| Triplet::new(r, c, if r == c { v + shift } else { v }))
        .collect();
    // Diagonal entries with no contacts still need the shift to stay positive.
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
