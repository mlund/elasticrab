//! Rotation-Translation Blocks (RTB) projection.
//!
//! Each block of atoms is treated as a rigid body with up to six degrees of
//! freedom (three translations, three rotations). The projection matrix `P`
//! maps those block DOF onto all-atom Cartesian displacements; the elastic
//! network is then solved in the reduced space `Pᵀ H P` and lifted back with
//! `P·v`. Ported from ProDy's `rtbtools.c::dblock_projections2`, generalized so
//! per-atom weights (unit, or atomic mass) work the same way.

use std::collections::HashMap;

use nalgebra::{DMatrix, Matrix3, Vector3};

use crate::Error;

/// An inertia eigenvalue below this fraction of the block's largest means the
/// block is rank-deficient (collinear or coincident atoms) and has no
/// well-defined rotational basis.
const DEGENERATE_RATIO: f64 = 1e-9;

/// Build the RTB projection `P` of shape `3·n_atoms × nb6`.
///
/// `weights` are per-atom (all `1.0` reproduces the conventional unit-mass RTB;
/// atomic masses give the mass-weighted variant, consistent with how the
/// Hessian itself is weighted). Atoms are grouped by `blocks[i]`; block ids need
/// not be contiguous and are taken in first-appearance order so the column
/// layout is independent of the actual id values. A single-atom block keeps only
/// its 3 translations; any larger block keeps all 6.
pub(crate) fn projection(
    positions: &[[f64; 3]],
    weights: &[f64],
    blocks: &[usize],
) -> Result<DMatrix<f64>, Error> {
    let groups = group_by_block(blocks);
    let nb6: usize = groups.iter().map(|g| block_dof(g.len())).sum();

    let mut p = DMatrix::zeros(3 * positions.len(), nb6);
    let mut col = 0;
    for atoms in &groups {
        block_columns(&mut p, positions, weights, atoms, col)?;
        col += block_dof(atoms.len());
    }
    Ok(p)
}

const fn block_dof(size: usize) -> usize {
    if size == 1 {
        3
    } else {
        6
    }
}

/// Atom indices grouped by block id, in first-appearance order of the ids.
fn group_by_block(blocks: &[usize]) -> Vec<Vec<usize>> {
    let mut slot_of: HashMap<usize, usize> = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (atom, &id) in blocks.iter().enumerate() {
        let slot = *slot_of.entry(id).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[slot].push(atom);
    }
    groups
}

/// Fill the translation (and, for multi-atom blocks, rotation) columns for one
/// block starting at column `col`.
fn block_columns(
    p: &mut DMatrix<f64>,
    positions: &[[f64; 3]],
    weights: &[f64],
    atoms: &[usize],
    col: usize,
) -> Result<(), Error> {
    let total_w: f64 = atoms.iter().map(|&i| weights[i]).sum();
    let sqrt_total = total_w.sqrt();

    let mut center = Vector3::zeros();
    for &i in atoms {
        center += weights[i] * Vector3::from(positions[i]);
    }
    center /= total_w;

    // Translations: a uniform shift of the block, normalized in the (weighted)
    // metric so the column is unit length.
    for &i in atoms {
        let s = weights[i].sqrt() / sqrt_total;
        for axis in 0..3 {
            p[(3 * i + axis, col + axis)] = s;
        }
    }

    if atoms.len() == 1 {
        return Ok(());
    }

    // Rotations: orthonormalized via the inverse square root of the block's
    // (weighted) inertia tensor, so the three rotational columns are unit length
    // and mutually orthogonal.
    let offsets: Vec<Vector3<f64>> = atoms
        .iter()
        .map(|&i| Vector3::from(positions[i]) - center)
        .collect();

    let mut inertia = Matrix3::zeros();
    for (&i, x) in atoms.iter().zip(&offsets) {
        inertia += weights[i] * (x.dot(x) * Matrix3::identity() - x * x.transpose());
    }
    let isqrt = inverse_sqrt(inertia)?;

    // The three orthonormalized rotation generators (rows of the inertia inverse
    // square root) are the same for every atom, so build them once.
    let generators: [Vector3<f64>; 3] =
        std::array::from_fn(|axis| Vector3::from(isqrt.row(axis).transpose()));

    for (&i, x) in atoms.iter().zip(&offsets) {
        let s = weights[i].sqrt();
        for axis in 0..3 {
            // Displacement of atom i under rotation `axis` is generator × offset.
            let rot = generators[axis].cross(x);
            for coord in 0..3 {
                p[(3 * i + coord, col + 3 + axis)] = s * rot[coord];
            }
        }
    }
    Ok(())
}

/// Symmetric inverse square root `A·diag(1/√λ)·Aᵀ` of a positive-definite 3×3
/// matrix, or [`Error::DegenerateBlock`] if it is rank-deficient.
fn inverse_sqrt(m: Matrix3<f64>) -> Result<Matrix3<f64>, Error> {
    let eig = m.symmetric_eigen();
    let max = eig.eigenvalues.max();
    if max <= 0.0 || eig.eigenvalues.iter().any(|&l| l < DEGENERATE_RATIO * max) {
        return Err(Error::DegenerateBlock);
    }
    let inv_sqrt = eig.eigenvalues.map(|l| 1.0 / l.sqrt());
    Ok(eig.eigenvectors * Matrix3::from_diagonal(&inv_sqrt) * eig.eigenvectors.transpose())
}
