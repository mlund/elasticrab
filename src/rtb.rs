//! Rotation-Translation Blocks (RTB) projection.
//!
//! Each block of atoms is treated as a rigid body with up to six degrees of
//! freedom (three translations, three rotations). The projection matrix `P`
//! maps those block DOF onto all-atom Cartesian displacements; the elastic
//! network is then solved in the reduced space `Pᵀ H P` and lifted back with
//! `P·v`. Ported from ProDy's `rtbtools.c::dblock_projections2`, generalized so
//! per-atom weights (unit, or atomic mass) work the same way.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Unit, Vector3};

use crate::Error;

/// Below this angular speed a block's nonlinear motion is treated as a pure
/// translation, which also guards the rotation-axis normalization against a
/// near-zero vector.
const ROTATION_EPS: f64 = 1e-12;

/// Sparse projection entries `(row, col, value)` plus the column count `nb6`.
pub(crate) type ProjectionEntries = (Vec<(usize, usize, f64)>, usize);

/// An inertia eigenvalue below this fraction of the block's largest means the
/// block is rank-deficient (collinear or coincident atoms) and has no
/// well-defined rotational basis.
const DEGENERATE_RATIO: f64 = 1e-9;

/// The rigid geometry of one block: which atoms it holds, where its columns sit
/// in the reduced space, and the quantities the projection and the nonlinear
/// extrapolation both need — centre of mass, total (weighted) mass, and the
/// inertia inverse square root (`None` for a single-atom block, which has no
/// rotational basis).
#[derive(Debug)]
pub(crate) struct BlockGeometry {
    pub atoms: Vec<usize>,
    pub col: usize,
    pub dof: usize,
    pub com: Vector3<f64>,
    pub total_mass: f64,
    pub isqrt: Option<Matrix3<f64>>,
}

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
    let (entries, nb6) = projection_entries(positions, weights, blocks)?;
    let mut p = DMatrix::zeros(3 * positions.len(), nb6);
    for (r, c, v) in entries {
        p[(r, c)] = v;
    }
    Ok(p)
}

/// The projection as `(row, col, value)` triplets plus its column count `nb6`.
/// Plain tuples (no matrix type) so both the dense [`projection`] and the sparse
/// matrix-free solver can consume them.
pub(crate) fn projection_entries(
    positions: &[[f64; 3]],
    weights: &[f64],
    blocks: &[usize],
) -> Result<ProjectionEntries, Error> {
    let geometry = block_geometry(positions, weights, blocks)?;
    let nb6 = geometry.iter().map(|g| g.dof).sum();

    let mut entries = Vec::new();
    for block in &geometry {
        emit_block_columns(
            &mut |r, c, v| entries.push((r, c, v)),
            positions,
            weights,
            block,
        );
    }
    Ok((entries, nb6))
}

/// Per-block rigid geometry, in first-appearance order of the block ids. Shared
/// by the projection (which turns it into matrix columns) and the nonlinear
/// extrapolation (which turns the reduced velocities into rigid motions).
pub(crate) fn block_geometry(
    positions: &[[f64; 3]],
    weights: &[f64],
    blocks: &[usize],
) -> Result<Vec<BlockGeometry>, Error> {
    let groups = group_by_block(blocks);
    let mut geometry = Vec::with_capacity(groups.len());
    let mut col = 0;
    for atoms in groups {
        let dof = block_dof(atoms.len());
        let total_mass: f64 = atoms.iter().map(|&i| weights[i]).sum();

        let mut com = Vector3::zeros();
        for &i in &atoms {
            com += weights[i] * Vector3::from(positions[i]);
        }
        com /= total_mass;

        // The inertia inverse square root orthonormalizes the rotational basis;
        // a single-atom block has no rotation, so none is needed.
        let isqrt = if atoms.len() == 1 {
            None
        } else {
            let mut inertia = Matrix3::zeros();
            for &i in &atoms {
                let x = Vector3::from(positions[i]) - com;
                inertia += weights[i] * (x.dot(&x) * Matrix3::identity() - x * x.transpose());
            }
            Some(inverse_sqrt(inertia)?)
        };

        geometry.push(BlockGeometry {
            atoms,
            col,
            dof,
            com,
            total_mass,
            isqrt,
        });
        col += dof;
    }
    Ok(geometry)
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

/// Emit the translation (and, for multi-atom blocks, rotation) column entries
/// for one block as `(row, col, value)` via `emit`. The geometry was already
/// computed by [`block_geometry`]; this turns it into the projection's columns.
fn emit_block_columns(
    emit: &mut impl FnMut(usize, usize, f64),
    positions: &[[f64; 3]],
    weights: &[f64],
    block: &BlockGeometry,
) {
    let sqrt_total = block.total_mass.sqrt();

    // Translations: a uniform shift of the block, normalized in the (weighted)
    // metric so the column is unit length.
    for &i in &block.atoms {
        let s = weights[i].sqrt() / sqrt_total;
        for axis in 0..3 {
            emit(3 * i + axis, block.col + axis, s);
        }
    }

    // Rotations: orthonormalized by the inertia inverse square root, so the three
    // rotational columns are unit length and mutually orthogonal.
    let Some(isqrt) = block.isqrt else {
        return;
    };
    let generators: [Vector3<f64>; 3] =
        std::array::from_fn(|axis| Vector3::from(isqrt.row(axis).transpose()));
    for &i in &block.atoms {
        let x = Vector3::from(positions[i]) - block.com;
        let s = weights[i].sqrt();
        for (axis, generator) in generators.iter().enumerate() {
            // Displacement of atom i under rotation `axis` is generator × offset.
            let rot = generator.cross(&x);
            for coord in 0..3 {
                emit(3 * i + coord, block.col + 3 + axis, s * rot[coord]);
            }
        }
    }
}

/// NOLB's nonlinear extrapolation: carry each block along the reduced
/// `velocities` (an `nb6` vector of *amplitude-scaled* per-block linear and
/// angular velocities — a single mode, or a combined sum of modes) as a rigid
/// motion. The block rotates about its centre of mass and translates, so
/// intra-block bond lengths are preserved at any amplitude. Atoms not in any
/// block keep their position.
///
/// The reduced velocities are un-weighted to physical ones (NOLB eq. 1.11):
/// `v = ṽ_w/√M_b`, `ω = I^{-1/2}·ω̃_w`; a single-atom block has no rotation. The
/// amplitude is folded into `velocities`, so the rotation angle is `‖ω‖` directly.
pub(crate) fn extrapolate(
    blocks: &[BlockGeometry],
    positions: &[[f64; 3]],
    velocities: &DVector<f64>,
) -> Vec<[f64; 3]> {
    let mut out = positions.to_vec();
    for block in blocks {
        let col = block.col;
        let velocity = Vector3::new(velocities[col], velocities[col + 1], velocities[col + 2]);
        let translation = velocity / block.total_mass.sqrt();
        // The rotation about the COM is the same for every atom, so build it once;
        // a singleton (no `isqrt`) or a vanishing rotation leaves only translation.
        let rotation = block.isqrt.and_then(|isqrt| {
            let omega = isqrt
                * Vector3::new(
                    velocities[col + 3],
                    velocities[col + 4],
                    velocities[col + 5],
                );
            let speed = omega.norm();
            (speed > ROTATION_EPS)
                .then(|| Rotation3::from_axis_angle(&Unit::new_normalize(omega), speed))
        });
        for &atom in &block.atoms {
            let position = Vector3::from(positions[atom]);
            let moved = rotation.as_ref().map_or_else(
                || position + translation,
                |rotation| rotation * (position - block.com) + block.com + translation,
            );
            out[atom] = [moved.x, moved.y, moved.z];
        }
    }
    out
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
