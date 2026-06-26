//! Assembly of the ANM Hessian from a list of contacts.

use nalgebra::DMatrix;

use crate::network::Contact;

/// Build the `3N×3N` Anisotropic Network Model Hessian.
///
/// Each contact contributes a 3×3 super-element `B = -(gamma / d²)·(d ⊗ d)`,
/// where `d` is the inter-atom displacement. `B` is written to the two
/// off-diagonal blocks `(i, j)` and `(j, i)`, and *subtracted* from the two
/// diagonal blocks `(i, i)` and `(j, j)`. Subtracting on the diagonal is what
/// makes every 3-row block sum to zero, i.e. a uniform translation of all atoms
/// costs no energy — the physical invariant that produces the three zero
/// translational modes.
pub(crate) fn build(n_atoms: usize, gamma: f64, contacts: &[Contact]) -> DMatrix<f64> {
    let dof = 3 * n_atoms;
    let mut h = DMatrix::zeros(dof, dof);

    for c in contacts {
        let scale = -gamma / c.dist2;
        for (a, &da) in c.delta.iter().enumerate() {
            for (b, &db) in c.delta.iter().enumerate() {
                // The super-element d ⊗ d is symmetric, so both off-diagonal
                // blocks (i,j) and (j,i) receive the same value.
                let block = scale * da * db;
                h[(3 * c.i + a, 3 * c.j + b)] += block;
                h[(3 * c.j + a, 3 * c.i + b)] += block;
                h[(3 * c.i + a, 3 * c.i + b)] -= block;
                h[(3 * c.j + a, 3 * c.j + b)] -= block;
            }
        }
    }
    h
}

/// Apply symmetric mass-weighting in place: `H ← M^{-1/2} H M^{-1/2}` with
/// `M = diag(mass)` repeated over the three Cartesian axes.
///
/// The *symmetric* form is used (rather than the asymmetric `M^{-1} H`) so the
/// matrix stays symmetric and can still go through a symmetric eigensolver; its
/// eigenvalues are the squared vibrational frequencies `ω²`.
pub(crate) fn mass_weight(h: &mut DMatrix<f64>, masses: &[f64]) {
    // One inverse-√mass per Cartesian DOF, so the inner loop is a plain product
    // with no per-element division by 3.
    let scale = dof_scale(masses);
    let dof = h.nrows();
    for r in 0..dof {
        for col in 0..dof {
            h[(r, col)] *= scale[r] * scale[col];
        }
    }
}

/// Inverse-√mass weight per Cartesian DOF: one value per atom, repeated ×3.
/// Shared by [`mass_weight`] and the sparse solver so the convention stays in
/// one place.
pub(crate) fn dof_scale(masses: &[f64]) -> Vec<f64> {
    masses.iter().flat_map(|m| [1.0 / m.sqrt(); 3]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::contacts;
    use approx::assert_relative_eq;

    /// Two atoms separated by 2 Å along x, gamma = 1. The only spring is along
    /// x, so the super-element is `-1` in the xx slot and 0 elsewhere; the
    /// diagonal blocks carry +1. Hand-computed against the ANM formula.
    #[test]
    fn two_atom_super_element() {
        let pos = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let h = build(2, 1.0, &contacts(&pos, 5.0));

        assert_relative_eq!(h[(0, 3)], -1.0); // (atom0_x, atom1_x)
        assert_relative_eq!(h[(3, 0)], -1.0);
        assert_relative_eq!(h[(0, 0)], 1.0); // diagonal block negates it
        assert_relative_eq!(h[(3, 3)], 1.0);
        // No coupling perpendicular to the bond.
        assert_relative_eq!(h[(1, 4)], 0.0);
    }

    #[test]
    fn is_symmetric_and_blocks_sum_to_zero() {
        let pos = [
            [0.0, 0.0, 0.0],
            [1.2, 0.3, 0.0],
            [0.4, 1.1, 0.7],
            [-0.6, 0.5, 1.3],
        ];
        let h = build(4, 1.0, &contacts(&pos, 5.0));
        let dof = h.nrows();

        for r in 0..dof {
            for col in 0..dof {
                assert_relative_eq!(h[(r, col)], h[(col, r)], epsilon = 1e-12);
            }
        }
        // Each row summed over whole-atom blocks vanishes (translational invariance).
        for r in 0..dof {
            let s: f64 = (0..dof).map(|col| h[(r, col)]).sum();
            assert_relative_eq!(s, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn unit_mass_weighting_is_identity() {
        let pos = [[0.0, 0.0, 0.0], [1.5, 0.2, 0.0], [0.3, 1.4, 0.5]];
        let cs = contacts(&pos, 5.0);
        let plain = build(3, 1.0, &cs);
        let mut weighted = build(3, 1.0, &cs);
        mass_weight(&mut weighted, &[1.0, 1.0, 1.0]);
        for r in 0..plain.nrows() {
            for col in 0..plain.ncols() {
                assert_relative_eq!(plain[(r, col)], weighted[(r, col)], epsilon = 1e-12);
            }
        }
    }
}
