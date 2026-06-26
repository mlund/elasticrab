//! Symmetric eigendecomposition of the Hessian into modes.
//!
//! With the `sparse` feature (which already pulls faer) this uses faer's
//! SIMD-vectorized self-adjoint eigensolver; otherwise nalgebra's scalar
//! `SymmetricEigen`. Both return eigenvalues in ascending order with the matching
//! eigenvector as column `k`.

use nalgebra::DMatrix;

/// Eigenvalues and eigenvectors of a symmetric matrix, sorted by ascending
/// eigenvalue. `eigenvalues[k]` pairs with column `k` of `eigenvectors`; the
/// (near-zero) rigid-body modes come first, matching every NMA reference.
pub(crate) struct Spectrum {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: DMatrix<f64>,
}

/// Dense solve via nalgebra. `SymmetricEigen` does not order its output, so we
/// sort ascending here.
#[cfg(not(feature = "sparse"))]
pub(crate) fn solve(matrix: DMatrix<f64>) -> Spectrum {
    use nalgebra::SymmetricEigen;
    let dof = matrix.nrows();
    let eig = SymmetricEigen::new(matrix);

    let mut order: Vec<usize> = (0..dof).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));

    let eigenvalues = order.iter().map(|&k| eig.eigenvalues[k]).collect();
    let eigenvectors = DMatrix::from_fn(dof, dof, |r, c| eig.eigenvectors[(r, order[c])]);

    Spectrum {
        eigenvalues,
        eigenvectors,
    }
}

/// Dense solve via faer's SIMD self-adjoint eigensolver, which already returns
/// eigenvalues in nondecreasing order. faer reads only the lower triangle; the
/// Hessian is symmetric, so that is exact.
#[cfg(feature = "sparse")]
pub(crate) fn solve(matrix: DMatrix<f64>) -> Spectrum {
    let dof = matrix.nrows();
    let m = faer::Mat::from_fn(dof, dof, |i, j| matrix[(i, j)]);
    let eig = m
        .self_adjoint_eigen(faer::Side::Lower)
        .expect("self-adjoint eigendecomposition");

    let diag = eig.S();
    let s = diag.column_vector();
    let eigenvalues = (0..dof).map(|i| s[i]).collect();
    let u = eig.U();
    let eigenvectors = DMatrix::from_fn(dof, dof, |r, c| u[(r, c)]);

    Spectrum {
        eigenvalues,
        eigenvectors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn returns_ascending_eigenvalues() {
        // diag(3, 1, 2) -> sorted eigenvalues 1, 2, 3.
        let m = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![3.0, 1.0, 2.0]));
        let s = solve(m);
        assert_relative_eq!(s.eigenvalues[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(s.eigenvalues[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(s.eigenvalues[2], 3.0, epsilon = 1e-10);
    }
}
