//! Symmetric eigendecomposition of the Hessian into modes.

use nalgebra::{DMatrix, SymmetricEigen};

/// Eigenvalues and eigenvectors of a symmetric matrix, sorted by ascending
/// eigenvalue.
///
/// `eigenvalues[k]` pairs with column `k` of `eigenvectors`. nalgebra's
/// `SymmetricEigen` does not order its output, so we sort here: ascending order
/// is the convention that puts the (near-zero) rigid-body modes first, matching
/// every NMA reference.
pub(crate) struct Spectrum {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: DMatrix<f64>,
}

pub(crate) fn solve(matrix: DMatrix<f64>) -> Spectrum {
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
