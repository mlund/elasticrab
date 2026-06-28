//! Kabsch optimal superposition: the rigid rotation and translation that best
//! overlays one point set onto another (minimizing RMSD), from an SVD of the
//! cross-covariance with a reflection guard so the result is always a proper
//! rotation. Used to remove the rigid-body component of a target conformation
//! before projecting its deformation onto the normal modes.

use nalgebra::{Matrix3, Vector3};

/// A rigid transform: rotate, then translate.
pub(crate) struct Superposition {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
}

impl Superposition {
    /// Apply the transform to a point set: `R·p + t` for each point.
    pub(crate) fn apply(&self, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
        points
            .iter()
            .map(|p| {
                let moved = self.rotation * Vector3::from(*p) + self.translation;
                [moved.x, moved.y, moved.z]
            })
            .collect()
    }
}

/// The rigid transform that best superposes `mobile` onto `reference` (paired by
/// index, same length), minimizing the RMSD between `R·mobile + t` and
/// `reference`. Kabsch via an SVD of the cross-covariance, with the `diag(1,1,d)`
/// correction so an otherwise-optimal *reflection* is replaced by the best proper
/// rotation (`det R = +1`).
pub(crate) fn superpose(reference: &[[f64; 3]], mobile: &[[f64; 3]]) -> Superposition {
    let n = reference.len() as f64;
    let centroid = |points: &[[f64; 3]]| {
        points
            .iter()
            .fold(Vector3::zeros(), |acc, p| acc + Vector3::from(*p))
            / n
    };
    let c_ref = centroid(reference);
    let c_mobile = centroid(mobile);

    // Cross-covariance H = Σ (mobile − c_mobile)(reference − c_ref)ᵀ.
    let mut h = Matrix3::zeros();
    for (r, m) in reference.iter().zip(mobile) {
        h += (Vector3::from(*m) - c_mobile) * (Vector3::from(*r) - c_ref).transpose();
    }

    // R = V·diag(1,1,d)·Uᵀ from H = U·Σ·Vᵀ, with d = sign(det(V·Uᵀ)) forbidding a
    // reflection (the lone case where the naive V·Uᵀ has det −1).
    let svd = h.svd(true, true);
    let (u, v_t) = (svd.u.unwrap(), svd.v_t.unwrap());
    let d = (v_t.transpose() * u.transpose()).determinant().signum();
    let rotation =
        v_t.transpose() * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, d)) * u.transpose();
    let translation = c_ref - rotation * c_mobile;
    Superposition {
        rotation,
        translation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Unit};

    /// A small non-collinear, non-planar point set (a distorted tetrahedron).
    fn fixture() -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.4, 0.2, -0.3],
            [0.1, 1.7, 0.5],
            [-0.6, 0.4, 1.9],
            [2.1, -1.0, 0.8],
        ]
    }

    #[test]
    fn recovers_a_known_rigid_transform() {
        let reference = fixture();
        // Move the set by a known rotation + translation to make the mobile copy.
        let rot =
            Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::new(1.0, -2.0, 0.5)), 0.9);
        let shift = Vector3::new(3.0, -1.5, 2.0);
        let mobile: Vec<[f64; 3]> = reference
            .iter()
            .map(|p| {
                let q = rot * Vector3::from(*p) + shift;
                [q.x, q.y, q.z]
            })
            .collect();

        // Superposing mobile back onto reference must reproduce reference.
        let aligned = superpose(&reference, &mobile).apply(&mobile);
        for (a, r) in aligned.iter().zip(&reference) {
            for c in 0..3 {
                assert!((a[c] - r[c]).abs() < 1e-9, "atom mismatch: {a:?} vs {r:?}");
            }
        }
    }

    #[test]
    fn returns_a_proper_rotation_for_a_reflected_target() {
        let reference = fixture();
        // A mirror image cannot be matched by a rotation; the result must still be
        // a proper rotation (det +1), not the det −1 reflection the SVD would give
        // without the guard.
        let mobile: Vec<[f64; 3]> = reference.iter().map(|p| [p[0], p[1], -p[2]]).collect();
        let rotation = superpose(&reference, &mobile).rotation;
        assert!((rotation.determinant() - 1.0).abs() < 1e-9);
    }
}
