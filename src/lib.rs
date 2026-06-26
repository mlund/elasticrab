//! Anisotropic Network Model (ANM) normal-mode analysis.
//!
//! Give it atoms, get back the vibrational modes of an elastic network: a
//! coarse spring model where every pair of atoms closer than a cutoff is joined
//! by a harmonic spring. Diagonalizing the resulting Hessian yields the normal
//! modes — the collective, low-energy motions a structure most readily makes.
//!
//! ```
//! use elasticrab::{Atom, Params, NormalModes};
//!
//! let atoms = vec![
//!     Atom { position: [0.0, 0.0, 0.0], mass: 12.0 },
//!     Atom { position: [3.8, 0.0, 0.0], mass: 12.0 },
//!     Atom { position: [3.8, 3.8, 0.0], mass: 12.0 },
//! ];
//! let modes = NormalModes::new(&atoms, &Params::default()).unwrap();
//!
//! // Eigenvalues are ascending; the lowest few are the ~zero rigid-body modes.
//! assert_eq!(modes.len(), 9); // 3 atoms × 3 Cartesian axes
//! ```
//!
//! # Scope
//!
//! This is the standard ANM with a uniform spring constant, the same
//! super-element Hessian used by tools such as ProDy and Pepsi-SAXS. It
//! deliberately stops at "frequencies and modes": structure parsing, hydration,
//! coarse-graining, and any fitting of amplitudes to data belong to the caller.
//!
//! The eigensolver is **dense** (cost grows with the cube of the atom count),
//! which is ideal for small and medium systems and what the test suite
//! validates. Larger systems will want a sparse partial solver; that is an
//! internal change that would not affect this public API.

#![deny(missing_docs)]
// Deliberate choices that conflict with two `clippy::nursery` lints:
// `pub(crate)` is kept in private modules as explicit intent (vs. bare `pub`),
// and the hot distance loop favors a readable sum over fused multiply-add.
#![allow(clippy::redundant_pub_crate, clippy::suboptimal_flops)]

mod eigen;
mod hessian;
mod network;

/// A point mass in the elastic network.
///
/// `mass` is in arbitrary units and is **ignored** unless
/// [`Params::mass_weighted`] is set; the default analysis treats every atom
/// equally, matching the conventional ANM.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Atom {
    /// Cartesian coordinates, in ångström.
    pub position: [f64; 3],
    /// Atomic mass; only used when mass-weighting is enabled.
    pub mass: f64,
}

/// Parameters of the elastic-network model.
///
/// [`Default`] reproduces the conventional ANM settings (15 Å cutoff, unit
/// spring constant, no mass-weighting), which is also the configuration the
/// crate validates against ProDy.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Params {
    /// Maximum distance (ångström) for two atoms to be joined by a spring.
    pub cutoff: f64,
    /// Uniform spring constant. Scales all eigenvalues linearly, so its
    /// absolute value only matters relative to how you interpret the output.
    pub gamma: f64,
    /// When true, diagonalize the mass-weighted Hessian `M^{-1/2} H M^{-1/2}`
    /// instead of `H`; eigenvalues are then squared frequencies `ω²`.
    pub mass_weighted: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            cutoff: 15.0,
            gamma: 1.0,
            mass_weighted: false,
        }
    }
}

/// Why an analysis could not be performed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Fewer than two atoms were given; a network needs at least one spring.
    TooFewAtoms,
    /// A coordinate was not finite, or mass-weighting was requested with a
    /// non-positive or non-finite mass (which has no real square root).
    NotFinite,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewAtoms => write!(f, "at least two atoms are required"),
            Self::NotFinite => write!(f, "non-finite coordinate or mass"),
        }
    }
}

impl std::error::Error for Error {}

/// The normal modes of an elastic network: eigenvalues paired with mode shapes,
/// sorted by ascending eigenvalue.
///
/// The lowest few eigenvalues are ~zero and correspond to rigid-body motion
/// (three translations plus, for a non-degenerate shape, three rotations); the
/// genuine internal motions follow.
#[derive(Debug)]
pub struct NormalModes {
    eigenvalues: Vec<f64>,
    /// Displacements flattened mode-major: mode `k` occupies
    /// `modes[k * n_atoms .. (k + 1) * n_atoms]`, one `[f64; 3]` per atom.
    modes: Vec<[f64; 3]>,
    n_atoms: usize,
}

/// Eigenvalues at or below this magnitude are treated as rigid-body (zero)
/// modes when deriving thermal amplitudes, which guards the `1/ω` against the
/// tiny positive *and negative* values a finite-precision solver returns.
const ZERO_EIGENVALUE: f64 = 1e-6;

/// Boltzmann constant in kcal·mol⁻¹·K⁻¹. The absolute amplitude scale is only
/// meaningful relative to `gamma` and your unit choices, so callers commonly
/// rescale the result regardless.
const BOLTZMANN_KCAL_PER_MOL_K: f64 = 1.987_204_259e-3;

impl NormalModes {
    /// Build the ANM Hessian for `atoms` and diagonalize it.
    ///
    /// Heavy and fallible by design — it assembles a `3N×3N` matrix and runs a
    /// symmetric eigendecomposition. Named `new` to match the decomposition
    /// constructors of the numeric ecosystem it builds on (nalgebra's
    /// `SymmetricEigen::new`, and `Regex::new`).
    pub fn new(atoms: &[Atom], params: &Params) -> Result<Self, Error> {
        if atoms.len() < 2 {
            return Err(Error::TooFewAtoms);
        }

        let positions: Vec<[f64; 3]> = atoms.iter().map(|a| a.position).collect();
        if positions.iter().flatten().any(|x| !x.is_finite()) {
            return Err(Error::NotFinite);
        }
        if params.mass_weighted && atoms.iter().any(|a| !(a.mass.is_finite() && a.mass > 0.0)) {
            return Err(Error::NotFinite);
        }

        let contacts = network::contacts(&positions, params.cutoff);
        let mut h = hessian::build(atoms.len(), params.gamma, &contacts);
        if params.mass_weighted {
            let masses: Vec<f64> = atoms.iter().map(|a| a.mass).collect();
            hessian::mass_weight(&mut h, &masses);
        }

        let spectrum = eigen::solve(h);
        let n_atoms = atoms.len();
        let mut modes = Vec::with_capacity(spectrum.eigenvalues.len() * n_atoms);
        for col in spectrum.eigenvectors.column_iter() {
            modes.extend((0..n_atoms).map(|a| [col[3 * a], col[3 * a + 1], col[3 * a + 2]]));
        }

        Ok(Self {
            eigenvalues: spectrum.eigenvalues,
            modes,
            n_atoms,
        })
    }

    /// Number of modes, equal to three times the atom count.
    pub const fn len(&self) -> usize {
        self.eigenvalues.len()
    }

    /// Whether there are no modes (only possible for an empty result).
    pub const fn is_empty(&self) -> bool {
        self.eigenvalues.is_empty()
    }

    /// Eigenvalues in ascending order. The first ~6 are approximately zero.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Mode `i` as a per-atom displacement field; entry `a` is the motion of
    /// atom `a`. The vector is unit-normalized over all atoms.
    ///
    /// # Panics
    /// If `i >= self.len()`.
    pub fn eigenvector(&self, i: usize) -> &[[f64; 3]] {
        &self.modes[i * self.n_atoms..(i + 1) * self.n_atoms]
    }

    /// Thermal RMS amplitudes `√(2·k_B·T / λ_i)` per mode at temperature `T`
    /// (kelvin). Rigid-body modes (eigenvalue ≈ 0) are reported as `0.0` so the
    /// returned slice stays index-aligned with [`eigenvalues`](Self::eigenvalues).
    ///
    /// The absolute scale is arbitrary in an ANM (it rides on `gamma`); the
    /// useful information is the *relative* amplitude across modes.
    pub fn thermal_amplitudes(&self, temperature_k: f64) -> Vec<f64> {
        let two_kt = 2.0 * BOLTZMANN_KCAL_PER_MOL_K * temperature_k;
        self.eigenvalues
            .iter()
            .map(|&lambda| {
                if lambda > ZERO_EIGENVALUE {
                    (two_kt / lambda).sqrt()
                } else {
                    0.0
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn carbon(x: f64, y: f64, z: f64) -> Atom {
        Atom {
            position: [x, y, z],
            mass: 12.0,
        }
    }

    #[test]
    fn too_few_atoms_is_rejected() {
        let r = NormalModes::new(&[carbon(0.0, 0.0, 0.0)], &Params::default());
        assert!(matches!(r, Err(Error::TooFewAtoms)));
    }

    #[test]
    fn non_finite_coordinate_is_rejected() {
        let atoms = [carbon(0.0, 0.0, 0.0), carbon(f64::NAN, 0.0, 0.0)];
        let r = NormalModes::new(&atoms, &Params::default());
        assert!(matches!(r, Err(Error::NotFinite)));
    }

    /// A non-collinear, well-connected cluster has exactly six rigid-body modes:
    /// three translations and three rotations, all with ~zero eigenvalue.
    #[test]
    fn six_zero_modes_for_a_3d_cluster() {
        let atoms = [
            carbon(0.0, 0.0, 0.0),
            carbon(1.5, 0.0, 0.0),
            carbon(0.0, 1.5, 0.0),
            carbon(0.0, 0.0, 1.5),
            carbon(1.0, 1.0, 1.0),
        ];
        let modes = NormalModes::new(&atoms, &Params::default()).unwrap();

        let zeros = modes
            .eigenvalues()
            .iter()
            .filter(|&&v| v.abs() < ZERO_EIGENVALUE)
            .count();
        assert_eq!(zeros, 6);
        // The seventh mode is a genuine internal motion with positive eigenvalue.
        assert!(modes.eigenvalues()[6] > ZERO_EIGENVALUE);
    }

    #[test]
    fn thermal_amplitudes_align_and_zero_out_rigid_modes() {
        let atoms = [
            carbon(0.0, 0.0, 0.0),
            carbon(1.5, 0.0, 0.0),
            carbon(0.0, 1.5, 0.0),
            carbon(1.0, 1.0, 1.0),
        ];
        let modes = NormalModes::new(&atoms, &Params::default()).unwrap();
        let amps = modes.thermal_amplitudes(300.0);

        assert_eq!(amps.len(), modes.len());
        // First six modes are rigid-body -> zero amplitude.
        for &a in &amps[..6] {
            assert_relative_eq!(a, 0.0);
        }
        // Softer modes (smaller eigenvalue) fluctuate more than stiffer ones.
        assert!(amps[6] >= amps[7]);
    }

    /// Mass-weighting check with a closed-form answer: a diatomic has a single
    /// internal (stretching) mode whose eigenvalue is the reduced-mass relation
    /// `ω² = γ·(1/m₁ + 1/m₂)`; its other five modes are rigid-body.
    #[test]
    fn mass_weighting_reproduces_diatomic_reduced_mass() {
        let atoms = [
            Atom {
                position: [0.0, 0.0, 0.0],
                mass: 12.0,
            },
            Atom {
                position: [2.0, 0.0, 0.0],
                mass: 16.0,
            },
        ];
        let params = Params {
            cutoff: 5.0,
            gamma: 1.0,
            mass_weighted: true,
        };
        let modes = NormalModes::new(&atoms, &params).unwrap();

        let nonzero = modes
            .eigenvalues()
            .iter()
            .filter(|&&v| v.abs() > ZERO_EIGENVALUE)
            .count();
        assert_eq!(nonzero, 1);
        assert_relative_eq!(
            *modes.eigenvalues().last().unwrap(),
            1.0 / 12.0 + 1.0 / 16.0,
            epsilon = 1e-10
        );
    }

    /// With every mass equal to `m`, weighting reduces to `H/m`, so the whole
    /// spectrum is the unweighted one scaled by `1/m`.
    #[test]
    fn equal_mass_weighting_scales_spectrum() {
        let m = 4.0;
        let atoms = [
            Atom {
                position: [0.0, 0.0, 0.0],
                mass: m,
            },
            Atom {
                position: [1.5, 0.0, 0.0],
                mass: m,
            },
            Atom {
                position: [0.0, 1.5, 0.0],
                mass: m,
            },
            Atom {
                position: [1.0, 1.0, 1.0],
                mass: m,
            },
        ];
        let base = Params {
            cutoff: 5.0,
            gamma: 1.0,
            mass_weighted: false,
        };
        let weighted = Params {
            mass_weighted: true,
            ..base
        };

        let unit = NormalModes::new(&atoms, &base).unwrap();
        let scaled = NormalModes::new(&atoms, &weighted).unwrap();
        for k in 0..unit.len() {
            assert_relative_eq!(
                scaled.eigenvalues()[k],
                unit.eigenvalues()[k] / m,
                epsilon = 1e-9
            );
        }
    }
}
