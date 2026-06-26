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
//! The default solver is **dense** (cost grows with the cube of the atom count),
//! ideal for small and medium systems. For large systems, the optional `sparse`
//! feature adds a partial solver: set [`Params::k_modes`] to compute only the
//! lowest *k* non-zero modes from the sparse Hessian, never forming a dense one.

#![deny(missing_docs)]
// Deliberate choices that conflict with two `clippy::nursery` lints:
// `pub(crate)` is kept in private modules as explicit intent (vs. bare `pub`),
// and the hot distance loop favors a readable sum over fused multiply-add.
#![allow(clippy::redundant_pub_crate, clippy::suboptimal_flops)]

mod eigen;
mod hessian;
mod network;
mod rtb;
#[cfg(feature = "sparse")]
mod sparse;

use nalgebra::DMatrix;

use network::Contact;

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
    /// Number of lowest *non-zero* modes to compute. `None` (the default) returns
    /// all modes, including the ~6 rigid-body ones, via a dense solve. `Some(k)`
    /// returns exactly the `k` lowest non-zero modes from a sparse partial solver
    /// — rigid-body modes excluded — and requires the `sparse` crate feature.
    pub k_modes: Option<usize>,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            cutoff: 15.0,
            gamma: 1.0,
            mass_weighted: false,
            k_modes: None,
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
    /// The block list passed to [`NormalModes::with_blocks`] did not have one
    /// entry per atom.
    BlockCountMismatch,
    /// A multi-atom block is rank-deficient (collinear or coincident atoms), so
    /// it has no well-defined rotational basis. Use a single-atom block, or
    /// blocks of three or more non-collinear atoms.
    DegenerateBlock,
    /// [`Params::k_modes`] was set but the crate was built without the `sparse`
    /// feature, which provides the partial eigensolver.
    SparseFeatureRequired,
    /// The sparse solver could not factor the Hessian or did not converge.
    SparseSolverFailed,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewAtoms => write!(f, "at least two atoms are required"),
            Self::NotFinite => write!(f, "non-finite coordinate or mass"),
            Self::BlockCountMismatch => write!(f, "blocks must have one entry per atom"),
            Self::DegenerateBlock => write!(f, "a multi-atom block is collinear or coincident"),
            Self::SparseFeatureRequired => write!(f, "k_modes requires the `sparse` feature"),
            Self::SparseSolverFailed => write!(f, "the sparse solver failed"),
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
    /// Original indices of atoms dropped for being disconnected (degree 0).
    disconnected: Vec<usize>,
}

/// Eigenvalues at or below this magnitude are treated as rigid-body (zero)
/// modes when deriving thermal amplitudes, which guards the `1/ω` against the
/// tiny positive *and negative* values a finite-precision solver returns.
const ZERO_EIGENVALUE: f64 = 1e-6;

/// Boltzmann constant in kcal·mol⁻¹·K⁻¹. The absolute amplitude scale is only
/// meaningful relative to `gamma` and your unit choices, so callers commonly
/// rescale the result regardless.
const BOLTZMANN_KCAL_PER_MOL_K: f64 = 1.987_204_259e-3;

/// Shared validation: returns per-atom positions and the per-atom weights used
/// by the Hessian and any block projection (atomic masses when mass-weighting,
/// otherwise unit).
fn validated_inputs(atoms: &[Atom], params: &Params) -> Result<(Vec<[f64; 3]>, Vec<f64>), Error> {
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
    let weights = if params.mass_weighted {
        atoms.iter().map(|a| a.mass).collect()
    } else {
        vec![1.0; atoms.len()]
    };
    Ok((positions, weights))
}

/// Assemble the all-atom ANM Hessian from precomputed contacts, mass-weighted
/// when requested. `weights` equals the atomic masses on the mass-weighted path
/// and is otherwise unused.
fn build_hessian(
    n_atoms: usize,
    weights: &[f64],
    contacts: &[Contact],
    params: &Params,
) -> DMatrix<f64> {
    let mut h = hessian::build(n_atoms, params.gamma, contacts);
    if params.mass_weighted {
        hessian::mass_weight(&mut h, weights);
    }
    h
}

/// The connected elastic network, ready to solve: positions, weights, and
/// contacts renumbered to the atoms a spring actually touches. `keep[p]` is the
/// original index of kept atom `p`; `disconnected` lists the degree-0 atoms that
/// were removed (empty for a fully connected structure).
struct Network {
    positions: Vec<[f64; 3]>,
    weights: Vec<f64>,
    contacts: Vec<Contact>,
    keep: Vec<usize>,
    disconnected: Vec<usize>,
}

/// Validate the atoms, build the cell-list contacts, and drop disconnected atoms
/// — the shared front half of both constructors. Fails if fewer than two atoms
/// remain connected.
fn prepare(atoms: &[Atom], params: &Params) -> Result<Network, Error> {
    let (positions, weights) = validated_inputs(atoms, params)?;
    let contacts = network::contacts(&positions, params.cutoff);
    let net = drop_disconnected(positions, weights, contacts);
    if net.keep.len() < 2 {
        return Err(Error::TooFewAtoms);
    }
    Ok(net)
}

/// Remove atoms that no spring touches (degree 0). A fully connected network is
/// returned unchanged with the identity `keep`, so the common case copies
/// nothing.
fn drop_disconnected(
    positions: Vec<[f64; 3]>,
    weights: Vec<f64>,
    contacts: Vec<Contact>,
) -> Network {
    let n = positions.len();
    let disconnected = network::disconnected_atoms(n, &contacts);
    if disconnected.is_empty() {
        return Network {
            positions,
            weights,
            contacts,
            keep: (0..n).collect(),
            disconnected,
        };
    }

    let mut dropped = vec![false; n];
    for &d in &disconnected {
        dropped[d] = true;
    }
    let mut new_index = vec![usize::MAX; n];
    let mut keep = Vec::with_capacity(n - disconnected.len());
    for old in 0..n {
        if !dropped[old] {
            new_index[old] = keep.len();
            keep.push(old);
        }
    }

    // Disconnected atoms have no contacts, so both endpoints are always kept.
    let contacts = contacts
        .into_iter()
        .map(|c| Contact {
            i: new_index[c.i],
            j: new_index[c.j],
            delta: c.delta,
            dist2: c.dist2,
        })
        .collect();
    Network {
        positions: keep.iter().map(|&old| positions[old]).collect(),
        weights: keep.iter().map(|&old| weights[old]).collect(),
        contacts,
        keep,
        disconnected,
    }
}

impl NormalModes {
    /// Build the ANM Hessian for `atoms` and diagonalize it.
    ///
    /// Heavy and fallible by design — it assembles a `3N×3N` matrix and runs a
    /// symmetric eigendecomposition. Named `new` to match the decomposition
    /// constructors of the numeric ecosystem it builds on (nalgebra's
    /// `SymmetricEigen::new`, and `Regex::new`).
    pub fn new(atoms: &[Atom], params: &Params) -> Result<Self, Error> {
        let net = prepare(atoms, params)?;

        if let Some(k) = params.k_modes {
            return Self::solve_partial(&net, params, k, atoms.len());
        }

        let h = build_hessian(net.keep.len(), &net.weights, &net.contacts, params);
        let spectrum = eigen::solve(h);
        Ok(Self::from_modes(
            spectrum.eigenvalues,
            &spectrum.eigenvectors,
            &net,
            atoms.len(),
        ))
    }

    /// The `k` lowest non-zero modes via the sparse partial solver.
    #[cfg(feature = "sparse")]
    fn solve_partial(
        net: &Network,
        params: &Params,
        k: usize,
        n_original: usize,
    ) -> Result<Self, Error> {
        let (eigenvalues, vectors) = sparse::lowest_nonzero_modes(
            net.keep.len(),
            params.gamma,
            &net.weights,
            &net.contacts,
            k,
        )?;
        Ok(Self::from_modes(eigenvalues, &vectors, net, n_original))
    }

    /// Without the `sparse` feature there is no partial solver, so `k_modes` is
    /// an explicit error rather than a silent dense solve.
    #[cfg(not(feature = "sparse"))]
    const fn solve_partial(_: &Network, _: &Params, _: usize, _: usize) -> Result<Self, Error> {
        Err(Error::SparseFeatureRequired)
    }

    /// Group atoms into rigid blocks (the Rotation-Translation Blocks method)
    /// and solve the reduced eigenproblem.
    ///
    /// `blocks` gives one block id per atom (parallel to `atoms`); ids need not
    /// be contiguous. Each block keeps six rigid degrees of freedom — three if
    /// it is a single atom — so the problem shrinks to `nb6 ≤ 6·n_blocks`
    /// coordinates. The returned modes are the same per-atom displacement fields
    /// as [`new`](Self::new), lifted back from the reduced space.
    ///
    /// This is the model used by Pepsi-SAXS / NOLB. With every atom in its own
    /// block it reduces exactly to [`new`](Self::new).
    ///
    /// With [`Params::k_modes`] set, only the lowest `k` non-zero modes are
    /// computed, via a matrix-free partial solver that never forms the reduced
    /// matrix (requires the `sparse` feature).
    pub fn with_blocks(atoms: &[Atom], blocks: &[usize], params: &Params) -> Result<Self, Error> {
        if blocks.len() != atoms.len() {
            return Err(Error::BlockCountMismatch);
        }
        let net = prepare(atoms, params)?;
        // The drop carries the blocks along: a block keeps only its connected atoms.
        let blocks: Vec<usize> = net.keep.iter().map(|&old| blocks[old]).collect();

        if let Some(k) = params.k_modes {
            return Self::solve_rtb_partial(&net, &blocks, params, k, atoms.len());
        }

        let h = build_hessian(net.keep.len(), &net.weights, &net.contacts, params);
        // Reduce to the block subspace, solve there, then lift modes back with P.
        // `tr_mul` forms Pᵀ·(H·P) without materializing the transpose of P.
        let p = rtb::projection(&net.positions, &net.weights, &blocks)?;
        let reduced = p.tr_mul(&(&h * &p));
        let spectrum = eigen::solve(reduced);
        let all_atom = &p * spectrum.eigenvectors;
        Ok(Self::from_modes(
            spectrum.eigenvalues,
            &all_atom,
            &net,
            atoms.len(),
        ))
    }

    /// The `k` lowest non-zero RTB modes via the matrix-free partial solver.
    #[cfg(feature = "sparse")]
    fn solve_rtb_partial(
        net: &Network,
        blocks: &[usize],
        params: &Params,
        k: usize,
        n_original: usize,
    ) -> Result<Self, Error> {
        let (eigenvalues, vectors) = sparse::lowest_rtb_modes(
            &net.positions,
            &net.weights,
            blocks,
            params.gamma,
            &net.contacts,
            k,
        )?;
        Ok(Self::from_modes(eigenvalues, &vectors, net, n_original))
    }

    #[cfg(not(feature = "sparse"))]
    const fn solve_rtb_partial(
        _: &Network,
        _: &[usize],
        _: &Params,
        _: usize,
        _: usize,
    ) -> Result<Self, Error> {
        Err(Error::SparseFeatureRequired)
    }

    /// Repackage an eigendecomposition (columns = modes, rows = `3·keep.len()`
    /// Cartesian coordinates of the connected atoms) into per-atom storage
    /// indexed by the *original* atoms. `net.keep[p]` is the original index of
    /// solved atom `p`; disconnected atoms get a zero displacement in every mode.
    /// With an identity `keep` (a fully connected network) this is a plain repack.
    fn from_modes(
        eigenvalues: Vec<f64>,
        vectors: &DMatrix<f64>,
        net: &Network,
        n_original: usize,
    ) -> Self {
        let mut modes = vec![[0.0; 3]; eigenvalues.len() * n_original];
        for (m, col) in vectors.column_iter().enumerate() {
            let base = m * n_original;
            for (p, &orig) in net.keep.iter().enumerate() {
                modes[base + orig] = [col[3 * p], col[3 * p + 1], col[3 * p + 2]];
            }
        }
        Self {
            eigenvalues,
            modes,
            n_atoms: n_original,
            disconnected: net.disconnected.clone(),
        }
    }

    /// Number of modes: three per connected atom for the plain model, or the
    /// reduced rigid-block degree-of-freedom count for
    /// [`with_blocks`](Self::with_blocks).
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

    /// Original indices of atoms dropped from the analysis for being
    /// disconnected — no spring within the cutoff (degree 0). Empty for a fully
    /// connected network. A dropped atom's entry is `[0, 0, 0]` in every mode.
    ///
    /// This mirrors Pepsi-SAXS / NOLB, which exclude such atoms before solving.
    pub fn disconnected(&self) -> &[usize] {
        &self.disconnected
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
            k_modes: None,
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
            k_modes: None,
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

    // --- RTB (Rotation-Translation Blocks) ---

    /// A connected, non-collinear six-atom cluster; any three of its atoms are
    /// non-collinear, so it is safe to carve into multi-atom blocks.
    fn cluster6() -> Vec<Atom> {
        [
            (0.0, 0.0, 0.0),
            (1.2, 0.0, 0.0),
            (0.0, 1.2, 0.0),
            (0.0, 0.0, 1.2),
            (1.2, 1.2, 0.0),
            (1.0, 0.5, 1.0),
        ]
        .iter()
        .map(|&(x, y, z)| carbon(x, y, z))
        .collect()
    }

    fn rtb_params() -> Params {
        Params {
            cutoff: 5.0,
            ..Params::default()
        }
    }

    /// Each atom in its own block ⇒ the projection is the identity, so RTB must
    /// reproduce the plain ANM spectrum exactly. Ties RTB to the ProDy-validated path.
    #[test]
    fn all_singleton_blocks_match_plain_anm() {
        let atoms = cluster6();
        let blocks: Vec<usize> = (0..atoms.len()).collect();

        let plain = NormalModes::new(&atoms, &rtb_params()).unwrap();
        let rtb = NormalModes::with_blocks(&atoms, &blocks, &rtb_params()).unwrap();

        assert_eq!(rtb.len(), plain.len());
        for (a, b) in rtb.eigenvalues().iter().zip(plain.eigenvalues()) {
            assert_relative_eq!(a, b, epsilon = 1e-9);
        }
    }

    /// Only the grouping matters, not the id values: relabeling the blocks
    /// leaves the spectrum unchanged.
    #[test]
    fn block_id_values_are_remapped() {
        let atoms = cluster6();
        let a = NormalModes::with_blocks(&atoms, &[0, 0, 0, 1, 1, 1], &rtb_params()).unwrap();
        let b = NormalModes::with_blocks(&atoms, &[42, 42, 42, 7, 7, 7], &rtb_params()).unwrap();
        for (x, y) in a.eigenvalues().iter().zip(b.eigenvalues()) {
            assert_relative_eq!(x, y, epsilon = 1e-12);
        }
    }

    /// Blocks are grouped by id regardless of atom order: interleaved ids put
    /// non-adjacent atoms in the same block and still yield two 6-DOF blocks.
    #[test]
    fn interleaved_blocks_are_grouped_by_id() {
        let atoms = cluster6();
        let modes = NormalModes::with_blocks(&atoms, &[0, 1, 0, 1, 0, 1], &rtb_params()).unwrap();
        assert_eq!(modes.len(), 12); // two non-singleton blocks, 6 DOF each
        let zeros = modes
            .eigenvalues()
            .iter()
            .filter(|&&v| v.abs() < 1e-6)
            .count();
        assert_eq!(zeros, 6);
    }

    /// A single block spanning the whole structure has only the six rigid-body modes.
    #[test]
    fn whole_structure_is_one_rigid_block() {
        let atoms = cluster6();
        let modes = NormalModes::with_blocks(&atoms, &[0; 6], &rtb_params()).unwrap();
        assert_eq!(modes.len(), 6);
        for &v in modes.eigenvalues() {
            assert!(v.abs() < 1e-6);
        }
    }

    /// DOF accounting: a 3-atom block (6 DOF) plus a singleton (3 DOF) ⇒ nb6 = 9.
    #[test]
    fn dof_accounting_mixes_block_sizes() {
        let atoms = &cluster6()[..4];
        let modes = NormalModes::with_blocks(atoms, &[0, 0, 0, 1], &rtb_params()).unwrap();
        assert_eq!(modes.len(), 9);
    }

    /// `PᵀP = I`: since the reduced eigenvectors are orthonormal, the lifted
    /// all-atom modes stay unit length.
    #[test]
    fn lifted_modes_are_unit_norm() {
        let atoms = cluster6();
        let modes = NormalModes::with_blocks(&atoms, &[0, 0, 0, 1, 1, 1], &rtb_params()).unwrap();
        for i in 0..modes.len() {
            let norm2: f64 = modes.eigenvector(i).iter().flatten().map(|x| x * x).sum();
            assert_relative_eq!(norm2, 1.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn block_count_must_match_atoms() {
        let atoms = cluster6();
        let r = NormalModes::with_blocks(&atoms, &[0, 0], &rtb_params());
        assert!(matches!(r, Err(Error::BlockCountMismatch)));
    }

    #[test]
    fn collinear_block_is_degenerate() {
        // Block 0 holds two atoms — always collinear, so no rotational basis.
        let atoms = cluster6();
        let r = NormalModes::with_blocks(&atoms, &[0, 0, 1, 1, 1, 2], &rtb_params());
        assert!(matches!(r, Err(Error::DegenerateBlock)));
    }

    // --- Disconnected atoms (degree 0) ---

    /// An isolated atom is dropped: it is reported, contributes nothing to any
    /// mode, and the kept spectrum keeps only its six rigid-body modes.
    #[test]
    fn isolated_atom_is_dropped() {
        let mut atoms = cluster6();
        atoms.push(carbon(100.0, 100.0, 100.0)); // no neighbour within cutoff
        let modes = NormalModes::new(&atoms, &rtb_params()).unwrap();

        assert_eq!(modes.disconnected(), &[6]);
        assert_eq!(modes.len(), 18); // 6 connected atoms × 3, not 21
        for i in 0..modes.len() {
            assert_eq!(modes.eigenvector(i)[6], [0.0, 0.0, 0.0]);
        }
        let zeros = modes
            .eigenvalues()
            .iter()
            .filter(|&&v| v.abs() < ZERO_EIGENVALUE)
            .count();
        assert_eq!(zeros, 6); // not 9 — the isolated atom's spurious modes are gone

        // The kept spectrum equals solving the six connected atoms alone.
        let reference = NormalModes::new(&cluster6(), &rtb_params()).unwrap();
        for (a, b) in modes.eigenvalues().iter().zip(reference.eigenvalues()) {
            assert_relative_eq!(a, b, epsilon = 1e-9);
        }
    }

    /// RTB drops the isolated atom's block and remaps the rest: the two real
    /// blocks survive and match the same blocks solved without the dummy.
    #[test]
    fn isolated_atom_is_dropped_with_blocks() {
        let mut atoms = cluster6();
        atoms.push(carbon(100.0, 100.0, 100.0));
        // Two 3-atom blocks over the cluster; the isolated atom is its own block.
        let modes =
            NormalModes::with_blocks(&atoms, &[0, 0, 0, 1, 1, 1, 2], &rtb_params()).unwrap();

        assert_eq!(modes.disconnected(), &[6]);
        assert_eq!(modes.len(), 12); // two 6-DOF blocks; the dummy block is gone
        for i in 0..modes.len() {
            assert_eq!(modes.eigenvector(i)[6], [0.0, 0.0, 0.0]);
        }
        let reference =
            NormalModes::with_blocks(&cluster6(), &[0, 0, 0, 1, 1, 1], &rtb_params()).unwrap();
        for (a, b) in modes.eigenvalues().iter().zip(reference.eigenvalues()) {
            assert_relative_eq!(a, b, epsilon = 1e-9);
        }
    }

    /// If every atom is isolated (nothing within cutoff) there is no network.
    #[test]
    fn all_atoms_disconnected_is_too_few() {
        let atoms = [carbon(0.0, 0.0, 0.0), carbon(100.0, 0.0, 0.0)];
        assert!(matches!(
            NormalModes::new(&atoms, &rtb_params()),
            Err(Error::TooFewAtoms)
        ));
    }

    /// Without the `sparse` feature, requesting `k_modes` is an explicit error
    /// rather than a silent dense solve — on both constructors.
    #[cfg(not(feature = "sparse"))]
    #[test]
    fn k_modes_requires_sparse_feature() {
        let atoms = cluster6();
        let mut params = rtb_params();
        params.k_modes = Some(2);
        assert!(matches!(
            NormalModes::new(&atoms, &params),
            Err(Error::SparseFeatureRequired)
        ));
        assert!(matches!(
            NormalModes::with_blocks(&atoms, &[0, 0, 0, 1, 1, 1], &params),
            Err(Error::SparseFeatureRequired)
        ));
    }
}
