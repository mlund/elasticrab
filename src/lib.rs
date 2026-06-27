//! Anisotropic Network Model (ANM) normal-mode analysis.
//!
//! Give it atoms, get back the vibrational modes of an elastic network: a
//! coarse spring model where every pair of atoms closer than a cutoff is joined
//! by a harmonic spring. Diagonalizing the resulting Hessian yields the normal
//! modes — the collective, low-energy motions a structure most readily makes.
//!
//! ```
//! use elasticrab::{Atom, NormalModes};
//!
//! let atoms = vec![
//!     Atom { position: [0.0, 0.0, 0.0], mass: 12.0 },
//!     Atom { position: [3.8, 0.0, 0.0], mass: 12.0 },
//!     Atom { position: [3.8, 3.8, 0.0], mass: 12.0 },
//! ];
//! let modes = NormalModes::builder(&atoms).cutoff(15.0).solve().unwrap();
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
//! ideal for small and medium systems. [`Builder::k_modes`] returns only the
//! lowest *k* non-zero modes; the optional `sparse` feature then computes them
//! without forming the dense Hessian, which is what scales to large systems.

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

use nalgebra::{DMatrix, Rotation3, Unit, Vector3};

use network::Contact;
use rtb::BlockGeometry;

/// A point mass in the elastic network.
///
/// `mass` is in arbitrary units and is **ignored** unless mass-weighting is
/// enabled (see [`Builder::mass_weighted`]); the default analysis treats every
/// atom equally, matching the conventional ANM.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Atom {
    /// Cartesian coordinates, in ångström.
    pub position: [f64; 3],
    /// Atomic mass; only used when mass-weighting is enabled.
    pub mass: f64,
}

/// Internal solve configuration assembled by [`Builder`] (the connectivity lives
/// separately, in the `Springs` value the builder constructs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Params {
    /// Uniform spring constant `γ₀`; the effective spring constant of edge `ij` is
    /// `gamma · weight`. It scales every eigenvalue, so it sets only the overall
    /// scale, not the mode shapes or eigenvalue ratios.
    pub gamma: f64,
    /// When true, diagonalize the mass-weighted Hessian `M^{-1/2} H M^{-1/2}`
    /// instead of `H`; eigenvalues are then squared frequencies `ω²`.
    pub mass_weighted: bool,
    /// Number of lowest *non-zero* modes to compute; `None` returns all (including
    /// the ~6 rigid-body modes). `Some(k)` returns exactly the `k` lowest non-zero
    /// modes; the `sparse` feature computes those without forming the dense Hessian.
    pub k_modes: Option<usize>,
}

impl Default for Params {
    fn default() -> Self {
        Self {
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
    /// The block list given to [`Builder::blocks`] did not have one entry per atom.
    BlockCountMismatch,
    /// A multi-atom block is rank-deficient (collinear or coincident atoms), so
    /// it has no well-defined rotational basis. Use a single-atom block, or
    /// blocks of three or more non-collinear atoms.
    DegenerateBlock,
    /// The sparse solver could not factor the Hessian or did not converge.
    SparseSolverFailed,
    /// [`NormalModes::displace_nonlinear`] was called on a result that has no
    /// rigid blocks — it needs the per-block velocities only [`Builder::blocks`]
    /// retains. Build the modes with [`Builder::blocks`].
    NotRigidBlocks,
    /// [`Builder::solve`] was called without connectivity — set a
    /// [`cutoff`](Builder::cutoff) or explicit [`springs`](Builder::springs).
    NoNetwork,
    /// An explicit [`Spring`] referenced an atom out of range, or itself.
    InvalidSpring,
    /// Two connected atoms occupy (almost) the same position, giving a
    /// zero-length spring the Hessian's `1/d²` term cannot handle — usually a
    /// retained alternate-location or duplicate atom record.
    CoincidentAtoms,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewAtoms => write!(f, "at least two atoms are required"),
            Self::NotFinite => write!(f, "non-finite coordinate or mass"),
            Self::BlockCountMismatch => write!(f, "blocks must have one entry per atom"),
            Self::DegenerateBlock => write!(f, "a multi-atom block is collinear or coincident"),
            Self::SparseSolverFailed => write!(f, "the sparse solver failed"),
            Self::NotRigidBlocks => write!(f, "nonlinear modes require rigid blocks"),
            Self::NoNetwork => write!(f, "no network: set a cutoff or springs"),
            Self::InvalidSpring => write!(f, "a spring references an invalid atom"),
            Self::CoincidentAtoms => write!(f, "two connected atoms share a position"),
        }
    }
}

impl std::error::Error for Error {}

/// One spring of an explicit elastic network: two atoms (by index into the
/// `atoms` slice) and a relative stiffness `weight`. Pass these to
/// [`Builder::springs`] — e.g. Voronoi contacts weighted by area. The effective
/// spring constant is `gamma · weight`, and the rest length is the atoms'
/// equilibrium separation, so only the connectivity and weight are given here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spring {
    /// Index of the first atom, into the `atoms` slice.
    pub i: usize,
    /// Index of the second atom, into the `atoms` slice.
    pub j: usize,
    /// Relative stiffness; the spring constant is `gamma · weight`.
    pub weight: f64,
}

/// One spring of the *built* network, kept so [`energy`](NormalModes::energy) can
/// score any conformation: the atom pair (by *original* index), the equilibrium
/// rest length, and the relative stiffness weight.
#[derive(Debug)]
struct Bond {
    i: usize,
    j: usize,
    rest: f64,
    weight: f64,
}

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
    /// Rigid-block decomposition, kept only for rigid-block (RTB) results; it
    /// carries the per-block velocities the nonlinear extrapolation needs.
    rtb: Option<Rtb>,
    /// Equilibrium springs and the global constant, kept so
    /// [`energy`](NormalModes::energy) can score any conformation without
    /// rebuilding the network.
    springs: Vec<Bond>,
    gamma: f64,
}

/// The data the nonlinear extrapolation needs beyond the per-atom mode shapes:
/// the reduced (block-space) eigenvectors and each block's rigid geometry.
#[derive(Debug)]
struct Rtb {
    /// Reduced eigenvectors, `nb6 × n_modes`; column `i` holds mode `i`'s
    /// per-block linear and angular velocities in the orthonormal RTB basis.
    reduced: DMatrix<f64>,
    /// Per-block geometry, with atom indices in the *original* numbering.
    blocks: Vec<BlockGeometry>,
}

/// Eigenvalues at or below this magnitude are treated as rigid-body (zero)
/// modes when deriving thermal amplitudes, which guards the `1/ω` against the
/// tiny positive *and negative* values a finite-precision solver returns.
const ZERO_EIGENVALUE: f64 = 1e-6;

/// Molar gas constant R = N_A·k_B, in kJ·mol⁻¹·K⁻¹, matching a `gamma` expressed in
/// kJ·mol⁻¹·Å⁻²: energies are per-mole, so amplitudes and fluctuations use RT, not
/// the per-particle kT. Their absolute scale is only meaningful relative to `gamma`,
/// so callers commonly rescale regardless.
const GAS_CONSTANT_KJ_PER_MOL_K: f64 = 8.314_462_618e-3;

/// Below this angular speed a block's nonlinear motion is treated as a pure
/// translation, which also guards the rotation-axis normalization against a
/// near-zero vector.
const ROTATION_EPS: f64 = 1e-12;

/// A spring shorter than this (squared, Å²) is treated as a zero-length spring
/// between coincident atoms — `1e-6 Å²` is a 0.001 Å separation, far below any
/// real interatomic distance, so this only fires on duplicate/altloc artifacts.
const MIN_CONTACT_DIST_SQ: f64 = 1e-6;

/// Shared validation: per-atom positions and raw masses. Coordinates must be
/// finite; masses are checked finite-and-positive only when `mass_weighted`,
/// since that is the only path that takes their square root.
fn validated_inputs(
    atoms: &[Atom],
    mass_weighted: bool,
) -> Result<(Vec<[f64; 3]>, Vec<f64>), Error> {
    if atoms.len() < 2 {
        return Err(Error::TooFewAtoms);
    }
    let positions: Vec<[f64; 3]> = atoms.iter().map(|a| a.position).collect();
    if positions.iter().flatten().any(|x| !x.is_finite()) {
        return Err(Error::NotFinite);
    }
    if mass_weighted && atoms.iter().any(|a| !(a.mass.is_finite() && a.mass > 0.0)) {
        return Err(Error::NotFinite);
    }
    let masses = atoms.iter().map(|a| a.mass).collect();
    Ok((positions, masses))
}

/// The per-DOF mass-weighting weights for a solve: the kept atoms' masses when
/// mass-weighting (already validated positive), otherwise unit.
fn solve_weights(net: &Network, params: &Params) -> Vec<f64> {
    if params.mass_weighted {
        net.masses.clone()
    } else {
        vec![1.0; net.keep.len()]
    }
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

/// Indices of the lowest `k` non-zero (non-rigid-body) modes of an
/// ascending-sorted dense spectrum — the dense fallback for `k_modes` when the
/// `sparse` partial solver is not compiled in. The rigid-body zeros are the
/// leading entries, so the wanted modes are the contiguous tail above them.
fn lowest_nonzero_columns(eigenvalues: &[f64], k: usize) -> Vec<usize> {
    eigenvalues
        .iter()
        .enumerate()
        .filter(|(_, &lambda)| lambda > ZERO_EIGENVALUE)
        .map(|(column, _)| column)
        .take(k)
        .collect()
}

/// A new matrix holding the given columns of `m`, in order.
fn pick_columns(m: &DMatrix<f64>, columns: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(m.nrows(), columns.len(), |r, idx| m[(r, columns[idx])])
}

/// The connected elastic network, ready to solve: positions, raw masses, and
/// weighted contacts renumbered to the atoms a spring actually touches. `keep[p]`
/// is the original index of kept atom `p`; `disconnected` lists the degree-0 atoms
/// that were removed (empty for a fully connected structure).
struct Network {
    positions: Vec<[f64; 3]>,
    masses: Vec<f64>,
    contacts: Vec<Contact>,
    keep: Vec<usize>,
    disconnected: Vec<usize>,
}

/// Validate the atoms, build the contacts (cutoff or explicit springs), and drop
/// disconnected atoms. Fails if fewer than two atoms remain connected.
fn build_network(
    atoms: &[Atom],
    cutoff: Option<f64>,
    springs: Option<&[Spring]>,
    mass_weighted: bool,
) -> Result<Network, Error> {
    let (positions, masses) = validated_inputs(atoms, mass_weighted)?;
    let contacts = match (cutoff, springs) {
        (Some(c), None) => network::contacts(&positions, c),
        (None, Some(s)) => network::contacts_from_edges(&positions, s)?,
        _ => return Err(Error::NoNetwork),
    };
    // A zero-length spring makes the Hessian's `-γ/d²` term blow up to NaN, so
    // reject coincident atoms rather than emit a spectrum of NaNs.
    if contacts.iter().any(|c| c.dist2 < MIN_CONTACT_DIST_SQ) {
        return Err(Error::CoincidentAtoms);
    }
    let net = drop_disconnected(positions, masses, contacts);
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
    masses: Vec<f64>,
    contacts: Vec<Contact>,
) -> Network {
    let n = positions.len();
    let disconnected = network::disconnected_atoms(n, &contacts);
    if disconnected.is_empty() {
        return Network {
            positions,
            masses,
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
            weight: c.weight,
        })
        .collect();
    Network {
        positions: keep.iter().map(|&old| positions[old]).collect(),
        masses: keep.iter().map(|&old| masses[old]).collect(),
        contacts,
        keep,
        disconnected,
    }
}

impl NormalModes {
    /// Start configuring a normal-mode analysis over `atoms`. Set the network
    /// ([`cutoff`](Builder::cutoff) or [`springs`](Builder::springs)) and any
    /// options, then call [`solve`](Builder::solve).
    ///
    /// ```
    /// # use elasticrab::{Atom, NormalModes};
    /// # let atoms = [Atom{position:[0.0;3],mass:12.0}, Atom{position:[3.8,0.0,0.0],mass:12.0}];
    /// let modes = NormalModes::builder(&atoms).cutoff(15.0).solve()?;
    /// # Ok::<(), elasticrab::Error>(())
    /// ```
    pub fn builder(atoms: &[Atom]) -> Builder<'_> {
        Builder::new(atoms)
    }

    /// Assemble the all-atom Hessian and diagonalize it. Heavy and fallible by
    /// design — it forms a `3N×3N` matrix and runs a symmetric eigendecomposition.
    fn solve_all_atom(net: &Network, params: &Params, n_original: usize) -> Result<Self, Error> {
        // The `sparse` feature computes only the lowest `k` modes directly; without
        // it `k_modes` still works, via a full dense solve truncated to those modes.
        #[cfg(feature = "sparse")]
        if let Some(k) = params.k_modes {
            return Self::solve_partial(net, params, k, n_original);
        }

        let weights = solve_weights(net, params);
        let h = build_hessian(net.keep.len(), &weights, &net.contacts, params);
        let spectrum = eigen::solve(h);
        Ok(match params.k_modes {
            None => Self::from_modes(
                spectrum.eigenvalues,
                &spectrum.eigenvectors,
                net,
                n_original,
                params.gamma,
            ),
            Some(k) => {
                let columns = lowest_nonzero_columns(&spectrum.eigenvalues, k);
                let eigenvalues = columns.iter().map(|&c| spectrum.eigenvalues[c]).collect();
                let vectors = pick_columns(&spectrum.eigenvectors, &columns);
                Self::from_modes(eigenvalues, &vectors, net, n_original, params.gamma)
            }
        })
    }

    /// The `k` lowest non-zero modes via the sparse partial solver.
    #[cfg(feature = "sparse")]
    fn solve_partial(
        net: &Network,
        params: &Params,
        k: usize,
        n_original: usize,
    ) -> Result<Self, Error> {
        let weights = solve_weights(net, params);
        let (eigenvalues, vectors) =
            sparse::lowest_nonzero_modes(net.keep.len(), params.gamma, &weights, &net.contacts, k)?;
        Ok(Self::from_modes(
            eigenvalues,
            &vectors,
            net,
            n_original,
            params.gamma,
        ))
    }

    /// Solve the rigid-block (RTB) reduced eigenproblem. `blocks` is parallel to
    /// the original atoms; the modes are the same per-atom fields as the all-atom
    /// solve, lifted back from the reduced space.
    fn solve_blocks(
        net: &Network,
        blocks: &[usize],
        params: &Params,
        n_original: usize,
    ) -> Result<Self, Error> {
        if blocks.len() != n_original {
            return Err(Error::BlockCountMismatch);
        }
        // The drop carries the blocks along: a block keeps only its connected atoms.
        let blocks: Vec<usize> = net.keep.iter().map(|&old| blocks[old]).collect();
        let weights = solve_weights(net, params);

        #[cfg(feature = "sparse")]
        if let Some(k) = params.k_modes {
            return Self::solve_rtb_partial(net, &blocks, &weights, params, k, n_original);
        }

        let h = build_hessian(net.keep.len(), &weights, &net.contacts, params);
        // Reduce to the block subspace and solve there; `tr_mul` forms Pᵀ·(H·P)
        // without materializing the transpose of P.
        let p = rtb::projection(&net.positions, &weights, &blocks)?;
        let reduced_hessian = p.tr_mul(&(&h * &p));
        let spectrum = eigen::solve(reduced_hessian);
        // Keep all modes, or — the dense `k_modes` fallback — the lowest k non-zero.
        let (eigenvalues, reduced) = match params.k_modes {
            None => (spectrum.eigenvalues, spectrum.eigenvectors),
            Some(k) => {
                let columns = lowest_nonzero_columns(&spectrum.eigenvalues, k);
                let eigenvalues = columns.iter().map(|&c| spectrum.eigenvalues[c]).collect();
                (eigenvalues, pick_columns(&spectrum.eigenvectors, &columns))
            }
        };
        // Lift the reduced modes back with P, and keep them for nonlinear modes.
        let all_atom = &p * &reduced;
        let rtb = Self::build_rtb(net, &blocks, &weights, reduced)?;
        let mut modes = Self::from_modes(eigenvalues, &all_atom, net, n_original, params.gamma);
        modes.rtb = Some(rtb);
        Ok(modes)
    }

    /// The `k` lowest non-zero RTB modes via the matrix-free partial solver.
    #[cfg(feature = "sparse")]
    fn solve_rtb_partial(
        net: &Network,
        blocks: &[usize],
        weights: &[f64],
        params: &Params,
        k: usize,
        n_original: usize,
    ) -> Result<Self, Error> {
        let (eigenvalues, vectors, reduced) = sparse::lowest_rtb_modes(
            &net.positions,
            weights,
            blocks,
            params.gamma,
            &net.contacts,
            k,
        )?;
        let rtb = Self::build_rtb(net, blocks, weights, reduced)?;
        let mut modes = Self::from_modes(eigenvalues, &vectors, net, n_original, params.gamma);
        modes.rtb = Some(rtb);
        Ok(modes)
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
        gamma: f64,
    ) -> Self {
        let mut modes = vec![[0.0; 3]; eigenvalues.len() * n_original];
        for (m, col) in vectors.column_iter().enumerate() {
            let base = m * n_original;
            for (p, &orig) in net.keep.iter().enumerate() {
                modes[base + orig] = [col[3 * p], col[3 * p + 1], col[3 * p + 2]];
            }
        }
        // Record each spring in original atom indices with its rest length and
        // weight, so `energy` can score later conformations directly.
        let springs = net
            .contacts
            .iter()
            .map(|c| Bond {
                i: net.keep[c.i],
                j: net.keep[c.j],
                rest: c.dist2.sqrt(),
                weight: c.weight,
            })
            .collect();
        Self {
            eigenvalues,
            modes,
            n_atoms: n_original,
            disconnected: net.disconnected.clone(),
            rtb: None,
            springs,
            gamma,
        }
    }

    /// Build the rigid-block decomposition kept by the RTB constructors: the
    /// reduced eigenvectors plus per-block geometry, with block atoms remapped
    /// from the connected numbering back to the original atom indices.
    fn build_rtb(
        net: &Network,
        blocks: &[usize],
        weights: &[f64],
        reduced: DMatrix<f64>,
    ) -> Result<Rtb, Error> {
        let mut geometry = rtb::block_geometry(&net.positions, weights, blocks)?;
        for block in &mut geometry {
            for atom in &mut block.atoms {
                *atom = net.keep[*atom];
            }
        }
        Ok(Rtb {
            reduced,
            blocks: geometry,
        })
    }

    /// Number of modes: three per connected atom for the plain model, or the
    /// reduced rigid-block degree-of-freedom count for a rigid-block (RTB) solve.
    pub const fn len(&self) -> usize {
        self.eigenvalues.len()
    }

    /// Whether there are no modes — always false for a successful solve, provided
    /// as the conventional companion to [`len`](Self::len).
    pub const fn is_empty(&self) -> bool {
        self.eigenvalues.is_empty()
    }

    /// Number of springs in the network — the edges that survived the
    /// disconnected-atom drop. Useful for comparing connectivity between a
    /// distance cutoff and a tessellation network.
    pub const fn spring_count(&self) -> usize {
        self.springs.len()
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

    /// New positions with the atoms pushed along mode `i` by `amplitude`:
    /// `xₐ + amplitude · vᵢ(a)` for each atom `a`. Disconnected atoms (zero in the
    /// mode) stay put. Sweep `amplitude` and write each result as a frame to
    /// animate the mode — see the `animate_pdb` example.
    ///
    /// This is the *linear* displacement; at large amplitude it stretches bonds.
    /// For the bond-preserving variant, see
    /// [`displace_nonlinear`](Self::displace_nonlinear).
    ///
    /// # Panics
    /// If `i >= self.len()`, or `positions.len()` is not the original atom count.
    pub fn displace(&self, positions: &[[f64; 3]], i: usize, amplitude: f64) -> Vec<[f64; 3]> {
        let mode = self.eigenvector(i);
        assert_eq!(
            positions.len(),
            mode.len(),
            "positions must have one entry per atom"
        );
        positions
            .iter()
            .zip(mode)
            .map(|(x, v)| {
                [
                    x[0] + amplitude * v[0],
                    x[1] + amplitude * v[1],
                    x[2] + amplitude * v[2],
                ]
            })
            .collect()
    }

    /// New positions with each rigid block carried along mode `i` by `amplitude`
    /// as a **rigid motion** (NOLB's nonlinear extrapolation): the block rotates
    /// about its centre of mass and translates, so bond lengths *within* a block
    /// are preserved at any amplitude — unlike [`displace`](Self::displace), whose
    /// straight-line motion stretches them. This is the physical motion, so it is
    /// independent of mass-weighting and agrees with `displace` only at small
    /// amplitude (and only on the unit-mass path).
    ///
    /// Disconnected atoms stay put.
    ///
    /// # Errors
    /// [`Error::NotRigidBlocks`] if the modes were not built with rigid blocks
    /// (no per-block velocities to extrapolate). Use [`Builder::blocks`].
    ///
    /// # Panics
    /// If `i >= self.len()`, or `positions.len()` is not the original atom count.
    pub fn displace_nonlinear(
        &self,
        positions: &[[f64; 3]],
        i: usize,
        amplitude: f64,
    ) -> Result<Vec<[f64; 3]>, Error> {
        let rtb = self.rtb.as_ref().ok_or(Error::NotRigidBlocks)?;
        assert!(i < self.len(), "mode index out of range");
        assert_eq!(
            positions.len(),
            self.n_atoms,
            "positions must have one entry per atom"
        );

        let mut out = positions.to_vec();
        for block in &rtb.blocks {
            // Un-weight the reduced velocities to physical ones (NOLB eq 1.11):
            // v = ṽ_w/√M_b, ω = I^{-1/2}·ω̃_w. A singleton has no rotation.
            let col = block.col;
            let velocity = Vector3::new(
                rtb.reduced[(col, i)],
                rtb.reduced[(col + 1, i)],
                rtb.reduced[(col + 2, i)],
            );
            let translation = amplitude * (velocity / block.total_mass.sqrt());
            // The block's rotation about its COM by Δφ = a‖ω‖ about ω̂ is the same
            // for every atom, so build it once; a singleton or vanishing rotation
            // leaves only the translation.
            let rotation = block.isqrt.and_then(|isqrt| {
                let omega = isqrt
                    * Vector3::new(
                        rtb.reduced[(col + 3, i)],
                        rtb.reduced[(col + 4, i)],
                        rtb.reduced[(col + 5, i)],
                    );
                let speed = omega.norm();
                (speed > ROTATION_EPS).then(|| {
                    Rotation3::from_axis_angle(&Unit::new_normalize(omega), amplitude * speed)
                })
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
        Ok(out)
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
        let two_rt = 2.0 * GAS_CONSTANT_KJ_PER_MOL_K * temperature_k;
        self.eigenvalues
            .iter()
            .map(|&lambda| {
                if lambda > ZERO_EIGENVALUE {
                    (two_rt / lambda).sqrt()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Predicted thermal fluctuation `⟨Δrₐ²⟩ = RT Σᵢ (1/λᵢ) |vᵢ(a)|²` of each
    /// atom at temperature `T` (kelvin), summed over the non-zero modes — the
    /// quantity behind crystallographic B-factors, `B = (8π²/3)⟨Δr²⟩`. (`RT`, not
    /// `kT`: the energies are molar, matching γ in kJ/mol/Å².)
    ///
    /// These are *configurational* fluctuations: independent of mass, so for
    /// physical B-factors build the modes **without** mass-weighting
    /// (without [`Builder::mass_weighted`]). The result is one value per original
    /// atom; a disconnected atom (zero in every mode) scores 0. With γ in
    /// kJ/mol/Å² the values are in Å².
    pub fn fluctuations(&self, temperature_k: f64) -> Vec<f64> {
        let rt = GAS_CONSTANT_KJ_PER_MOL_K * temperature_k;
        let mut msf = vec![0.0; self.n_atoms];
        for (i, &lambda) in self.eigenvalues.iter().enumerate() {
            if lambda <= ZERO_EIGENVALUE {
                continue;
            }
            let weight = rt / lambda;
            for (out, v) in msf.iter_mut().zip(self.eigenvector(i)) {
                *out += weight * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            }
        }
        msf
    }

    /// Predicted crystallographic B-factors `B = (8π²/3)⟨Δr²⟩` per atom at
    /// temperature `T`, from [`fluctuations`](Self::fluctuations) — the standard
    /// observable for comparing against, or calibrating `gamma` to, an
    /// experimental structure. Configurational like the fluctuations, so build the
    /// modes **without** mass-weighting; in Å² when γ is in kJ/mol/Å².
    pub fn predicted_b_factors(&self, temperature_k: f64) -> Vec<f64> {
        const PREFACTOR: f64 = 8.0 * std::f64::consts::PI * std::f64::consts::PI / 3.0;
        self.fluctuations(temperature_k)
            .into_iter()
            .map(|msf| PREFACTOR * msf)
            .collect()
    }

    /// Elastic-network energy of a conformation: `½γ Σ (|r_ij| − r⁰_ij)²` over the
    /// springs, in the energy units of `gamma` (kJ/mol when γ is in kJ/mol/Å²).
    ///
    /// This is the potential whose Boltzmann factor `exp(−E / RT)` weights a
    /// conformation — the quantity to reweight Monte-Carlo moves between
    /// structures sampled from [`displace`](Self::displace) /
    /// [`displace_nonlinear`](Self::displace_nonlinear). It depends only on the
    /// coordinates, not on masses or on which mode produced them, so energies from
    /// different modes are directly comparable. The input structure scores 0, as
    /// does any rigid-body motion of the whole structure; a disconnected atom has
    /// no spring and never contributes.
    ///
    /// # Panics
    /// If `positions.len()` is not the original atom count.
    pub fn energy(&self, positions: &[[f64; 3]]) -> f64 {
        assert_eq!(
            positions.len(),
            self.n_atoms,
            "positions must have one entry per atom"
        );
        let sum: f64 = self
            .springs
            .iter()
            .map(|s| {
                let p = positions[s.i];
                let q = positions[s.j];
                let distance =
                    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
                s.weight * (distance - s.rest).powi(2)
            })
            .sum();
        0.5 * self.gamma * sum
    }
}

/// Fluent configuration for a normal-mode solve, returned by
/// [`NormalModes::builder`]. Set connectivity ([`cutoff`](Self::cutoff) or
/// [`springs`](Self::springs)) and options, then [`solve`](Self::solve).
pub struct Builder<'a> {
    atoms: &'a [Atom],
    cutoff: Option<f64>,
    springs: Option<&'a [Spring]>,
    blocks: Option<&'a [usize]>,
    params: Params,
}

impl<'a> Builder<'a> {
    fn new(atoms: &'a [Atom]) -> Self {
        Self {
            atoms,
            cutoff: None,
            springs: None,
            blocks: None,
            params: Params::default(),
        }
    }

    /// Connect every pair of atoms within `cutoff` ångström by a uniform spring.
    /// Mutually exclusive with [`springs`](Self::springs).
    #[must_use]
    pub const fn cutoff(mut self, cutoff: f64) -> Self {
        self.cutoff = Some(cutoff);
        self
    }

    /// Use an explicit list of weighted springs (e.g. area-weighted Voronoi
    /// contacts) instead of a cutoff. Mutually exclusive with
    /// [`cutoff`](Self::cutoff).
    #[must_use]
    pub const fn springs(mut self, springs: &'a [Spring]) -> Self {
        self.springs = Some(springs);
        self
    }

    /// Global spring constant `γ₀` (default `1.0`); the spring constant of edge `ij`
    /// is `γ₀ · weight`. It scales every eigenvalue, setting only the overall scale.
    #[must_use]
    pub const fn gamma(mut self, gamma: f64) -> Self {
        self.params.gamma = gamma;
        self
    }

    /// Diagonalize the mass-weighted Hessian; eigenvalues become squared
    /// frequencies `ω²` (default: off, the plain ANM).
    #[must_use]
    pub const fn mass_weighted(mut self) -> Self {
        self.params.mass_weighted = true;
        self
    }

    /// Return only the `k` lowest non-zero modes (default: all). The `sparse`
    /// feature computes them without forming the dense Hessian.
    #[must_use]
    pub const fn k_modes(mut self, k: usize) -> Self {
        self.params.k_modes = Some(k);
        self
    }

    /// Group atoms into rigid blocks (Rotation-Translation Blocks): one block id
    /// per atom, parallel to `atoms`. Shrinks the eigenproblem and enables the
    /// nonlinear extrapolation; the modes are still per-atom fields.
    #[must_use]
    pub const fn blocks(mut self, blocks: &'a [usize]) -> Self {
        self.blocks = Some(blocks);
        self
    }

    /// Build the network and solve. Errors include [`Error::NoNetwork`] (no
    /// `cutoff`/`springs` set), [`Error::TooFewAtoms`], and [`Error::InvalidSpring`].
    pub fn solve(self) -> Result<NormalModes, Error> {
        let net = build_network(
            self.atoms,
            self.cutoff,
            self.springs,
            self.params.mass_weighted,
        )?;
        let n = self.atoms.len();
        match self.blocks {
            Some(blocks) => NormalModes::solve_blocks(&net, blocks, &self.params, n),
            None => NormalModes::solve_all_atom(&net, &self.params, n),
        }
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
        let r = NormalModes::builder(&[carbon(0.0, 0.0, 0.0)])
            .cutoff(15.0)
            .solve();
        assert!(matches!(r, Err(Error::TooFewAtoms)));
    }

    #[test]
    fn non_finite_coordinate_is_rejected() {
        let atoms = [carbon(0.0, 0.0, 0.0), carbon(f64::NAN, 0.0, 0.0)];
        let r = NormalModes::builder(&atoms).cutoff(15.0).solve();
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
        let modes = NormalModes::builder(&atoms).cutoff(15.0).solve().unwrap();

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
        let modes = NormalModes::builder(&atoms).cutoff(15.0).solve().unwrap();
        let amps = modes.thermal_amplitudes(300.0);

        assert_eq!(amps.len(), modes.len());
        // First six modes are rigid-body -> zero amplitude.
        for &a in &amps[..6] {
            assert_relative_eq!(a, 0.0);
        }
        // Softer modes (smaller eigenvalue) fluctuate more than stiffer ones.
        assert!(amps[6] >= amps[7]);
    }

    fn positions(atoms: &[Atom]) -> Vec<[f64; 3]> {
        atoms.iter().map(|a| a.position).collect()
    }

    fn cluster() -> [Atom; 4] {
        [
            carbon(0.0, 0.0, 0.0),
            carbon(1.5, 0.0, 0.0),
            carbon(0.0, 1.5, 0.0),
            carbon(1.0, 1.0, 1.0),
        ]
    }

    #[test]
    fn energy_of_the_input_structure_is_zero() {
        let atoms = cluster();
        let modes = NormalModes::builder(&atoms).cutoff(15.0).solve().unwrap();
        assert_relative_eq!(modes.energy(&positions(&atoms)), 0.0);
    }

    #[test]
    fn energy_is_zero_under_rigid_translation() {
        let atoms = cluster();
        let modes = NormalModes::builder(&atoms).cutoff(15.0).solve().unwrap();
        let shifted: Vec<_> = positions(&atoms)
            .iter()
            .map(|p| [p[0] + 5.0, p[1] - 3.0, p[2] + 1.0])
            .collect();
        assert_relative_eq!(modes.energy(&shifted), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn energy_is_zero_under_rigid_rotation() {
        let atoms = cluster();
        let modes = NormalModes::builder(&atoms).cutoff(15.0).solve().unwrap();
        // A 90° turn about z, (x, y, z) -> (-y, x, z), preserves every distance.
        let rotated: Vec<_> = positions(&atoms)
            .iter()
            .map(|p| [-p[1], p[0], p[2]])
            .collect();
        assert_relative_eq!(modes.energy(&rotated), 0.0, epsilon = 1e-9);
    }

    /// A stretched diatomic has the closed-form energy `½γΔ²`.
    #[test]
    fn diatomic_stretch_energy_is_half_gamma_delta_squared() {
        let atoms = [carbon(0.0, 0.0, 0.0), carbon(3.8, 0.0, 0.0)];
        let modes = NormalModes::builder(&atoms).cutoff(15.0).solve().unwrap();
        let delta = 0.4;
        let stretched = [[0.0, 0.0, 0.0], [3.8 + delta, 0.0, 0.0]];
        assert_relative_eq!(
            modes.energy(&stretched),
            0.5 * delta * delta,
            epsilon = 1e-12
        );
    }

    /// γ only rescales the energy, so doubling it doubles every conformation's score.
    #[test]
    fn energy_scales_linearly_with_gamma() {
        let atoms = cluster();
        let displaced = {
            let base = NormalModes::builder(&atoms).cutoff(15.0).solve().unwrap();
            base.displace(&positions(&atoms), 6, 0.7)
        };
        let unit = NormalModes::builder(&atoms).cutoff(15.0).solve().unwrap();
        let stiff = NormalModes::builder(&atoms)
            .cutoff(15.0)
            .gamma(3.0)
            .solve()
            .unwrap();
        assert_relative_eq!(
            stiff.energy(&displaced),
            3.0 * unit.energy(&displaced),
            epsilon = 1e-9
        );
    }

    /// A disconnected atom carries no spring, so moving it cannot change the energy.
    #[test]
    fn moving_a_disconnected_atom_leaves_energy_unchanged() {
        let atoms = [
            carbon(0.0, 0.0, 0.0),
            carbon(1.5, 0.0, 0.0),
            carbon(0.0, 1.5, 0.0),
            carbon(100.0, 100.0, 100.0), // far beyond the cutoff: dropped
        ];
        let modes = NormalModes::builder(&atoms).cutoff(15.0).solve().unwrap();
        assert_eq!(modes.disconnected(), &[3]);
        let mut moved = positions(&atoms);
        moved[3] = [200.0, -50.0, 7.0];
        assert_relative_eq!(modes.energy(&moved), 0.0);
    }

    #[test]
    fn a_displaced_mode_has_positive_energy() {
        let atoms = cluster();
        let modes = NormalModes::builder(&atoms).cutoff(15.0).solve().unwrap();
        let displaced = modes.displace(&positions(&atoms), 6, 0.5);
        assert!(modes.energy(&displaced) > ZERO_EIGENVALUE);
    }

    #[test]
    #[should_panic(expected = "one entry per atom")]
    fn energy_rejects_wrong_length_positions() {
        let atoms = cluster();
        let modes = NormalModes::builder(&atoms).cutoff(15.0).solve().unwrap();
        let _ = modes.energy(&[[0.0, 0.0, 0.0]]);
    }

    /// Eigenvalues scale with γ, so the fluctuations scale as 1/γ.
    #[test]
    fn fluctuations_scale_inversely_with_gamma() {
        let atoms = cluster();
        let unit = NormalModes::builder(&atoms).cutoff(15.0).solve().unwrap();
        let stiff = NormalModes::builder(&atoms)
            .cutoff(15.0)
            .gamma(2.0)
            .solve()
            .unwrap();
        for (u, s) in unit
            .fluctuations(300.0)
            .iter()
            .zip(stiff.fluctuations(300.0))
        {
            assert_relative_eq!(s, u / 2.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn fluctuations_are_zero_for_a_disconnected_atom() {
        let atoms = [
            carbon(0.0, 0.0, 0.0),
            carbon(1.5, 0.0, 0.0),
            carbon(0.0, 1.5, 0.0),
            carbon(100.0, 100.0, 100.0), // dropped
        ];
        let modes = NormalModes::builder(&atoms).cutoff(15.0).solve().unwrap();
        let msf = modes.fluctuations(300.0);
        assert_eq!(msf.len(), atoms.len());
        assert_relative_eq!(msf[3], 0.0);
        assert!(msf[0] > 0.0);
    }

    #[test]
    fn predicted_b_factors_are_fluctuations_times_the_prefactor() {
        let atoms = cluster();
        let modes = NormalModes::builder(&atoms).cutoff(15.0).solve().unwrap();
        let prefactor = 8.0 * std::f64::consts::PI * std::f64::consts::PI / 3.0;
        for (msf, b) in modes
            .fluctuations(300.0)
            .iter()
            .zip(modes.predicted_b_factors(300.0))
        {
            assert_relative_eq!(b, prefactor * msf);
        }
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
        let modes = NormalModes::builder(&atoms)
            .cutoff(5.0)
            .mass_weighted()
            .solve()
            .unwrap();

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
        let unit = NormalModes::builder(&atoms).cutoff(5.0).solve().unwrap();
        let scaled = NormalModes::builder(&atoms)
            .cutoff(5.0)
            .mass_weighted()
            .solve()
            .unwrap();
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

    /// A 5 Å rigid-block (RTB) solve — the helper the RTB tests build on.
    fn rtb(atoms: &[Atom], blocks: &[usize]) -> Result<NormalModes, Error> {
        NormalModes::builder(atoms)
            .cutoff(5.0)
            .blocks(blocks)
            .solve()
    }

    /// Each atom in its own block ⇒ the projection is the identity, so RTB must
    /// reproduce the plain ANM spectrum exactly. Ties RTB to the ProDy-validated path.
    #[test]
    fn all_singleton_blocks_match_plain_anm() {
        let atoms = cluster6();
        let blocks: Vec<usize> = (0..atoms.len()).collect();

        let plain = NormalModes::builder(&atoms).cutoff(5.0).solve().unwrap();
        let rtb = rtb(&atoms, &blocks).unwrap();

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
        let a = rtb(&atoms, &[0, 0, 0, 1, 1, 1]).unwrap();
        let b = rtb(&atoms, &[42, 42, 42, 7, 7, 7]).unwrap();
        for (x, y) in a.eigenvalues().iter().zip(b.eigenvalues()) {
            assert_relative_eq!(x, y, epsilon = 1e-12);
        }
    }

    /// Blocks are grouped by id regardless of atom order: interleaved ids put
    /// non-adjacent atoms in the same block and still yield two 6-DOF blocks.
    #[test]
    fn interleaved_blocks_are_grouped_by_id() {
        let atoms = cluster6();
        let modes = rtb(&atoms, &[0, 1, 0, 1, 0, 1]).unwrap();
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
        let modes = rtb(&atoms, &[0; 6]).unwrap();
        assert_eq!(modes.len(), 6);
        for &v in modes.eigenvalues() {
            assert!(v.abs() < 1e-6);
        }
    }

    /// DOF accounting: a 3-atom block (6 DOF) plus a singleton (3 DOF) ⇒ nb6 = 9.
    #[test]
    fn dof_accounting_mixes_block_sizes() {
        let atoms = &cluster6()[..4];
        let modes = rtb(atoms, &[0, 0, 0, 1]).unwrap();
        assert_eq!(modes.len(), 9);
    }

    /// `PᵀP = I`: since the reduced eigenvectors are orthonormal, the lifted
    /// all-atom modes stay unit length.
    #[test]
    fn lifted_modes_are_unit_norm() {
        let atoms = cluster6();
        let modes = rtb(&atoms, &[0, 0, 0, 1, 1, 1]).unwrap();
        for i in 0..modes.len() {
            let norm2: f64 = modes.eigenvector(i).iter().flatten().map(|x| x * x).sum();
            assert_relative_eq!(norm2, 1.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn block_count_must_match_atoms() {
        let atoms = cluster6();
        let r = rtb(&atoms, &[0, 0]);
        assert!(matches!(r, Err(Error::BlockCountMismatch)));
    }

    #[test]
    fn collinear_block_is_degenerate() {
        // Block 0 holds two atoms — always collinear, so no rotational basis.
        let atoms = cluster6();
        let r = rtb(&atoms, &[0, 0, 1, 1, 1, 2]);
        assert!(matches!(r, Err(Error::DegenerateBlock)));
    }

    // --- Disconnected atoms (degree 0) ---

    /// An isolated atom is dropped: it is reported, contributes nothing to any
    /// mode, and the kept spectrum keeps only its six rigid-body modes.
    #[test]
    fn isolated_atom_is_dropped() {
        let mut atoms = cluster6();
        atoms.push(carbon(100.0, 100.0, 100.0)); // no neighbour within cutoff
        let modes = NormalModes::builder(&atoms).cutoff(5.0).solve().unwrap();

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
        let reference = NormalModes::builder(&cluster6())
            .cutoff(5.0)
            .solve()
            .unwrap();
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
        let modes = rtb(&atoms, &[0, 0, 0, 1, 1, 1, 2]).unwrap();

        assert_eq!(modes.disconnected(), &[6]);
        assert_eq!(modes.len(), 12); // two 6-DOF blocks; the dummy block is gone
        for i in 0..modes.len() {
            assert_eq!(modes.eigenvector(i)[6], [0.0, 0.0, 0.0]);
        }
        let reference = rtb(&cluster6(), &[0, 0, 0, 1, 1, 1]).unwrap();
        for (a, b) in modes.eigenvalues().iter().zip(reference.eigenvalues()) {
            assert_relative_eq!(a, b, epsilon = 1e-9);
        }
    }

    /// If every atom is isolated (nothing within cutoff) there is no network.
    #[test]
    fn all_atoms_disconnected_is_too_few() {
        let atoms = [carbon(0.0, 0.0, 0.0), carbon(100.0, 0.0, 0.0)];
        assert!(matches!(
            NormalModes::builder(&atoms).cutoff(5.0).solve(),
            Err(Error::TooFewAtoms)
        ));
    }

    // --- Mode displacement (visualization) ---

    /// Displacing by `a` shifts every atom by `a·vᵢ`; amplitude 0 is the identity.
    #[test]
    fn displace_shifts_atoms_along_the_mode() {
        let atoms = cluster6();
        let positions: Vec<[f64; 3]> = atoms.iter().map(|a| a.position).collect();
        let modes = NormalModes::builder(&atoms).cutoff(5.0).solve().unwrap();

        assert_eq!(modes.displace(&positions, 6, 0.0), positions);

        let (i, a) = (6, 0.5); // mode 6 is the first non-zero (internal) mode
        let moved = modes.displace(&positions, i, a);
        let mode = modes.eigenvector(i);
        for (got, (orig, v)) in moved.iter().zip(positions.iter().zip(mode)) {
            for c in 0..3 {
                assert_relative_eq!(got[c], orig[c] + a * v[c], epsilon = 1e-12);
            }
        }
        assert!(moved.iter().zip(&positions).any(|(g, o)| g != o));
    }

    /// A dropped (disconnected) atom is zero in every mode, so it never moves.
    #[test]
    fn displace_leaves_disconnected_atom_fixed() {
        let mut atoms = cluster6();
        atoms.push(carbon(100.0, 100.0, 100.0));
        let positions: Vec<[f64; 3]> = atoms.iter().map(|a| a.position).collect();
        let modes = NormalModes::builder(&atoms).cutoff(5.0).solve().unwrap();

        let moved = modes.displace(&positions, 6, 5.0);
        assert_eq!(moved[6], positions[6]);
    }

    fn distance(p: &[[f64; 3]], a: usize, b: usize) -> f64 {
        ((p[a][0] - p[b][0]).powi(2) + (p[a][1] - p[b][1]).powi(2) + (p[a][2] - p[b][2]).powi(2))
            .sqrt()
    }

    /// The defining property: nonlinear extrapolation moves each block as a rigid
    /// body, so intra-block distances are preserved at large amplitude — and at
    /// least one mode genuinely rotates a block (so linear and nonlinear differ).
    #[test]
    fn nonlinear_preserves_intra_block_distances() {
        let atoms = cluster6();
        let positions: Vec<[f64; 3]> = atoms.iter().map(|a| a.position).collect();
        let modes = rtb(&atoms, &[0, 0, 0, 1, 1, 1]).unwrap();

        let mut saw_rotation = false;
        for i in 6..modes.len() {
            let moved = modes.displace_nonlinear(&positions, i, 3.0).unwrap();
            for group in [[0, 1, 2], [3, 4, 5]] {
                for &a in &group {
                    for &b in &group {
                        assert_relative_eq!(
                            distance(&moved, a, b),
                            distance(&positions, a, b),
                            epsilon = 1e-9
                        );
                    }
                }
            }
            let linear = modes.displace(&positions, i, 3.0);
            if moved
                .iter()
                .zip(&linear)
                .any(|(m, l)| (m[0] - l[0]).abs() > 1e-6)
            {
                saw_rotation = true;
            }
        }
        assert!(saw_rotation, "expected at least one mode to rotate a block");
    }

    /// Single-atom blocks have no rotation, so nonlinear reduces to the linear
    /// translation (on the unit-mass path the two coincide exactly).
    #[test]
    fn nonlinear_singleton_blocks_equal_linear() {
        let atoms = cluster6();
        let blocks: Vec<usize> = (0..atoms.len()).collect();
        let positions: Vec<[f64; 3]> = atoms.iter().map(|a| a.position).collect();
        let modes = rtb(&atoms, &blocks).unwrap();

        for i in 6..modes.len() {
            let nonlinear = modes.displace_nonlinear(&positions, i, 1.5).unwrap();
            let linear = modes.displace(&positions, i, 1.5);
            for (a, b) in nonlinear.iter().zip(&linear) {
                for c in 0..3 {
                    assert_relative_eq!(a[c], b[c], epsilon = 1e-9);
                }
            }
        }
    }

    /// Ties the nonlinear reconstruction to the ProDy/NOLB-validated modes: at
    /// small amplitude the rigid motion is the *physical* velocity field, which
    /// equals the mass-weighted lifted mode (what `displace` returns) divided by
    /// `√mass` per atom. Holding for every mode means `v = ṽ_w/√M_b`,
    /// `ω = I^{-1/2}·ω̃_w` reconstructs the modes exactly.
    #[test]
    fn nonlinear_small_amplitude_is_physical_mode() {
        let atoms = vec![
            Atom {
                position: [0.0, 0.0, 0.0],
                mass: 12.0,
            },
            Atom {
                position: [1.2, 0.0, 0.0],
                mass: 14.0,
            },
            Atom {
                position: [0.0, 1.2, 0.0],
                mass: 16.0,
            },
            Atom {
                position: [3.0, 0.0, 1.0],
                mass: 12.0,
            },
            Atom {
                position: [4.2, 0.2, 0.5],
                mass: 32.0,
            },
            Atom {
                position: [3.0, 1.2, 1.5],
                mass: 14.0,
            },
        ];
        let positions: Vec<[f64; 3]> = atoms.iter().map(|a| a.position).collect();
        let modes = NormalModes::builder(&atoms)
            .cutoff(5.0)
            .mass_weighted()
            .blocks(&[0, 0, 0, 1, 1, 1])
            .solve()
            .unwrap();

        let a = 1e-6;
        for i in 6..modes.len() {
            let nonlinear = modes.displace_nonlinear(&positions, i, a).unwrap();
            let lifted = modes.displace(&positions, i, a);
            for (k, atom) in atoms.iter().enumerate() {
                let sqrt_m = atom.mass.sqrt();
                for c in 0..3 {
                    let physical = (nonlinear[k][c] - positions[k][c]) * sqrt_m;
                    let lifted_c = lifted[k][c] - positions[k][c];
                    assert_relative_eq!(physical, lifted_c, epsilon = 1e-12, max_relative = 1e-5);
                }
            }
        }
    }

    /// Nonlinear modes need the rigid-block data, so the plain solver is rejected.
    #[test]
    fn nonlinear_requires_blocks() {
        let atoms = cluster6();
        let positions: Vec<[f64; 3]> = atoms.iter().map(|a| a.position).collect();
        let modes = NormalModes::builder(&atoms).cutoff(5.0).solve().unwrap();
        assert!(matches!(
            modes.displace_nonlinear(&positions, 6, 1.0),
            Err(Error::NotRigidBlocks)
        ));
    }

    /// Without the `sparse` feature, `k_modes` falls back to a dense solve and
    /// returns the same lowest `k` non-zero modes the full solve would — on both
    /// constructors. (With `sparse` the same is checked against the partial
    /// solver in `tests/sparse.rs`.)
    #[cfg(not(feature = "sparse"))]
    #[test]
    fn k_modes_falls_back_to_dense() {
        let atoms = cluster6();

        let lowest_two = |full: &NormalModes| -> Vec<f64> {
            full.eigenvalues()
                .iter()
                .filter(|&&v| v > ZERO_EIGENVALUE)
                .take(2)
                .copied()
                .collect()
        };

        let plain = NormalModes::builder(&atoms)
            .cutoff(5.0)
            .k_modes(2)
            .solve()
            .unwrap();
        assert_eq!(plain.len(), 2);
        let plain_full = NormalModes::builder(&atoms).cutoff(5.0).solve().unwrap();
        for (got, want) in plain.eigenvalues().iter().zip(lowest_two(&plain_full)) {
            assert_relative_eq!(got, &want, epsilon = 1e-9);
        }

        let blocks = [0, 0, 0, 1, 1, 1];
        let rtb_k = NormalModes::builder(&atoms)
            .cutoff(5.0)
            .k_modes(2)
            .blocks(&blocks)
            .solve()
            .unwrap();
        assert_eq!(rtb_k.len(), 2);
        let rtb_full = rtb(&atoms, &blocks).unwrap();
        for (got, want) in rtb_k.eigenvalues().iter().zip(lowest_two(&rtb_full)) {
            assert_relative_eq!(got, &want, epsilon = 1e-9);
        }
    }

    // --- explicit springs ---

    /// All pairs as unit-weight springs reproduce the cutoff network exactly (here
    /// the cluster is small enough that a 15 Å cutoff also connects every pair).
    #[test]
    fn unit_weight_springs_match_the_cutoff_network() {
        let atoms = cluster();
        let springs: Vec<Spring> = (0..atoms.len())
            .flat_map(|i| (i + 1..atoms.len()).map(move |j| Spring { i, j, weight: 1.0 }))
            .collect();
        let by_edges = NormalModes::builder(&atoms)
            .springs(&springs)
            .solve()
            .unwrap();
        let by_cutoff = NormalModes::builder(&atoms).cutoff(15.0).solve().unwrap();
        for (a, b) in by_edges.eigenvalues().iter().zip(by_cutoff.eigenvalues()) {
            assert_relative_eq!(a, b, epsilon = 1e-9);
        }
    }

    /// The weight scales an edge's stiffness, so a diatomic's stretch eigenvalue
    /// scales with it.
    #[test]
    fn doubling_a_spring_weight_doubles_its_eigenvalue() {
        let atoms = [carbon(0.0, 0.0, 0.0), carbon(3.8, 0.0, 0.0)];
        let stretch = |weight| {
            NormalModes::builder(&atoms)
                .springs(&[Spring { i: 0, j: 1, weight }])
                .solve()
                .unwrap()
                .eigenvalues()[5]
        };
        assert_relative_eq!(stretch(2.0), 2.0 * stretch(1.0), epsilon = 1e-9);
    }

    #[test]
    fn springs_reject_out_of_range_and_self_edges() {
        let atoms = [carbon(0.0, 0.0, 0.0), carbon(1.0, 0.0, 0.0)];
        let bad = |spring| {
            NormalModes::builder(&atoms)
                .springs(&[spring])
                .solve()
                .unwrap_err()
        };
        assert_eq!(
            bad(Spring {
                i: 0,
                j: 5,
                weight: 1.0
            }),
            Error::InvalidSpring
        );
        assert_eq!(
            bad(Spring {
                i: 1,
                j: 1,
                weight: 1.0
            }),
            Error::InvalidSpring
        );
    }

    #[test]
    fn springs_drop_a_degree_zero_atom() {
        let atoms = [
            carbon(0.0, 0.0, 0.0),
            carbon(1.5, 0.0, 0.0),
            carbon(5.0, 5.0, 5.0), // touched by no spring
        ];
        let modes = NormalModes::builder(&atoms)
            .springs(&[Spring {
                i: 0,
                j: 1,
                weight: 1.0,
            }])
            .solve()
            .unwrap();
        assert_eq!(modes.disconnected(), &[2]);
    }

    #[test]
    fn solve_without_connectivity_errors() {
        let atoms = cluster();
        assert_eq!(
            NormalModes::builder(&atoms).solve().unwrap_err(),
            Error::NoNetwork
        );
    }

    #[test]
    fn coincident_atoms_are_rejected() {
        // A duplicate atom at an existing position: the zero-length spring would
        // make the Hessian's 1/d² term blow up, so the solve must reject it.
        let atoms = [
            carbon(0.0, 0.0, 0.0),
            carbon(0.0, 0.0, 0.0),
            carbon(1.5, 0.0, 0.0),
        ];
        assert_eq!(
            NormalModes::builder(&atoms)
                .cutoff(5.0)
                .solve()
                .unwrap_err(),
            Error::CoincidentAtoms
        );
    }

    #[test]
    fn spring_count_reports_network_size() {
        let atoms = [
            carbon(0.0, 0.0, 0.0),
            carbon(1.0, 0.0, 0.0),
            carbon(0.0, 1.0, 0.0),
        ];
        // All three pairs are within the cutoff: the triangle has three springs.
        let by_cutoff = NormalModes::builder(&atoms).cutoff(15.0).solve().unwrap();
        assert_eq!(by_cutoff.spring_count(), 3);
        // An explicit two-edge path keeps exactly those two springs.
        let by_edges = NormalModes::builder(&atoms)
            .springs(&[
                Spring {
                    i: 0,
                    j: 1,
                    weight: 1.0,
                },
                Spring {
                    i: 1,
                    j: 2,
                    weight: 1.0,
                },
            ])
            .solve()
            .unwrap();
        assert_eq!(by_edges.spring_count(), 2);
    }
}
