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
//! ideal for small and medium systems. [`Params::k_modes`] returns only the
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
    /// Uniform spring constant. It scales every eigenvalue by the same factor, so
    /// its value only sets the overall scale; the mode shapes and the ratios
    /// between eigenvalues do not depend on it.
    pub gamma: f64,
    /// When true, diagonalize the mass-weighted Hessian `M^{-1/2} H M^{-1/2}`
    /// instead of `H`; eigenvalues are then squared frequencies `ω²`.
    pub mass_weighted: bool,
    /// Number of lowest *non-zero* modes to compute. `None` (the default) returns
    /// all modes, including the ~6 rigid-body ones. `Some(k)` returns exactly the
    /// `k` lowest non-zero modes (rigid-body modes excluded). The `sparse` feature
    /// computes them without ever forming the dense Hessian, which is what makes
    /// large systems feasible; without it the result is the same but comes from a
    /// full dense solve, so it is practical only up to medium systems.
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
    /// The sparse solver could not factor the Hessian or did not converge.
    SparseSolverFailed,
    /// [`NormalModes::displace_nonlinear`] was called on a result that has no
    /// rigid blocks — it needs the per-block velocities only
    /// [`NormalModes::with_blocks`] retains. Build the modes with `with_blocks`.
    NotRigidBlocks,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewAtoms => write!(f, "at least two atoms are required"),
            Self::NotFinite => write!(f, "non-finite coordinate or mass"),
            Self::BlockCountMismatch => write!(f, "blocks must have one entry per atom"),
            Self::DegenerateBlock => write!(f, "a multi-atom block is collinear or coincident"),
            Self::SparseSolverFailed => write!(f, "the sparse solver failed"),
            Self::NotRigidBlocks => write!(f, "nonlinear modes require with_blocks"),
        }
    }
}

impl std::error::Error for Error {}

/// One harmonic spring of the network: a pair of atoms (by *original* index) and
/// their equilibrium separation, the rest length a displacement is measured from.
#[derive(Debug)]
struct Spring {
    i: usize,
    j: usize,
    rest: f64,
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
    /// Rigid-block decomposition, kept only for [`with_blocks`](NormalModes::with_blocks)
    /// results; it carries the per-block velocities the nonlinear extrapolation needs.
    rtb: Option<Rtb>,
    /// Equilibrium springs and their uniform constant, kept so
    /// [`energy`](NormalModes::energy) can score any conformation without
    /// rebuilding the network.
    springs: Vec<Spring>,
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

/// Boltzmann constant in kcal·mol⁻¹·K⁻¹. The absolute amplitude scale is only
/// meaningful relative to `gamma` and your unit choices, so callers commonly
/// rescale the result regardless.
const BOLTZMANN_KCAL_PER_MOL_K: f64 = 1.987_204_259e-3;

/// Below this angular speed a block's nonlinear motion is treated as a pure
/// translation, which also guards the rotation-axis normalization against a
/// near-zero vector.
const ROTATION_EPS: f64 = 1e-12;

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
    /// symmetric eigendecomposition.
    pub fn new(atoms: &[Atom], params: &Params) -> Result<Self, Error> {
        let net = prepare(atoms, params)?;

        // The `sparse` feature computes only the lowest `k` modes directly; without
        // it `k_modes` still works, via a full dense solve truncated to those modes.
        #[cfg(feature = "sparse")]
        if let Some(k) = params.k_modes {
            return Self::solve_partial(&net, params, k, atoms.len());
        }

        let h = build_hessian(net.keep.len(), &net.weights, &net.contacts, params);
        let spectrum = eigen::solve(h);
        match params.k_modes {
            None => Ok(Self::from_modes(
                spectrum.eigenvalues,
                &spectrum.eigenvectors,
                &net,
                atoms.len(),
                params.gamma,
            )),
            Some(k) => {
                let columns = lowest_nonzero_columns(&spectrum.eigenvalues, k);
                let eigenvalues = columns.iter().map(|&c| spectrum.eigenvalues[c]).collect();
                let vectors = pick_columns(&spectrum.eigenvectors, &columns);
                Ok(Self::from_modes(
                    eigenvalues,
                    &vectors,
                    &net,
                    atoms.len(),
                    params.gamma,
                ))
            }
        }
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
        Ok(Self::from_modes(
            eigenvalues,
            &vectors,
            net,
            n_original,
            params.gamma,
        ))
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
    /// returned. The `sparse` feature computes them with a matrix-free partial
    /// solver that never forms the reduced matrix; without it they come from a
    /// full dense reduction.
    pub fn with_blocks(atoms: &[Atom], blocks: &[usize], params: &Params) -> Result<Self, Error> {
        if blocks.len() != atoms.len() {
            return Err(Error::BlockCountMismatch);
        }
        let net = prepare(atoms, params)?;
        // The drop carries the blocks along: a block keeps only its connected atoms.
        let blocks: Vec<usize> = net.keep.iter().map(|&old| blocks[old]).collect();

        #[cfg(feature = "sparse")]
        if let Some(k) = params.k_modes {
            return Self::solve_rtb_partial(&net, &blocks, params, k, atoms.len());
        }

        let h = build_hessian(net.keep.len(), &net.weights, &net.contacts, params);
        // Reduce to the block subspace and solve there; `tr_mul` forms Pᵀ·(H·P)
        // without materializing the transpose of P.
        let p = rtb::projection(&net.positions, &net.weights, &blocks)?;
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
        let rtb = Self::build_rtb(&net, &blocks, reduced)?;
        let mut modes = Self::from_modes(eigenvalues, &all_atom, &net, atoms.len(), params.gamma);
        modes.rtb = Some(rtb);
        Ok(modes)
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
        let (eigenvalues, vectors, reduced) = sparse::lowest_rtb_modes(
            &net.positions,
            &net.weights,
            blocks,
            params.gamma,
            &net.contacts,
            k,
        )?;
        let rtb = Self::build_rtb(net, blocks, reduced)?;
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
        // Record each spring in original atom indices with its rest length, so
        // `energy` can score later conformations directly.
        let springs = net
            .contacts
            .iter()
            .map(|c| Spring {
                i: net.keep[c.i],
                j: net.keep[c.j],
                rest: c.dist2.sqrt(),
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
    fn build_rtb(net: &Network, blocks: &[usize], reduced: DMatrix<f64>) -> Result<Rtb, Error> {
        let mut geometry = rtb::block_geometry(&net.positions, &net.weights, blocks)?;
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
    /// reduced rigid-block degree-of-freedom count for
    /// [`with_blocks`](Self::with_blocks).
    pub const fn len(&self) -> usize {
        self.eigenvalues.len()
    }

    /// Whether there are no modes — always false for a successful solve, provided
    /// as the conventional companion to [`len`](Self::len).
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
    /// [`Error::NotRigidBlocks`] if the modes came from [`new`](Self::new), which
    /// keeps no blocks. Build them with [`with_blocks`](Self::with_blocks).
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

    /// Elastic-network energy of a conformation: `½γ Σ (|r_ij| − r⁰_ij)²` over the
    /// springs, in the energy units of `gamma` (kJ/mol when γ is in kJ/mol/Å²).
    ///
    /// This is the potential whose Boltzmann factor `exp(−E / k_B T)` weights a
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
                (distance - s.rest).powi(2)
            })
            .sum();
        0.5 * self.gamma * sum
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
        let modes = NormalModes::new(&atoms, &Params::default()).unwrap();
        assert_relative_eq!(modes.energy(&positions(&atoms)), 0.0);
    }

    #[test]
    fn energy_is_zero_under_rigid_translation() {
        let atoms = cluster();
        let modes = NormalModes::new(&atoms, &Params::default()).unwrap();
        let shifted: Vec<_> = positions(&atoms)
            .iter()
            .map(|p| [p[0] + 5.0, p[1] - 3.0, p[2] + 1.0])
            .collect();
        assert_relative_eq!(modes.energy(&shifted), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn energy_is_zero_under_rigid_rotation() {
        let atoms = cluster();
        let modes = NormalModes::new(&atoms, &Params::default()).unwrap();
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
        let modes = NormalModes::new(&atoms, &Params::default()).unwrap();
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
            let base = NormalModes::new(&atoms, &Params::default()).unwrap();
            base.displace(&positions(&atoms), 6, 0.7)
        };
        let unit = NormalModes::new(&atoms, &Params::default()).unwrap();
        let stiff = NormalModes::new(
            &atoms,
            &Params {
                gamma: 3.0,
                ..Params::default()
            },
        )
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
        let modes = NormalModes::new(&atoms, &Params::default()).unwrap();
        assert_eq!(modes.disconnected(), &[3]);
        let mut moved = positions(&atoms);
        moved[3] = [200.0, -50.0, 7.0];
        assert_relative_eq!(modes.energy(&moved), 0.0);
    }

    #[test]
    fn a_displaced_mode_has_positive_energy() {
        let atoms = cluster();
        let modes = NormalModes::new(&atoms, &Params::default()).unwrap();
        let displaced = modes.displace(&positions(&atoms), 6, 0.5);
        assert!(modes.energy(&displaced) > ZERO_EIGENVALUE);
    }

    #[test]
    #[should_panic(expected = "one entry per atom")]
    fn energy_rejects_wrong_length_positions() {
        let atoms = cluster();
        let modes = NormalModes::new(&atoms, &Params::default()).unwrap();
        let _ = modes.energy(&[[0.0, 0.0, 0.0]]);
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

    // --- Mode displacement (visualization) ---

    /// Displacing by `a` shifts every atom by `a·vᵢ`; amplitude 0 is the identity.
    #[test]
    fn displace_shifts_atoms_along_the_mode() {
        let atoms = cluster6();
        let positions: Vec<[f64; 3]> = atoms.iter().map(|a| a.position).collect();
        let modes = NormalModes::new(&atoms, &rtb_params()).unwrap();

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
        let modes = NormalModes::new(&atoms, &rtb_params()).unwrap();

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
        let modes = NormalModes::with_blocks(&atoms, &[0, 0, 0, 1, 1, 1], &rtb_params()).unwrap();

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
        let modes = NormalModes::with_blocks(&atoms, &blocks, &rtb_params()).unwrap();

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
        let params = Params {
            cutoff: 5.0,
            mass_weighted: true,
            ..Params::default()
        };
        let modes = NormalModes::with_blocks(&atoms, &[0, 0, 0, 1, 1, 1], &params).unwrap();

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
        let modes = NormalModes::new(&atoms, &rtb_params()).unwrap();
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
        let mut params = rtb_params();
        params.k_modes = Some(2);

        let lowest_two = |full: &NormalModes| -> Vec<f64> {
            full.eigenvalues()
                .iter()
                .filter(|&&v| v > ZERO_EIGENVALUE)
                .take(2)
                .copied()
                .collect()
        };

        let plain = NormalModes::new(&atoms, &params).unwrap();
        assert_eq!(plain.len(), 2);
        let plain_full = NormalModes::new(&atoms, &rtb_params()).unwrap();
        for (got, want) in plain.eigenvalues().iter().zip(lowest_two(&plain_full)) {
            assert_relative_eq!(got, &want, epsilon = 1e-9);
        }

        let blocks = [0, 0, 0, 1, 1, 1];
        let rtb = NormalModes::with_blocks(&atoms, &blocks, &params).unwrap();
        assert_eq!(rtb.len(), 2);
        let rtb_full = NormalModes::with_blocks(&atoms, &blocks, &rtb_params()).unwrap();
        for (got, want) in rtb.eigenvalues().iter().zip(lowest_two(&rtb_full)) {
            assert_relative_eq!(got, &want, epsilon = 1e-9);
        }
    }
}
