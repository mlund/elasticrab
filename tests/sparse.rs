//! Tests for the sparse partial eigensolver (`--features sparse`).
//!
//! The partial solver must return the same lowest non-zero modes as the dense
//! full solve, and match ProDy's reference 1UBI eigenvalues.

#![cfg(feature = "sparse")]

use elasticrab::{Atom, NormalModes, Params};

mod common;
use common::{read_ca_pdb, read_eigenvalues};

fn ubiquitin() -> Vec<Atom> {
    read_ca_pdb("1ubi_ca.pdb")
}

fn partial(atoms: &[Atom], k: usize, mass_weighted: bool) -> NormalModes {
    let mut params = Params::default();
    params.k_modes = Some(k);
    params.mass_weighted = mass_weighted;
    NormalModes::new(atoms, &params).unwrap()
}

/// Group consecutive residues into rigid blocks of `size` Cα atoms each.
fn blocks_of(n_atoms: usize, size: usize) -> Vec<usize> {
    (0..n_atoms).map(|i| i / size).collect()
}

fn lowest_nonzero(modes: &NormalModes, k: usize) -> Vec<f64> {
    modes
        .eigenvalues()
        .iter()
        .filter(|&&v| v.abs() > 1e-6)
        .take(k)
        .copied()
        .collect()
}

/// Matrix-free RTB (`with_blocks` + `k_modes`) must match the dense RTB solve's
/// lowest non-zero modes.
#[test]
fn matrixfree_rtb_matches_dense_rtb() {
    let atoms = ubiquitin();
    let blocks = blocks_of(atoms.len(), 4); // 19 rigid blocks of 4 Cα
    let k = 10;

    let dense = NormalModes::with_blocks(&atoms, &blocks, &Params::default()).unwrap();
    let dense_nonzero = lowest_nonzero(&dense, k);

    let mut params = Params::default();
    params.k_modes = Some(k);
    let mf = NormalModes::with_blocks(&atoms, &blocks, &params).unwrap();

    assert_eq!(mf.len(), k);
    for (got, want) in mf.eigenvalues().iter().zip(&dense_nonzero) {
        assert!(
            (got - want).abs() < 1e-5 * want.max(1.0),
            "matrix-free {got:e} vs dense RTB {want:e}"
        );
    }
    assert!(mf.eigenvalues().iter().all(|&v| v > 1e-6));
}

/// With every atom its own block, `P` is the identity, so matrix-free RTB and
/// the all-atom sparse solver compute the same lowest non-zero modes (via
/// regular-mode and shift-invert Lanczos respectively).
#[test]
fn matrixfree_all_singleton_matches_all_atom_partial() {
    let atoms = ubiquitin();
    let singletons: Vec<usize> = (0..atoms.len()).collect();
    let k = 10;

    let mut params = Params::default();
    params.k_modes = Some(k);
    let rtb = NormalModes::with_blocks(&atoms, &singletons, &params).unwrap();
    let all_atom = NormalModes::new(&atoms, &params).unwrap();

    for (a, b) in rtb.eigenvalues().iter().zip(all_atom.eigenvalues()) {
        assert!((a - b).abs() < 1e-5 * b.max(1.0), "{a:e} vs {b:e}");
    }
}

/// Mass-weighted matrix-free RTB matches the dense mass-weighted RTB solve.
#[test]
fn matrixfree_rtb_mass_weighted() {
    let mut atoms = ubiquitin();
    for (i, a) in atoms.iter_mut().enumerate() {
        a.mass = 12.0 + (i % 3) as f64;
    }
    let blocks = blocks_of(atoms.len(), 5);
    let k = 8;

    let mut dense_params = Params::default();
    dense_params.mass_weighted = true;
    let dense = NormalModes::with_blocks(&atoms, &blocks, &dense_params).unwrap();
    let dense_nonzero = lowest_nonzero(&dense, k);

    let mut params = dense_params;
    params.k_modes = Some(k);
    let mf = NormalModes::with_blocks(&atoms, &blocks, &params).unwrap();

    for (got, want) in mf.eigenvalues().iter().zip(&dense_nonzero) {
        assert!(
            (got - want).abs() < 1e-5 * want.max(1.0),
            "mass-weighted matrix-free {got:e} vs dense {want:e}"
        );
    }
}

/// The k lowest non-zero modes from the sparse solver must equal the dense
/// solve's lowest k non-zero eigenvalues.
#[test]
fn sparse_matches_dense() {
    let atoms = ubiquitin();
    let k = 12;

    let dense = NormalModes::new(&atoms, &Params::default()).unwrap();
    let dense_nonzero: Vec<f64> = dense
        .eigenvalues()
        .iter()
        .filter(|&&v| v.abs() > 1e-6)
        .take(k)
        .copied()
        .collect();

    let sparse = partial(&atoms, k, false);
    assert_eq!(sparse.len(), k);
    for (got, want) in sparse.eigenvalues().iter().zip(&dense_nonzero) {
        assert!(
            (got - want).abs() < 1e-6,
            "sparse {got:e} vs dense {want:e}"
        );
    }
    // No rigid-body mode leaked into the result.
    assert!(sparse.eigenvalues().iter().all(|&v| v > 1e-6));
}

/// The sparse path matches ProDy's reference 1UBI eigenvalues. ProDy stores the
/// lowest 36; the first six are the rigid-body modes, so the lowest non-zero
/// modes are entries 6.. .
#[test]
fn sparse_matches_prody() {
    let atoms = ubiquitin();
    let k = 12;
    let reference = read_eigenvalues("anm1ubi_evalues.dat");

    let modes = partial(&atoms, k, false);
    for (i, &got) in modes.eigenvalues().iter().enumerate() {
        let want = reference[6 + i];
        assert!((got - want).abs() < 1e-4, "mode {i}: {got:e} vs {want:e}");
    }
}

/// Mass-weighting on the sparse path agrees with the dense mass-weighted solve.
#[test]
fn sparse_mass_weighted_matches_dense() {
    let mut atoms = ubiquitin();
    for (i, a) in atoms.iter_mut().enumerate() {
        a.mass = 12.0 + (i % 3) as f64; // distinct masses
    }
    let k = 8;

    let mut dense_params = Params::default();
    dense_params.mass_weighted = true;
    let dense = NormalModes::new(&atoms, &dense_params).unwrap();
    let dense_nonzero: Vec<f64> = dense
        .eigenvalues()
        .iter()
        .filter(|&&v| v.abs() > 1e-8)
        .take(k)
        .copied()
        .collect();

    let sparse = partial(&atoms, k, true);
    for (got, want) in sparse.eigenvalues().iter().zip(&dense_nonzero) {
        assert!(
            (got - want).abs() < 1e-7,
            "sparse {got:e} vs dense {want:e}"
        );
    }
}

/// Requesting more modes than exist clamps to all non-zero modes.
#[test]
fn k_larger_than_spectrum_clamps() {
    let atoms = ubiquitin();
    let dof = 3 * atoms.len();
    let modes = partial(&atoms, dof + 100, false);
    assert!(modes.len() <= dof - 6);
    assert!(!modes.is_empty());
}

/// The partial solver drops an isolated atom before solving, recovering the
/// connected spectrum exactly like the dense path.
#[test]
fn sparse_drops_disconnected_atom() {
    let mut atoms = ubiquitin();
    let isolated = atoms.len();
    atoms.push(Atom {
        position: [9999.0, 9999.0, 9999.0],
        mass: 1.0,
    });

    let modes = partial(&atoms, 10, false);
    assert_eq!(modes.disconnected(), &[isolated]);
    assert_eq!(modes.len(), 10);

    let reference = partial(&ubiquitin(), 10, false);
    for (got, want) in modes.eigenvalues().iter().zip(reference.eigenvalues()) {
        assert!(
            (got - want).abs() < 1e-5 * want.max(1.0),
            "with isolated {got:e} vs connected {want:e}"
        );
    }
}
