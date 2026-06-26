//! Mass-weighted RTB golden test against NOLB, the engine Pepsi-SAXS wraps.
//!
//! NOLB (Grudinin's NOn-Linear rigid Block NMA, the engine Pepsi-SAXS wraps) is
//! the reference for the mass-weighted RTB path. Its frequency is
//! `sqrt(eigenvalue of the mass-weighted, RTB-reduced Hessian)` up to a fixed
//! global unit constant, so we check that elasticrab's spectrum is *proportional*
//! to NOLB's. The reference frequencies are vendored in `tests/data/` (generated
//! once with the NOLB binary); this test never invokes it. See the fixture
//! header and `docs/PEPSI_COMPARISON.md`.

use elasticrab::{Atom, NormalModes, Params};

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

/// Standard atomic weights for the elements present in crambin.
fn element_mass(symbol: &str) -> f64 {
    match symbol {
        "C" => 12.011,
        "N" => 14.007,
        "O" => 15.999,
        "S" => 32.06,
        other => panic!("unexpected element {other}"),
    }
}

/// Parse an all-atom PDB into atoms (coordinates + element mass) and per-atom
/// rigid blocks taken from the residue sequence number (NOLB's default: one
/// rigid block per residue).
fn read_atoms_and_residue_blocks(path: &str) -> (Vec<Atom>, Vec<usize>) {
    let text = std::fs::read_to_string(path).unwrap();
    let mut atoms = Vec::new();
    let mut blocks = Vec::new();
    for line in text.lines().filter(|l| l.starts_with("ATOM")) {
        let f = |a: usize, b: usize| line[a..b].trim().parse::<f64>().unwrap();
        atoms.push(Atom {
            position: [f(30, 38), f(38, 46), f(46, 54)],
            mass: element_mass(line[76..78].trim()),
        });
        blocks.push(line[22..26].trim().parse().unwrap());
    }
    (atoms, blocks)
}

fn read_reference_freqs(path: &str) -> Vec<f64> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| l.trim().parse().unwrap())
        .collect()
}

#[test]
fn mass_weighted_rtb_matches_nolb() {
    let (atoms, blocks) = read_atoms_and_residue_blocks(&format!("{DATA}/crambin_heavy.pdb"));
    assert_eq!(atoms.len(), 327);

    let mut params = Params::default();
    params.cutoff = 5.0;
    params.mass_weighted = true;
    let modes = NormalModes::with_blocks(&atoms, &blocks, &params).unwrap();

    assert_eq!(modes.len(), 276); // 46 residues × 6 rigid DOF

    // Exactly six rigid-body (~zero) modes; the rest are genuine internal motions.
    let eigenvalues = modes.eigenvalues();
    let zeros = eigenvalues.iter().filter(|&&v| v.abs() < 1e-9).count();
    assert_eq!(zeros, 6);

    let ours: Vec<f64> = eigenvalues
        .iter()
        .filter(|&&v| v > 1e-9)
        .take(10)
        .map(|v| v.sqrt())
        .collect();
    let nolb = read_reference_freqs(&format!("{DATA}/nolb_crambin_freqs.txt"));

    // The two spectra are equal up to NOLB's global unit constant, so every
    // per-mode ratio must equal the first one.
    let scale = ours[0] / nolb[0];
    for (k, (o, n)) in ours.iter().zip(&nolb).enumerate() {
        let ratio = o / n;
        assert!(
            (ratio / scale - 1.0).abs() < 1e-3,
            "mode {k}: ratio {ratio} deviates from global scale {scale}"
        );
    }
}

/// Disconnected-atom parity with NOLB. `crambin_heavy_isolated.pdb` is crambin
/// plus one isolated carbon (its own residue, far from the protein). Run on it,
/// NOLB reports "Number of disconnected atoms : 1" and returns the *same* ten
/// frequencies as crambin alone (verified: bit-identical to
/// `nolb_crambin_freqs.txt`). elasticrab must likewise drop that atom and
/// recover the crambin spectrum — so this reuses the crambin fixture.
#[test]
fn disconnected_atom_dropped_matches_nolb() {
    let (atoms, blocks) =
        read_atoms_and_residue_blocks(&format!("{DATA}/crambin_heavy_isolated.pdb"));
    assert_eq!(atoms.len(), 328); // 327 crambin + 1 isolated carbon

    let mut params = Params::default();
    params.cutoff = 5.0;
    params.mass_weighted = true;
    let modes = NormalModes::with_blocks(&atoms, &blocks, &params).unwrap();

    // The isolated atom (the last one) is dropped, exactly as NOLB drops it.
    assert_eq!(modes.disconnected(), &[327]);
    assert_eq!(modes.len(), 276); // 46 residue blocks × 6; the dummy block is gone

    let eigenvalues = modes.eigenvalues();
    let zeros = eigenvalues.iter().filter(|&&v| v.abs() < 1e-9).count();
    assert_eq!(zeros, 6); // not 9 — the spurious modes of the isolated atom are gone

    let ours: Vec<f64> = eigenvalues
        .iter()
        .filter(|&&v| v > 1e-9)
        .take(10)
        .map(|v| v.sqrt())
        .collect();
    let nolb = read_reference_freqs(&format!("{DATA}/nolb_crambin_freqs.txt"));
    let scale = ours[0] / nolb[0];
    for (k, (o, n)) in ours.iter().zip(&nolb).enumerate() {
        let ratio = o / n;
        assert!(
            (ratio / scale - 1.0).abs() < 1e-3,
            "mode {k}: ratio {ratio} deviates from global scale {scale}"
        );
    }
}
