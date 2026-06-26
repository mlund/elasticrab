//! Golden test for the RTB (Rotation-Translation Blocks) path against ProDy's
//! reference reduced Hessian for the truncated 2GB1 Cα model.
//!
//! ProDy stores the block-reduced Hessian `Pᵀ H P` (`rtb2gb1_hessian.coo`). A
//! block's rotational basis is only defined up to orientation, so the reduced
//! matrices need not match element-wise — but their **eigenvalue spectra** are
//! basis-invariant and must agree exactly. We therefore diagonalize ProDy's
//! reference matrix and compare its spectrum to ours. See `tests/data/ATTRIBUTION.md`.

use nalgebra::{DMatrix, SymmetricEigen};

use elasticrab::{Atom, NormalModes, Params};

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

/// Read Cα coordinates (columns 31–54) and the per-atom block id from the beta
/// column (61–66) of a PDB file.
fn read_blocked_pdb(path: &str) -> (Vec<Atom>, Vec<usize>) {
    let mut atoms = Vec::new();
    let mut blocks = Vec::new();
    for line in std::fs::read_to_string(path).unwrap().lines() {
        if !line.starts_with("ATOM") {
            continue;
        }
        let f = |a: usize, b: usize| line[a..b].trim().parse::<f64>().unwrap();
        atoms.push(Atom {
            position: [f(30, 38), f(38, 46), f(46, 54)],
            mass: 1.0,
        });
        blocks.push(f(60, 66).round() as usize);
    }
    (atoms, blocks)
}

/// Ascending eigenvalues of a symmetric matrix stored as 1-indexed COO triples.
fn coo_spectrum(path: &str, dof: usize) -> Vec<f64> {
    let mut m = DMatrix::zeros(dof, dof);
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let mut it = line.split_whitespace();
        let i: usize = it.next().unwrap().parse().unwrap();
        let j: usize = it.next().unwrap().parse().unwrap();
        let v: f64 = it.next().unwrap().parse().unwrap();
        m[(i - 1, j - 1)] = v;
        m[(j - 1, i - 1)] = v;
    }
    let mut vals: Vec<f64> = SymmetricEigen::new(m).eigenvalues.iter().copied().collect();
    vals.sort_by(f64::total_cmp);
    vals
}

#[test]
fn rtb_spectrum_matches_prody() {
    let (atoms, blocks) = read_blocked_pdb(&format!("{DATA}/2gb1_truncated.pdb"));
    let modes = NormalModes::with_blocks(&atoms, &blocks, &Params::default()).unwrap();

    // 28 atoms in 5 blocks: one singleton (3 DOF) + four multi-atom blocks (6 each).
    assert_eq!(modes.len(), 27);

    let reference = coo_spectrum(&format!("{DATA}/rtb2gb1_hessian.coo"), 27);
    for (k, (&got, &expected)) in modes.eigenvalues().iter().zip(&reference).enumerate() {
        assert!(
            (got - expected).abs() < 1e-5,
            "RTB eigenvalue {k}: got {got:e}, expected {expected:e}"
        );
    }

    // A connected structure keeps exactly six rigid-body (~zero) modes.
    let zeros = modes
        .eigenvalues()
        .iter()
        .filter(|&&v| v.abs() < 1e-6)
        .count();
    assert_eq!(zeros, 6);
}
