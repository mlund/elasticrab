//! Golden test against ProDy's reference ANM for ubiquitin (PDB 1UBI).
//!
//! ProDy is an independent, widely used implementation. Reproducing its Hessian
//! and eigenvalues to tight tolerance is the crate's primary correctness check
//! for the conventional (unit-mass) ANM. See `tests/data/ATTRIBUTION.md`.
//!
//! Parameters are ProDy's defaults — and `Params::default()` — namely a 15 Å
//! cutoff and unit spring constant; tolerances match ProDy's own test suite
//! (`atol = 1e-5` for the Hessian, `1e-4` for the eigenvalues).

// The Hessian reconstruction reads more clearly as a plain sum than as
// `mul_add`; this is test code, not a hot path.
#![allow(clippy::suboptimal_flops)]

use elasticrab::{Atom, NormalModes, Params};

mod common;
use common::{read_ca_pdb, read_eigenvalues, DATA};

/// Reference Hessian as a dense `dof × dof` row-major matrix, expanded from the
/// symmetric COO triangle (1-indexed `i j value`).
fn read_coo(path: &str, dof: usize) -> Vec<f64> {
    let mut m = vec![0.0; dof * dof];
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let mut it = line.split_whitespace();
        let i: usize = it.next().unwrap().parse().unwrap();
        let j: usize = it.next().unwrap().parse().unwrap();
        let v: f64 = it.next().unwrap().parse().unwrap();
        m[(i - 1) * dof + (j - 1)] = v;
        m[(j - 1) * dof + (i - 1)] = v;
    }
    m
}

fn compute() -> (Vec<Atom>, NormalModes) {
    let atoms = read_ca_pdb("1ubi_ca.pdb");
    let modes = NormalModes::new(&atoms, &Params::default()).unwrap();
    (atoms, modes)
}

#[test]
fn hessian_matches_prody() {
    let (atoms, modes) = compute();
    let dof = 3 * atoms.len();
    let reference = read_coo(&format!("{DATA}/anm1ubi_hessian.coo"), dof);

    // The library intentionally does not expose the Hessian, so validate it
    // indirectly: the full spectrum must reconstruct it as H = V Λ Vᵀ. This
    // checks the eigenvalues and eigenvectors together against ProDy's matrix.
    let mut reconstructed = vec![0.0; dof * dof];
    for k in 0..modes.len() {
        let lambda = modes.eigenvalues()[k];
        let v = modes.eigenvector(k);
        for a in 0..atoms.len() {
            for b in 0..atoms.len() {
                for da in 0..3 {
                    for db in 0..3 {
                        reconstructed[(3 * a + da) * dof + (3 * b + db)] +=
                            lambda * v[a][da] * v[b][db];
                    }
                }
            }
        }
    }

    let max_diff = reference
        .iter()
        .zip(&reconstructed)
        .map(|(r, c)| (r - c).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-5,
        "max Hessian difference {max_diff:e} exceeds 1e-5"
    );
}

#[test]
fn eigenvalues_match_prody() {
    let (_, modes) = compute();
    let reference = read_eigenvalues("anm1ubi_evalues.dat");

    assert_eq!(modes.len(), 228); // 76 Cα × 3

    for (k, &expected) in reference.iter().enumerate() {
        let got = modes.eigenvalues()[k];
        assert!(
            (got - expected).abs() < 1e-4,
            "eigenvalue {k}: got {got:e}, expected {expected:e}"
        );
    }
}
