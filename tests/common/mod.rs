//! Shared helpers for the integration tests (reference-file parsers).
//!
//! Lives in a `mod.rs` subdirectory so Cargo does not treat it as its own test
//! binary; each test file pulls it in with `mod common;`.

#![allow(dead_code)] // not every test binary uses every helper

use elasticrab::Atom;

/// Directory holding the vendored reference fixtures.
pub const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

/// Read Cα coordinates from a fixture PDB (fixed columns 31–54), unit mass.
/// Parsing belongs to the tests, not the library.
pub fn read_ca_pdb(name: &str) -> Vec<Atom> {
    std::fs::read_to_string(format!("{DATA}/{name}"))
        .unwrap()
        .lines()
        .filter(|l| l.starts_with("ATOM"))
        .map(|l| {
            let f = |a: usize, b: usize| l[a..b].trim().parse::<f64>().unwrap();
            Atom {
                position: [f(30, 38), f(38, 46), f(46, 54)],
                mass: 1.0,
            }
        })
        .collect()
}

/// Read reference eigenvalues from an `index value` table fixture.
pub fn read_eigenvalues(name: &str) -> Vec<f64> {
    std::fs::read_to_string(format!("{DATA}/{name}"))
        .unwrap()
        .lines()
        .map(|l| l.split_whitespace().nth(1).unwrap().parse().unwrap())
        .collect()
}
