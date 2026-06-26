//! Wall-clock comparison of the four solver paths, to quantify the speedup of
//! the partial (sparse / matrix-free) solvers over the dense ones.
//!
//! Run with `cargo bench --features sparse`. Structures are vendored Cα-only
//! PDBs (glycogen phosphorylase 1A8I, ~800 residues; GroEL–GroES 1AON, ~8000
//! residues). The dense solvers run only on the medium structure — dense
//! all-atom on the large one would need a ~24k×24k diagonalization.

use divan::Bencher;
use elasticrab::{Atom, NormalModes, Params};

fn main() {
    divan::main();
}

const MEDIUM: &str = "bench_medium.pdb";
const LARGE: &str = "bench_large.pdb";
const K: usize = 10; // lowest non-zero modes for the partial solvers
const BLOCK: usize = 4; // Cα atoms per rigid block

/// Load a vendored Cα PDB into atoms plus fixed-size rigid blocks.
fn load(name: &str) -> (Vec<Atom>, Vec<usize>) {
    let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
    let atoms: Vec<Atom> = std::fs::read_to_string(path)
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
        .collect();
    let blocks = (0..atoms.len()).map(|i| i / BLOCK).collect();
    (atoms, blocks)
}

fn partial() -> Params {
    let mut p = Params::default();
    p.k_modes = Some(K);
    p
}

/// (1) Dense all-atom — medium only; the large one would not fit in memory.
#[divan::bench]
fn dense_all_atom(bencher: Bencher) {
    bencher
        .with_inputs(|| load(MEDIUM))
        .bench_values(|(atoms, _)| NormalModes::new(&atoms, &Params::default()).unwrap());
}

/// (2) Dense RTB — medium only.
#[divan::bench]
fn dense_rtb(bencher: Bencher) {
    bencher
        .with_inputs(|| load(MEDIUM))
        .bench_values(|(atoms, blocks)| {
            NormalModes::with_blocks(&atoms, &blocks, &Params::default()).unwrap()
        });
}

/// (3) Sparse partial all-atom — both sizes.
#[divan::bench(args = [MEDIUM, LARGE])]
fn sparse_partial(bencher: Bencher, file: &str) {
    bencher
        .with_inputs(|| load(file))
        .bench_values(|(atoms, _)| NormalModes::new(&atoms, &partial()).unwrap());
}

/// (4) Matrix-free RTB partial — both sizes.
#[divan::bench(args = [MEDIUM, LARGE])]
fn matrixfree_rtb(bencher: Bencher, file: &str) {
    bencher
        .with_inputs(|| load(file))
        .bench_values(|(atoms, blocks)| {
            NormalModes::with_blocks(&atoms, &blocks, &partial()).unwrap()
        });
}
