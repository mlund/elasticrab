//! Iterative nonlinear transition (NOLB `--nlin`/`--nIter`) tests: analytic invariants
//! plus a regime cross-check against a vendored NOLB golden. The binary is never
//! invoked — `nolb_nlin_hinge.txt` was generated once, offline. Cases that do not need
//! a real protein use a tiny cluster, since each re-solve diagonalizes in debug mode.

// Readable scale expressions over fused multiply-add (clippy::nursery).
#![allow(clippy::suboptimal_flops)]

use elasticrab::{transition_iterative, Atom, Error, NormalModes};

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

fn element_mass(symbol: &str) -> f64 {
    match symbol {
        "C" => 12.011,
        "N" => 14.007,
        "O" => 15.999,
        "S" => 32.06,
        other => panic!("unexpected element {other}"),
    }
}

/// crambin heavy atoms with one rigid block per residue (NOLB's default).
fn crambin() -> (Vec<Atom>, Vec<usize>) {
    let text = std::fs::read_to_string(format!("{DATA}/crambin_heavy.pdb")).unwrap();
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

/// A tiny two-block cluster — instant to solve, for the cases that only need *some*
/// rigid-block network rather than a real protein.
fn tiny() -> (Vec<Atom>, Vec<usize>) {
    let positions = [
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [0.0, 1.5, 0.0],
        [1.5, 1.5, 1.2],
        [3.0, 0.2, 0.0],
        [3.0, 1.5, 0.3],
    ];
    let atoms = positions
        .iter()
        .map(|&position| Atom {
            position,
            mass: 12.0,
        })
        .collect();
    (atoms, vec![0, 0, 0, 1, 1, 1])
}

fn read_positions(name: &str) -> Vec<[f64; 3]> {
    std::fs::read_to_string(format!("{DATA}/{name}"))
        .unwrap()
        .lines()
        .filter(|l| l.starts_with("ATOM"))
        .map(|line| {
            let f = |a: usize, b: usize| line[a..b].trim().parse::<f64>().unwrap();
            [f(30, 38), f(38, 46), f(46, 54)]
        })
        .collect()
}

fn golden_value(name: &str, key: &str) -> f64 {
    let text = std::fs::read_to_string(format!("{DATA}/{name}")).unwrap();
    for line in text.lines().filter(|l| !l.trim_start().starts_with('#')) {
        let mut it = line.split_whitespace();
        if it.next() == Some(key) {
            return it.next().unwrap().parse().unwrap();
        }
    }
    panic!("key {key} not found in {name}");
}

fn positions(atoms: &[Atom]) -> Vec<[f64; 3]> {
    atoms.iter().map(|a| a.position).collect()
}

/// A re-solve closure: a 5 Å mass-weighted rigid-block network of the lowest `k` modes,
/// rebuilt from the per-iteration positions.
fn rigid_block_solver<'a>(
    masses: &'a [f64],
    blocks: &'a [usize],
    k: usize,
) -> impl FnMut(&[[f64; 3]]) -> Result<NormalModes, Error> + 'a {
    move |coords: &[[f64; 3]]| {
        let atoms: Vec<Atom> = coords
            .iter()
            .zip(masses)
            .map(|(&position, &mass)| Atom { position, mass })
            .collect();
        NormalModes::builder(&atoms)
            .cutoff(5.0)
            .blocks(blocks)
            .mass_weighted()
            .k_modes(k)
            .solve()
    }
}

fn masses_of(atoms: &[Atom]) -> Vec<f64> {
    atoms.iter().map(|a| a.mass).collect()
}

#[test]
fn redo_zero_reproduces_the_single_morph_endpoint() {
    let (atoms, blocks) = crambin();
    let native = positions(&atoms);
    let target = read_positions("crambin_hinge_large.pdb");
    let masses = masses_of(&atoms);

    // Use *all* modes (no k_modes): the combined velocity is then the full non-rigid
    // projection, basis-invariant, so the two independent solves can't disagree at a
    // truncation boundary where a near-degenerate pair could split between them.
    let solve = |coords: &[[f64; 3]]| {
        let atoms: Vec<Atom> = coords
            .iter()
            .zip(&masses)
            .map(|(&position, &mass)| Atom { position, mass })
            .collect();
        NormalModes::builder(&atoms)
            .cutoff(5.0)
            .blocks(&blocks)
            .mass_weighted()
            .solve()
    };

    // One iteration, one frame = the full nonlinear extrapolation from native.
    let iter = transition_iterative(&native, &target, 0, 1, solve).unwrap();

    // The same thing through the single-morph path.
    let modes = NormalModes::builder(&atoms)
        .cutoff(5.0)
        .blocks(&blocks)
        .mass_weighted()
        .solve()
        .unwrap();
    let morph = modes
        .transition(&native, &target)
        .unwrap()
        .morph(2, true)
        .unwrap();

    for (a, b) in iter
        .frames()
        .last()
        .unwrap()
        .iter()
        .zip(morph.last().unwrap())
    {
        for c in 0..3 {
            assert!(
                (a[c] - b[c]).abs() < 1e-6,
                "endpoint differs from single morph"
            );
        }
    }
}

#[test]
fn identity_target_has_no_motion() {
    let (atoms, blocks) = tiny();
    let native = positions(&atoms);
    let masses = masses_of(&atoms);
    let result = transition_iterative(
        &native,
        &native,
        3,
        2,
        rigid_block_solver(&masses, &blocks, 3),
    )
    .unwrap();
    assert!(result.final_rmsd() < 1e-6);
    for &r in result.step_rmsds() {
        assert!(r < 1e-6);
    }
}

#[test]
fn re_diagonalization_converges_and_matches_nolb() {
    // Regime cross-check, not bit-exact — see the header of tests/data/nolb_nlin_hinge.txt
    // for the command, NOLB version, and why elasticrab closes a little less of the gap
    // (NOLB keeps near-rigid modes and uses a step-size line search). This one test
    // covers monotone convergence, the "iterating helps" margin, and the golden.
    let (atoms, blocks) = crambin();
    let native = positions(&atoms);
    let target = read_positions("crambin_hinge_large.pdb");
    let masses = masses_of(&atoms);

    let result = transition_iterative(
        &native,
        &target,
        5,
        2,
        rigid_block_solver(&masses, &blocks, 10),
    )
    .unwrap();

    // Each re-diagonalization closes the gap or holds — never worse.
    for pair in result.step_rmsds().windows(2) {
        assert!(
            pair[1] <= pair[0] + 1e-9,
            "RMSD went up: {} -> {}",
            pair[0],
            pair[1]
        );
    }

    // Re-diagonalizing beats a single morph by a clear margin on this large hinge.
    let single = transition_iterative(
        &native,
        &target,
        0,
        2,
        rigid_block_solver(&masses, &blocks, 10),
    )
    .unwrap();
    assert!(
        result.final_rmsd() < single.final_rmsd() - 0.02,
        "iterating did not help: {} vs {}",
        result.final_rmsd(),
        single.final_rmsd()
    );

    // Same regime as NOLB: the aligned initial gap matches closely; the converged RMSD
    // is a real reduction and within 20% of NOLB's.
    let nolb_initial = golden_value("nolb_nlin_hinge.txt", "initial_rmsd");
    let nolb_final = golden_value("nolb_nlin_hinge.txt", "final_rmsd");
    assert!(
        (result.initial_rmsd() - nolb_initial).abs() < 0.03 * nolb_initial,
        "initial RMSD {} vs NOLB {nolb_initial}",
        result.initial_rmsd()
    );
    assert!(
        result.final_rmsd() < 0.85 * result.initial_rmsd(),
        "no real reduction"
    );
    assert!(
        (result.final_rmsd() - nolb_final).abs() < 0.20 * nolb_final,
        "final RMSD {} vs NOLB {nolb_final}",
        result.final_rmsd()
    );
}

#[test]
fn mismatched_atom_count_is_rejected() {
    let (atoms, blocks) = tiny();
    let native = positions(&atoms);
    let masses = masses_of(&atoms);
    let short = &native[..native.len() - 1];
    // The length check fires before any solve.
    assert!(matches!(
        transition_iterative(
            &native,
            short,
            1,
            2,
            rigid_block_solver(&masses, &blocks, 3)
        ),
        Err(Error::AtomCountMismatch)
    ));
}

#[test]
fn block_less_modes_are_rejected() {
    let (atoms, _blocks) = tiny();
    let native = positions(&atoms);
    let masses = masses_of(&atoms);
    // Same atoms slightly displaced, so there is a real (but unprojectable) target.
    let target: Vec<[f64; 3]> = native.iter().map(|p| [p[0] + 0.3, p[1], p[2]]).collect();
    // A solver that omits rigid blocks: the nonlinear extrapolation has nothing to drive.
    let solve = |coords: &[[f64; 3]]| {
        let atoms: Vec<Atom> = coords
            .iter()
            .zip(&masses)
            .map(|(&position, &mass)| Atom { position, mass })
            .collect();
        NormalModes::builder(&atoms)
            .cutoff(5.0)
            .mass_weighted()
            .k_modes(3)
            .solve()
    };
    assert!(matches!(
        transition_iterative(&native, &target, 1, 2, solve),
        Err(Error::NotRigidBlocks)
    ));
}
