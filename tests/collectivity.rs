//! Brüschweiler collectivity κ, cross-checked against the NOLB binary on crambin.
//! NOLB's `--analyze` reports κ for the lowest non-zero modes (manual Eq 3.1/3.2);
//! elasticrab's full RTB solve numbers those the same way from mode 6 onward. The
//! formula (eigenvector squared per atom, divided by mass, normalized, entropy/N)
//! was confirmed against the manual.

use elasticrab::{Atom, NormalModes};

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

/// The `collectivity` column of `nolb_collectivity_crambin.txt`, indexed by NOLB mode.
fn nolb_collectivities() -> Vec<f64> {
    std::fs::read_to_string(format!("{DATA}/nolb_collectivity_crambin.txt"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .map(|l| l.split_whitespace().nth(1).unwrap().parse().unwrap())
        .collect()
}

#[test]
fn collectivity_is_bounded_and_matches_nolb() {
    let (atoms, blocks) = crambin();
    let modes = NormalModes::builder(&atoms)
        .cutoff(5.0)
        .blocks(&blocks)
        .mass_weighted()
        .solve()
        .unwrap();
    let n_connected = (atoms.len() - modes.disconnected().len()) as f64;

    // Every mode's κ lies in [1/N, 1].
    for i in 0..modes.len() {
        let kappa = modes.collectivity(i);
        assert!(
            kappa >= 1.0 / n_connected - 1e-9 && kappa <= 1.0 + 1e-9,
            "mode {i}: κ {kappa} out of [1/N, 1]"
        );
    }

    // NOLB mode [k] (1-based) is elasticrab's mode 5+k. The shapes differ slightly
    // (same ~0.1% tessellation difference as the frequency cross-check), so match
    // to a loose absolute tolerance — the point is the formula and mode ordering,
    // including the two localized modes (NOLB 3 and 7, κ < 0.2).
    let nolb = nolb_collectivities();
    for (k, &reference) in nolb.iter().enumerate() {
        let ours = modes.collectivity(k + 6);
        assert!(
            (ours - reference).abs() < 0.03,
            "NOLB mode {} (elasticrab {}): κ {ours} vs NOLB {reference}",
            k + 1,
            k + 6
        );
    }
    assert!(
        nolb[2] < 0.2 && nolb[6] < 0.2,
        "localized modes should be low κ"
    );
}
