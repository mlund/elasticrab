//! CLI smoke tests: spawn the built `elasticrab` binary and check each subcommand
//! wires up. Run under `--features cli` (the binary needs it). The numerics are
//! covered by the library/golden tests; here we only pin the command grammar.
#![cfg(feature = "cli")]

use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_elasticrab");
const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

fn run(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .output()
        .expect("failed to run the elasticrab binary")
}

fn input() -> String {
    format!("{DATA}/crambin_heavy.pdb")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn animate_reports_the_spectrum() {
    let out = run(&["-i", &input(), "-n", "2", "-s", "0", "animate"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout(&out).contains("mode  frequency  collectivity"),
        "{}",
        stdout(&out)
    );
}

#[test]
fn transition_reports_overlaps() {
    let target = format!("{DATA}/crambin_hinge.pdb");
    let out = run(&[
        "-i",
        &input(),
        "-n",
        "3",
        "-s",
        "0",
        "transition",
        "--target",
        &target,
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout(&out).contains("mass-weighted modes"),
        "{}",
        stdout(&out)
    );
}

#[test]
fn iterative_transition_reports_the_ladder() {
    let target = format!("{DATA}/crambin_hinge_large.pdb");
    let out = run(&[
        "-i",
        &input(),
        "-n",
        "5",
        "-s",
        "0",
        "transition",
        "--target",
        &target,
        "--n-iter",
        "1",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout(&out).contains("re-diagonalization"),
        "{}",
        stdout(&out)
    );
}

#[test]
fn energy_writes_the_csv() {
    let dir = std::env::temp_dir();
    let csv = dir.join("elasticrab_cli_energy.csv");
    let pdb = dir.join("elasticrab_cli_energy.pdb");
    let out = run(&[
        "-i",
        &input(),
        "-n",
        "2",
        "-s",
        "2",
        "-o",
        pdb.to_str().unwrap(),
        "energy",
        "--csv",
        csv.to_str().unwrap(),
        "--voromqa",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(csv.exists(), "energy CSV not written");
    let _ = std::fs::remove_file(&csv);
    let _ = std::fs::remove_file(&pdb);
}

#[test]
fn every_verb_has_help() {
    for args in [
        &["--help"][..],
        &["animate", "--help"],
        &["transition", "--help"],
        &["energy", "--help"],
    ] {
        let out = run(args);
        assert!(out.status.success(), "{args:?} failed");
        assert!(stdout(&out).contains("Usage"), "{args:?}: {}", stdout(&out));
    }
}

#[test]
fn missing_verb_or_required_arg_fails() {
    // No subcommand: help is shown and the exit is non-success.
    assert!(!run(&[]).status.success());
    // transition without --target: clap rejects the missing required option.
    assert!(!run(&["-i", &input(), "transition"]).status.success());
}

#[test]
fn out_of_range_modes_fail_cleanly_not_panic() {
    // -n beyond the spectrum: a clean CLI error (exit 1), never an out-of-bounds panic.
    for verb in [
        &["-i", &input(), "-n", "9999", "animate"][..],
        &["-i", &input(), "-n", "9999", "energy", "--csv", "/dev/null"],
    ] {
        let out = run(verb);
        assert_eq!(out.status.code(), Some(1), "{verb:?}: {}", stdout(&out));
        assert!(String::from_utf8_lossy(&out.stderr).contains("non-zero modes exist"));
    }
}

#[test]
fn zero_modes_are_rejected_on_every_verb() {
    let target = format!("{DATA}/crambin_hinge.pdb");
    for verb in [
        &["-i", &input(), "-n", "0", "animate"][..],
        &["-i", &input(), "-n", "0", "transition", "--target", &target],
        &["-i", &input(), "-n", "0", "energy", "--csv", "/dev/null"],
    ] {
        assert!(!run(verb).status.success(), "{verb:?} should reject -n 0");
    }
}

#[test]
fn energy_honors_mode_selection() {
    let dir = std::env::temp_dir();
    let csv = dir.join("elasticrab_cli_mode_sel.csv");
    let pdb = dir.join("elasticrab_cli_mode_sel.pdb");
    let out = run(&[
        "-i",
        &input(),
        "-s",
        "2",
        "-o",
        pdb.to_str().unwrap(),
        "energy",
        "--csv",
        csv.to_str().unwrap(),
        "--mode",
        "6",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let table = std::fs::read_to_string(&csv).unwrap();
    // Every swept (non-native) row is mode 6, not the default 1..=n.
    assert!(table.lines().any(|l| l.starts_with("1,6,")), "{table}");
    assert!(!table.lines().any(|l| l.starts_with("1,1,")), "{table}");
    let _ = std::fs::remove_file(&csv);
    let _ = std::fs::remove_file(&pdb);
}
