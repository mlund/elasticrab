//! CLI smoke tests: spawn the built `elasticrab` binary and check each subcommand
//! wires up. Run under `--features cli` (the binary needs it). The numerics are
//! covered by the library/golden tests; here we only pin the command grammar.
#![cfg(feature = "cli")]

use std::path::PathBuf;
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

fn temp_file(name: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "elasticrab_cli_{name}_{}.{}",
        std::process::id(),
        ext
    ))
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
fn animate_writes_combined_nmd_when_requested() {
    let nmd = temp_file("modes with spaces", "nmd");
    let out = run(&[
        "-i",
        &input(),
        "-s",
        "0",
        "-o",
        nmd.to_str().unwrap(),
        "animate",
        "--mode",
        "1",
        "--mode",
        "3",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&nmd).unwrap();
    let first_line = text.lines().next().unwrap_or_default();
    assert!(first_line.starts_with("nmwiz_load "), "{text}");
    assert!(first_line.contains("\\ "), "{first_line}");
    assert!(text.contains("\ncoordinates "), "{text}");
    assert!(text.contains("\nmode 1 "), "{text}");
    assert!(text.contains("\nmode 3 "), "{text}");
    assert!(!text.contains("\nmode 2 "), "{text}");

    let atom_count = nmd_field_count(&text, "atomnames");
    assert!(atom_count > 0, "{text}");
    assert_eq!(nmd_field_count(&text, "coordinates"), 3 * atom_count);
    for mode in ["mode 1", "mode 3"] {
        // After `mode N`, NMD stores the NMWiz scale and a 3-vector per atom.
        assert_eq!(nmd_field_count(&text, mode), 1 + 3 * atom_count);
    }
    assert!(
        stdout(&out).contains(nmd.to_str().unwrap()),
        "{}",
        stdout(&out)
    );
    let _ = std::fs::remove_file(&nmd);
}

fn nmd_field_count(text: &str, label: &str) -> usize {
    text.lines()
        .find_map(|line| line.strip_prefix(label))
        .map(|rest| rest.split_whitespace().count())
        .unwrap_or(0)
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
    let csv = temp_file("energy", "csv");
    let pdb = temp_file("energy", "pdb");
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
    let csv = temp_file("out_of_range", "csv");
    let csv = csv.to_string_lossy().into_owned();
    // -n beyond the spectrum: a clean CLI error (exit 1), never an out-of-bounds panic.
    for verb in [
        &["-i", &input(), "-n", "9999", "animate"][..],
        &["-i", &input(), "-n", "9999", "energy", "--csv", &csv],
    ] {
        let out = run(verb);
        assert_eq!(out.status.code(), Some(1), "{verb:?}: {}", stdout(&out));
        assert!(String::from_utf8_lossy(&out.stderr).contains("non-zero modes exist"));
    }
    let _ = std::fs::remove_file(&csv);
}

#[test]
fn zero_modes_are_rejected_on_every_verb() {
    let target = format!("{DATA}/crambin_hinge.pdb");
    let csv = temp_file("zero_modes", "csv");
    let csv = csv.to_string_lossy().into_owned();
    for verb in [
        &["-i", &input(), "-n", "0", "animate"][..],
        &["-i", &input(), "-n", "0", "transition", "--target", &target],
        &["-i", &input(), "-n", "0", "energy", "--csv", &csv],
    ] {
        assert!(!run(verb).status.success(), "{verb:?} should reject -n 0");
    }
    let _ = std::fs::remove_file(&csv);
}

#[test]
fn energy_honors_mode_selection() {
    let csv = temp_file("mode_sel", "csv");
    let pdb = temp_file("mode_sel", "pdb");
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
