//! Animate a normal mode of a protein into a multi-model PDB trajectory.
//!
//! ```text
//! cargo run --example animate_pdb -- <in.pdb> [amplitude=2.0] [mode=6] [frames=20] > mode.pdb
//! ```
//!
//! Reads a PDB, builds a mass-weighted Rotation-Translation-Blocks model (one
//! rigid block per residue, the NOLB / Pepsi-SAXS convention), then writes
//! `frames` `MODEL` records that sweep the structure back and forth along one
//! mode — load the result in PyMOL or VMD to watch the motion. Mode 6 (the first
//! non-zero mode, after the six rigid-body ones) is the softest, most visible
//! collective motion.
//!
//! The PDB column parsing and atom filtering are adapted from voronota-ltr
//! (`src/input/pdb.rs`, MIT, K. Olechnovic & M. Lund).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::process::ExitCode;

use elasticrab::{Atom, NormalModes, Params};

/// A parsed `ATOM` record: the fields needed both to solve and to re-emit the line.
struct Record {
    serial: i32,
    name: String,
    res_name: String,
    chain_id: String,
    res_seq: i32,
    i_code: String,
    element: String,
    position: [f64; 3],
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: animate_pdb <in.pdb> [amplitude=2.0] [mode=6] [frames=20]");
        return ExitCode::FAILURE;
    };
    let amplitude = parse_arg(&args, 2, 2.0);
    let mode = parse_arg(&args, 3, 6);
    let frames = parse_arg(&args, 4, 20);

    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let records = parse_pdb(&text);
    if records.len() < 2 {
        eprintln!("need at least two atoms; parsed {}", records.len());
        return ExitCode::FAILURE;
    }

    let atoms: Vec<Atom> = records
        .iter()
        .map(|r| Atom {
            position: r.position,
            mass: element_mass(&r.element),
        })
        .collect();
    let blocks = residue_blocks(&records);
    let positions: Vec<[f64; 3]> = records.iter().map(|r| r.position).collect();

    // Mass-weighted RTB at the 5 Å all-atom cutoff, as NOLB / Pepsi-SAXS use.
    let mut params = Params::default();
    params.cutoff = 5.0;
    params.mass_weighted = true;
    let modes = match NormalModes::with_blocks(&atoms, &blocks, &params) {
        Ok(modes) => modes,
        Err(e) => {
            eprintln!("normal-mode analysis failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    if mode >= modes.len() {
        eprintln!("mode {mode} out of range (only {} modes)", modes.len());
        return ExitCode::FAILURE;
    }
    if !modes.disconnected().is_empty() {
        eprintln!(
            "note: dropped {} disconnected atom(s)",
            modes.disconnected().len()
        );
    }

    let mut out = String::new();
    for frame in 0..frames {
        // sin sweep: a smooth there-and-back loop that starts at the input.
        let phase = std::f64::consts::TAU * frame as f64 / frames as f64;
        let displaced = modes.displace(&positions, mode, amplitude * phase.sin());
        let _ = writeln!(out, "MODEL     {:>4}", frame + 1);
        for (record, &p) in records.iter().zip(&displaced) {
            write_atom_line(&mut out, record, p);
        }
        out.push_str("ENDMDL\n");
    }
    print!("{out}");
    ExitCode::SUCCESS
}

fn parse_arg<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> T {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// Standard atomic weights for the common protein elements; a neutral fallback
/// covers anything else (mass-weighting only rescales the spectrum).
fn element_mass(element: &str) -> f64 {
    match element {
        "N" => 14.007,
        "O" => 15.999,
        "S" => 32.06,
        "P" => 30.974,
        _ => 12.011,
    }
}

/// One rigid block per residue, keyed by chain + residue number + insertion code
/// (the grouping voronota-ltr's `build_residue_grouping` produces).
fn residue_blocks(records: &[Record]) -> Vec<usize> {
    let mut group_of: HashMap<(&str, i32, &str), usize> = HashMap::new();
    records
        .iter()
        .map(|r| {
            let next = group_of.len();
            *group_of
                .entry((&r.chain_id, r.res_seq, &r.i_code))
                .or_insert(next)
        })
        .collect()
}

/// Parse `ATOM` records, keeping the heavy protein atoms an ANM expects:
/// `HETATM`/water are skipped (so every residue is a ≥4-atom amino acid, with no
/// collinear rigid blocks), hydrogens are skipped, and only one altLoc is kept
/// (avoiding coincident atoms that would divide by zero in the Hessian).
fn parse_pdb(text: &str) -> Vec<Record> {
    let mut records = Vec::new();
    for line in text.lines() {
        if column(line, 1, 6) != "ATOM" {
            continue;
        }
        let alt_loc = column(line, 17, 17);
        if !alt_loc.is_empty() && alt_loc != "A" && alt_loc != "1" {
            continue;
        }
        let name = column(line, 13, 16);
        let element = column(line, 77, 78);
        if name.starts_with('H') || element == "H" || element == "D" {
            continue;
        }
        let (Some(x), Some(y), Some(z)) = (
            column_f64(line, 31, 38),
            column_f64(line, 39, 46),
            column_f64(line, 47, 54),
        ) else {
            continue;
        };
        let (Some(serial), Some(res_seq)) = (column_i32(line, 7, 11), column_i32(line, 23, 26))
        else {
            continue;
        };
        records.push(Record {
            serial,
            name: name.to_string(),
            res_name: column(line, 18, 20).to_string(),
            chain_id: column(line, 22, 22).to_string(),
            res_seq,
            i_code: column(line, 27, 27).to_string(),
            element: element.to_string(),
            position: [x, y, z],
        });
    }
    records
}

/// A 1-indexed, inclusive, trimmed PDB column range (the voronota-ltr helper).
fn column(line: &str, start: usize, end: usize) -> &str {
    let end = end.min(line.len());
    let start = start.saturating_sub(1);
    if start >= line.len() {
        return "";
    }
    line.get(start..end).unwrap_or("").trim()
}

fn column_f64(line: &str, start: usize, end: usize) -> Option<f64> {
    column(line, start, end).parse().ok()
}

fn column_i32(line: &str, start: usize, end: usize) -> Option<i32> {
    column(line, start, end).parse().ok()
}

/// Append a record as a fixed-column `ATOM` line at new coordinates — the writer
/// voronota-ltr's parser does not include.
fn write_atom_line(out: &mut String, r: &Record, p: [f64; 3]) {
    let chain = r.chain_id.chars().next().unwrap_or(' ');
    let icode = r.i_code.chars().next().unwrap_or(' ');
    let _ = writeln!(
        out,
        "ATOM  {:>5} {:<4} {:>3} {}{:>4}{}   {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}",
        r.serial, r.name, r.res_name, chain, r.res_seq, icode, p[0], p[1], p[2], r.element,
    );
}
