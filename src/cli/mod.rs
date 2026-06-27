//! The `elasticrab` command-line tool: read a structure, run rigid-block NMA, and
//! animate the lowest modes into PDB/XTC trajectories. Interface modelled on
//! NOLB but with idiomatic names and 1-indexed (rigid-body-free) modes.

mod io;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use elasticrab::{Atom, NormalModes, Params};
use voronota_ltr::input::{
    build_residue_grouping, parse_file_with_records, AtomRecord, ParseOptions, RadiiLookup,
    Selection,
};

/// Normal-mode analysis: animate a protein's softest vibrational modes.
///
/// Reads a PDB or mmCIF structure, builds a mass-weighted rigid-block elastic
/// network, and writes a multi-model PDB (or XTC) trajectory per mode.
#[derive(Parser)]
#[command(name = "elasticrab", version, about, long_about = None)]
struct Cli {
    /// Input structure (PDB or mmCIF; format auto-detected).
    input: PathBuf,

    /// Spring interaction cutoff, in ångström.
    #[arg(short, long, default_value_t = 5.0, value_name = "ANGSTROM")]
    cutoff: f64,

    /// Animate the N lowest modes (1 = softest).
    ///
    /// Ignored when --mode is given.
    #[arg(short = 'n', long, default_value_t = 1, value_name = "N")]
    modes: usize,

    /// Specific mode to animate (1 = softest); repeatable.
    #[arg(long = "mode", value_name = "INDEX")]
    mode: Vec<usize>,

    /// Frames per trajectory (0 = report only).
    #[arg(short = 's', long, default_value_t = 20, value_name = "N")]
    frames: usize,

    /// Peak displacement RMSD, in ångström.
    #[arg(short = 'a', long, default_value_t = 1.5, value_name = "RMSD")]
    amplitude: f64,

    /// Use linear displacement, not the nonlinear default.
    ///
    /// Straight-line motion stretches bonds; nonlinear keeps them rigid.
    #[arg(long)]
    linear: bool,

    /// Include HETATM records (ligands, ions).
    ///
    /// Waters (HOH) are always dropped by the parser.
    #[arg(long)]
    hetatm: bool,

    /// Keep only atoms matching a VMD-like selection.
    ///
    /// For example, "chain A and name CA".
    #[arg(long, value_name = "EXPR")]
    select: Option<String>,

    /// Trajectory output path; format by `.pdb`/`.xtc` extension.
    ///
    /// Defaults to `<input>_mode<k>.pdb`, one file per mode.
    #[arg(short, long, value_name = "PATH")]
    output: Option<PathBuf>,

    /// Also write the report (frequencies, counts) as JSON to this file.
    #[arg(long, value_name = "FILE")]
    json: Option<PathBuf>,

    /// Merge all modes into one trajectory + a per-frame energy CSV.
    ///
    /// Columns: frame, mode, rmsd (Å), energy (Å², γ=1). Native frame first.
    ///
    /// Weight a frame by exp(−γ·energy / kT).
    #[arg(long, value_name = "FILE")]
    energy: Option<PathBuf>,
}

/// Entry point: set up diagnostics, parse arguments, run, and turn any error into
/// a clean message and a failing exit code.
pub fn run() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    match execute(&Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn execute(cli: &Cli) -> Result<(), String> {
    let options = ParseOptions {
        exclude_heteroatoms: !cli.hetatm,
        ..Default::default()
    };
    let parsed = parse_file_with_records(&cli.input, &options, &RadiiLookup::new())
        .map_err(|e| format!("reading {}: {e}", cli.input.display()))?;
    let mut records = parsed.records;

    if let Some(expr) = &cli.select {
        let selection =
            Selection::parse(expr).map_err(|e| format!("invalid selection {expr:?}: {e}"))?;
        records.retain(|r| selection.matches(r));
    }
    if records.len() < 2 {
        return Err(format!(
            "need at least two atoms to build a network; found {}",
            records.len()
        ));
    }

    let positions: Vec<[f64; 3]> = records.iter().map(|r| [r.x, r.y, r.z]).collect();
    let atoms: Vec<Atom> = records
        .iter()
        .zip(&positions)
        .map(|(r, &position)| Atom {
            position,
            mass: io::element_mass(&r.element),
        })
        .collect();
    let blocks: Vec<usize> = build_residue_grouping(&records)
        .iter()
        .map(|&g| g as usize)
        .collect();

    let wanted = wanted_modes(cli)?;
    let k = *wanted.iter().max().expect("wanted is non-empty");

    let mut params = Params::default();
    params.cutoff = cli.cutoff;
    params.mass_weighted = true;
    params.k_modes = Some(k);
    let modes = NormalModes::with_blocks(&atoms, &blocks, &params)
        .map_err(|e| format!("normal-mode analysis failed: {e}"))?;
    for &m in &wanted {
        if m > modes.len() {
            return Err(format!(
                "mode {m} requested but only {} non-zero modes exist",
                modes.len()
            ));
        }
    }

    report(cli, &records, &blocks, &modes)?;

    if let Some(csv) = cli.energy.as_deref() {
        write_merged(cli, &modes, &positions, &records, &wanted, csv)?;
    } else if cli.frames > 0 {
        let multi = wanted.len() > 1;
        for &m in &wanted {
            let path = output_path(cli.output.as_deref(), &cli.input, m, multi);
            guard_input(&path, &cli.input)?;
            let frames = animate(&modes, &positions, m, cli.amplitude, cli.frames, cli.linear)?;
            write_trajectory(&path, &records, &frames)?;
        }
    }
    Ok(())
}

/// `--energy`: merge the native frame plus every mode's frames into one
/// trajectory and write the matching per-frame energy table. The energies are
/// the elastic-network spring energy of each frame (native = 0), comparable
/// across modes because the energy depends only on the coordinates.
fn write_merged(
    cli: &Cli,
    modes: &NormalModes,
    positions: &[[f64; 3]],
    records: &[AtomRecord],
    wanted: &[usize],
    csv: &Path,
) -> Result<(), String> {
    if cli.frames == 0 {
        return Err("--energy needs --frames greater than 0 (nothing to score otherwise)".into());
    }
    // Resolve and check every output path before animating, so a clobbering
    // mistake fails fast and never destroys the input or one output with another.
    let traj = cli.output.as_deref().map_or_else(
        || with_stem(&cli.input, |stem| format!("{stem}_modes.pdb")),
        Path::to_path_buf,
    );
    guard_input(&traj, &cli.input)?;
    guard_input(csv, &cli.input)?;
    if same_path(csv, &traj) {
        return Err(format!(
            "the energy table and the trajectory cannot be the same file ({})",
            csv.display()
        ));
    }

    // Frame 0 is the native structure — the energy zero and the MC rest state.
    let mut frames = vec![positions.to_vec()];
    let mut rows = vec![io::EnergyRow {
        frame: 0,
        mode: 0,
        rmsd: 0.0,
        energy: modes.energy(positions),
    }];
    for &m in wanted {
        for frame in animate(modes, positions, m, cli.amplitude, cli.frames, cli.linear)? {
            rows.push(io::EnergyRow {
                frame: frames.len(),
                mode: m,
                rmsd: rms_deviation(&frame, positions),
                energy: modes.energy(&frame),
            });
            frames.push(frame);
        }
    }

    write_trajectory(&traj, records, &frames)?;
    io::write_csv(csv, &rows)
}

/// Refuse to write a trajectory over the input structure.
fn guard_input(output: &Path, input: &Path) -> Result<(), String> {
    if same_path(output, input) {
        return Err(format!(
            "refusing to overwrite the input structure {}",
            input.display()
        ));
    }
    Ok(())
}

/// The 1-indexed modes to animate: `--mode` if given, otherwise `1..=modes`.
fn wanted_modes(cli: &Cli) -> Result<Vec<usize>, String> {
    let wanted = if cli.mode.is_empty() {
        (1..=cli.modes).collect::<Vec<_>>()
    } else {
        cli.mode.clone()
    };
    if wanted.is_empty() {
        return Err("no modes requested (use -n >= 1 or --mode)".into());
    }
    if wanted.contains(&0) {
        return Err("mode indices are 1-based; 0 is not a mode".into());
    }
    Ok(wanted)
}

/// Frames sweeping mode `mode` (1-indexed) back and forth, scaled so the peak
/// frame reaches `peak_rmsd` ångström from the input.
fn animate(
    modes: &NormalModes,
    positions: &[[f64; 3]],
    mode: usize,
    peak_rmsd: f64,
    frames: usize,
    linear: bool,
) -> Result<Vec<Vec<[f64; 3]>>, String> {
    let i = mode - 1;
    let displace = |factor: f64| {
        if linear {
            Ok(modes.displace(positions, i, factor))
        } else {
            modes
                .displace_nonlinear(positions, i, factor)
                .map_err(|e| format!("nonlinear displacement: {e}"))
        }
    };
    // Calibrate with the same displacement the frames use, so the requested peak
    // RMSD is honoured on the nonlinear path too (factor 1.0 stays in the linear
    // regime, the unit eigenvector being tiny).
    let scale = peak_rmsd / rms_deviation(&displace(1.0)?, positions);
    // Offset the phase by half a step: a 1- or 2-frame sweep then still samples
    // the moving extremes instead of landing only on sin = 0 (a motionless run).
    (0..frames)
        .map(|f| {
            let phase = std::f64::consts::TAU * (f as f64 + 0.5) / frames as f64;
            displace(scale * phase.sin())
        })
        .collect()
}

fn rms_deviation(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    let total: f64 = a
        .iter()
        .zip(b)
        .map(|(p, q)| (0..3).map(|c| (p[c] - q[c]).powi(2)).sum::<f64>())
        .sum();
    (total / a.len() as f64).sqrt()
}

fn write_trajectory(
    path: &Path,
    records: &[AtomRecord],
    frames: &io::Trajectory,
) -> Result<(), String> {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("xtc"))
    {
        io::write_xtc(path, frames)
    } else {
        io::write_pdb(path, records, frames)
    }
}

/// Where mode `mode`'s trajectory is written: the explicit `-o` for a single
/// mode, `_mode<k>` inserted when several modes share a prefix, else a default
/// name beside the input.
fn output_path(output: Option<&Path>, input: &Path, mode: usize, multi: bool) -> PathBuf {
    match output {
        Some(path) if !multi => path.to_path_buf(),
        Some(path) => {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("pdb");
            with_stem(path, |stem| format!("{stem}_mode{mode}.{ext}"))
        }
        None => with_stem(input, |stem| format!("{stem}_mode{mode}.pdb")),
    }
}

/// Whether two paths point at the same file, resolving each path's parent so a
/// relative `-o protein.pdb` is recognised as the input `protein.pdb`.
fn same_path(a: &Path, b: &Path) -> bool {
    fn resolved(path: &Path) -> PathBuf {
        let parent = match path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        let name = path.file_name().unwrap_or(path.as_os_str());
        parent
            .canonicalize()
            .map_or_else(|_| path.to_path_buf(), |dir| dir.join(name))
    }
    resolved(a) == resolved(b)
}

fn with_stem(path: &Path, f: impl FnOnce(&str) -> String) -> PathBuf {
    let mut out = path.to_path_buf();
    out.set_file_name(f(&file_stem(path)));
    out
}

fn file_stem(path: &Path) -> String {
    path.file_stem().map_or_else(
        || "structure".to_string(),
        |s| s.to_string_lossy().into_owned(),
    )
}

/// Print the human-readable report to stdout, and write JSON to `--json` if set.
fn report(
    cli: &Cli,
    records: &[AtomRecord],
    blocks: &[usize],
    modes: &NormalModes,
) -> Result<(), String> {
    let residues = blocks.iter().copied().max().map_or(0, |m| m + 1);
    let frequencies: Vec<f64> = modes.eigenvalues().iter().map(|&l| l.sqrt()).collect();

    println!("elasticrab — {}", cli.input.display());
    println!(
        "  atoms {}, residues {residues}, dropped {}",
        records.len(),
        modes.disconnected().len()
    );
    println!("  cutoff {} Å, mass-weighted", cli.cutoff);
    println!("  mode  frequency");
    for (j, frequency) in frequencies.iter().enumerate() {
        println!("  {:>4}  {frequency:.6}", j + 1);
    }

    if let Some(path) = &cli.json {
        let json = report_json(cli, records.len(), residues, modes, &frequencies);
        std::fs::write(path, json).map_err(|e| format!("writing {}: {e}", path.display()))?;
    }
    Ok(())
}

/// The report as a JSON object — all numeric/boolean but the (escaped) input
/// path, so a hand-written writer avoids a `serde` dependency.
fn report_json(
    cli: &Cli,
    atoms: usize,
    residues: usize,
    modes: &NormalModes,
    frequencies: &[f64],
) -> String {
    let dropped: Vec<String> = modes.disconnected().iter().map(usize::to_string).collect();
    let mut s = String::from("{\n");
    let _ = writeln!(
        s,
        "  \"input\": {},",
        json_string(&cli.input.to_string_lossy())
    );
    let _ = writeln!(s, "  \"atoms\": {atoms},");
    let _ = writeln!(s, "  \"residues\": {residues},");
    let _ = writeln!(s, "  \"dropped\": [{}],", dropped.join(", "));
    let _ = writeln!(s, "  \"cutoff\": {},", cli.cutoff);
    let _ = writeln!(s, "  \"mass_weighted\": true,");
    s.push_str("  \"modes\": [\n");
    let eigenvalues = modes.eigenvalues();
    for (j, (frequency, eigenvalue)) in frequencies.iter().zip(eigenvalues).enumerate() {
        let comma = if j + 1 < frequencies.len() { "," } else { "" };
        let _ = writeln!(
            s,
            "    {{\"index\": {}, \"frequency\": {frequency}, \"eigenvalue\": {eigenvalue}}}{comma}",
            j + 1
        );
    }
    s.push_str("  ]\n}\n");
    s
}

/// A JSON string literal, escaping the quote, backslash, and control characters
/// (a path may legally contain a tab or newline on Unix).
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
