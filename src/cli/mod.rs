//! The `elasticrab` command-line tool: read a structure, run rigid-block NMA, and
//! animate the lowest modes into PDB/XTC trajectories. Interface modelled on
//! NOLB but with idiomatic names and 1-indexed (rigid-body-free) modes.

mod io;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use elasticrab::{Atom, Builder, NormalModes, Spring};
use voronota_ltr::input::{
    build_residue_grouping, parse_file_with_records, AtomRecord, ParseOptions, RadiiLookup,
    Selection,
};
use voronota_ltr::{compute_contacts_only, Ball};

/// Default spring constant (kJ/mol/Å²): the median B-factor-fitted γ over a small
/// high-resolution PDB set (`scripts/calibrate-gamma.sh`). The fit is noisy across
/// structures, so for quantitative work pass `--b-factor-fit` or your own `--gamma`.
const DEFAULT_GAMMA: f64 = 11.5;

/// Boltzmann constant in kJ·mol⁻¹·K⁻¹, matching γ in kJ/mol/Å².
const BOLTZMANN_KJ_PER_MOL_K: f64 = 8.314_462_618e-3;

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

    /// Build springs from a Voronoi tessellation (contact-area weighted).
    ///
    /// Mutually exclusive with --cutoff.
    #[arg(long, conflicts_with = "cutoff")]
    voronota: bool,

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

    /// Write the report as JSON to this file.
    #[arg(long, value_name = "FILE")]
    json: Option<PathBuf>,

    /// Merge modes into one trajectory + an MC energy CSV.
    ///
    /// Each mode is sampled at thermal amplitudes (±--sigmas σ), not --amplitude.
    ///
    /// Columns: frame, mode, rmsd, energy (γ=1, Å²), energy_kJ_mol, weight.
    #[arg(long, value_name = "FILE")]
    energy: Option<PathBuf>,

    /// Spring constant γ (kJ/mol/Å²).
    ///
    /// Scales the energy and weight columns; the default is B-factor-calibrated.
    #[arg(short = 'g', long, default_value_t = DEFAULT_GAMMA, value_name = "VALUE")]
    gamma: f64,

    /// Temperature, in kelvin.
    ///
    /// Sets the fluctuations and the Boltzmann weights.
    #[arg(short = 'T', long, default_value_t = 298.15, value_name = "KELVIN")]
    temperature: f64,

    /// Fit γ to the input's B-factors; overrides --gamma.
    ///
    /// Runs a dense all-atom solve (memory-heavy for very large structures); on
    /// failure it warns and falls back to --gamma.
    #[arg(long)]
    b_factor_fit: bool,

    /// Thermal sampling width for --energy, in σ.
    ///
    /// Each mode is swept over ±N·σ of its own thermal fluctuation, so the pool
    /// is Boltzmann-relevant (peak energy ≈ ½N²·kT).
    #[arg(long, default_value_t = 3.0, value_name = "N")]
    sigmas: f64,
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
    for (name, value) in [
        ("--gamma", cli.gamma),
        ("--temperature", cli.temperature),
        ("--sigmas", cli.sigmas),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("{name} must be a positive number (got {value})"));
        }
    }

    let options = ParseOptions {
        exclude_heteroatoms: !cli.hetatm,
        ..Default::default()
    };
    // One radii table, shared by the parser and the tessellation (--voronota).
    let radii = RadiiLookup::new();
    let parsed = parse_file_with_records(&cli.input, &options, &radii)
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

    // Tessellation springs (when --voronota); otherwise the distance cutoff is used.
    let springs = cli
        .voronota
        .then(|| voronota_springs(&records, &radii))
        .transpose()?;

    let wanted = wanted_modes(cli)?;
    let k = *wanted.iter().max().expect("wanted is non-empty");

    let modes = with_connectivity(
        NormalModes::builder(&atoms)
            .mass_weighted()
            .k_modes(k)
            .blocks(&blocks),
        cli.cutoff,
        springs.as_deref(),
    )
    .solve()
    .map_err(|e| format!("normal-mode analysis failed: {e}"))?;
    for &m in &wanted {
        if m > modes.len() {
            return Err(format!(
                "mode {m} requested but only {} non-zero modes exist",
                modes.len()
            ));
        }
    }

    let (gamma, fit_r) = effective_gamma(cli, &atoms, &records, springs.as_deref())?;
    report(cli, &records, &blocks, &modes, gamma, fit_r)?;

    if let Some(csv) = cli.energy.as_deref() {
        write_merged(cli, &modes, &positions, &records, &wanted, csv, gamma)?;
    } else if cli.frames > 0 {
        let multi = wanted.len() > 1;
        for &m in &wanted {
            let path = output_path(cli.output.as_deref(), &cli.input, m, multi);
            guard_input(&path, &cli.input)?;
            let frames = animate(&modes, &positions, m, cli.amplitude, cli.frames, cli.linear)?;
            write_trajectory(&path, &records, &frames)?;
            println!("  wrote {}", path.display());
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
    gamma: f64,
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

    // Build a row, deriving the real energy (γ·E_geometric) and Boltzmann weight
    // (native E=0 ⇒ weight 1, the maximum) from the geometric γ=1 energy.
    let kt = BOLTZMANN_KJ_PER_MOL_K * cli.temperature;
    let row = |frame, mode, rmsd, energy: f64| {
        let energy_kj = gamma * energy;
        io::EnergyRow {
            frame,
            mode,
            rmsd,
            energy,
            energy_kj,
            weight: (-energy_kj / kt).exp(),
        }
    };

    // Frame 0 is the native structure — the energy zero and the MC rest state.
    let mut frames = vec![positions.to_vec()];
    let mut rows = vec![row(0, 0, 0.0, modes.energy(positions))];
    for &m in wanted {
        for frame in thermal_frames(modes, positions, m, gamma, cli)? {
            let energy = modes.energy(&frame);
            rows.push(row(
                frames.len(),
                m,
                rms_deviation(&frame, positions),
                energy,
            ));
            frames.push(frame);
        }
    }

    write_trajectory(&traj, records, &frames)?;
    io::write_csv(csv, &rows)?;
    println!("  wrote {} and {}", traj.display(), csv.display());
    Ok(())
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

/// Probe radius (ångström) for the Voronoi tessellation — the conventional water
/// probe, as used by voronota and VoroMQA.
const VORONOTA_PROBE: f64 = 1.4;

/// Configure the builder's connectivity: explicit tessellation springs when
/// `--voronota` produced them, otherwise the distance cutoff. Shared by the main
/// solve and the B-factor fit so both use the same network.
fn with_connectivity<'a>(
    builder: Builder<'a>,
    cutoff: f64,
    springs: Option<&'a [Spring]>,
) -> Builder<'a> {
    match springs {
        Some(s) => builder.springs(s),
        None => builder.cutoff(cutoff),
    }
}

/// Springs from the Voronoi/Laguerre tessellation: one per pair of atoms whose
/// cells share a face, weighted by the contact area `Aᵢⱼ` normalized to unit mean
/// (`weight = Aᵢⱼ / mean A`, so `γᵢⱼ = γ₀ · weight`). Unit-mean keeps the *average*
/// spring at `γ₀`; the networks still differ in connectivity, so absolute
/// frequencies and energies are not identical to the cutoff network's.
fn voronota_springs(records: &[AtomRecord], radii: &RadiiLookup) -> Result<Vec<Spring>, String> {
    let balls: Vec<Ball> = records
        .iter()
        .map(|r| Ball::new(r.x, r.y, r.z, radii.get_radius(&r.res_name, &r.name)))
        .collect();
    let contacts = compute_contacts_only(&balls, VORONOTA_PROBE, None, None);
    if contacts.is_empty() {
        return Err("--voronota: tessellation produced no contacts".into());
    }
    let mean = contacts.iter().map(|c| c.area).sum::<f64>() / contacts.len() as f64;
    Ok(contacts
        .iter()
        .map(|c| Spring {
            i: c.id_a,
            j: c.id_b,
            weight: c.area / mean,
        })
        .collect())
}

/// The γ for the energy/weight columns and report: fitted from B-factors when
/// `--b-factor-fit` is set (returning its correlation too), else `--gamma`.
fn effective_gamma(
    cli: &Cli,
    atoms: &[Atom],
    records: &[AtomRecord],
    springs: Option<&[Spring]>,
) -> Result<(f64, Option<f64>), String> {
    if cli.b_factor_fit {
        match fit_gamma(cli, atoms, records, springs) {
            Ok((gamma, r)) => Ok((gamma, Some(r))),
            // A failed fit shouldn't suppress the report and trajectory the user
            // also asked for; warn and fall back to the manual γ.
            Err(message) => {
                eprintln!("warning: {message}; falling back to --gamma {}", cli.gamma);
                Ok((cli.gamma, None))
            }
        }
    } else {
        Ok((cli.gamma, None))
    }
}

/// Fit γ (kJ/mol/Å²) by scaling predicted ANM fluctuations to the input's
/// B-factors; returns `(γ, Pearson r)`. Uses a non-mass-weighted all-atom solve —
/// the correct, mass-independent configurational-fluctuation model — and the
/// through-origin least-squares `γ = Σ B₁² / Σ B₁·B^exp` (since `B ∝ 1/γ`).
fn fit_gamma(
    cli: &Cli,
    atoms: &[Atom],
    records: &[AtomRecord],
    springs: Option<&[Spring]>,
) -> Result<(f64, f64), String> {
    let modes = with_connectivity(NormalModes::builder(atoms), cli.cutoff, springs)
        .solve()
        .map_err(|e| format!("--b-factor-fit: {e}"))?;

    // Predicted B at γ=1, paired with experimental B over the connected atoms
    // (non-zero prediction) that actually carry a B-factor.
    let (mut predicted, mut experimental) = (Vec::new(), Vec::new());
    for (b_pred, record) in modes
        .predicted_b_factors(cli.temperature)
        .iter()
        .zip(records)
    {
        if *b_pred > 0.0 && record.b_factor > 0.0 {
            predicted.push(*b_pred);
            experimental.push(record.b_factor);
        }
    }
    let sum_pe: f64 = predicted
        .iter()
        .zip(&experimental)
        .map(|(p, e)| p * e)
        .sum();
    if predicted.len() < 2 || sum_pe <= 0.0 {
        return Err("--b-factor-fit: input has no usable B-factors; set --gamma instead".into());
    }
    let sum_pp: f64 = predicted.iter().map(|p| p * p).sum();
    Ok((sum_pp / sum_pe, pearson(&predicted, &experimental)))
}

/// Pearson correlation of two equal-length series (0 if either is constant).
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        let (dx, dy) = (a - mean_x, b - mean_y);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        0.0
    } else {
        sxy / (sxx * syy).sqrt()
    }
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

/// Mode `mode` (1-indexed) displaced by `factor` along its eigenvector — linear,
/// or the bond-preserving nonlinear extrapolation.
fn displace_at(
    modes: &NormalModes,
    positions: &[[f64; 3]],
    mode: usize,
    factor: f64,
    linear: bool,
) -> Result<Vec<[f64; 3]>, String> {
    let i = mode - 1;
    if linear {
        Ok(modes.displace(positions, i, factor))
    } else {
        modes
            .displace_nonlinear(positions, i, factor)
            .map_err(|e| format!("nonlinear displacement: {e}"))
    }
}

/// Frames sweeping mode `mode` over ±`peak` (in displace-factor units) through one
/// period. The quarter-step phase offset keeps every frame off the rest position,
/// so even a single-frame sweep is displaced.
fn sweep(
    modes: &NormalModes,
    positions: &[[f64; 3]],
    mode: usize,
    peak: f64,
    frames: usize,
    linear: bool,
) -> Result<Vec<Vec<[f64; 3]>>, String> {
    (0..frames)
        .map(|f| {
            let phase = std::f64::consts::TAU * (f as f64 + 0.25) / frames as f64;
            displace_at(modes, positions, mode, peak * phase.sin(), linear)
        })
        .collect()
}

/// Visualization sweep: scaled so the peak frame reaches `peak_rmsd` ångström
/// (factor 1.0 is a tiny displacement, so the nonlinear path stays linear there).
fn animate(
    modes: &NormalModes,
    positions: &[[f64; 3]],
    mode: usize,
    peak_rmsd: f64,
    frames: usize,
    linear: bool,
) -> Result<Vec<Vec<[f64; 3]>>, String> {
    let unit = displace_at(modes, positions, mode, 1.0, linear)?;
    let scale = peak_rmsd / rms_deviation(&unit, positions);
    sweep(modes, positions, mode, scale, frames, linear)
}

/// Monte-Carlo sweep: ±`--sigmas` of mode `mode`'s thermal width. Each mode's
/// stiffness `k = 2γ·E(unit)` (from the energy at the tiny unit displacement)
/// gives `σ = √(kT/k)`, so the pool is Boltzmann-relevant.
fn thermal_frames(
    modes: &NormalModes,
    positions: &[[f64; 3]],
    mode: usize,
    gamma: f64,
    cli: &Cli,
) -> Result<Vec<Vec<[f64; 3]>>, String> {
    let unit_energy = modes.energy(&displace_at(modes, positions, mode, 1.0, cli.linear)?);
    if !unit_energy.is_finite() || unit_energy <= 0.0 {
        return Err(format!(
            "mode {mode} has no restoring energy to sample thermally"
        ));
    }
    let kt = BOLTZMANN_KJ_PER_MOL_K * cli.temperature;
    let peak = cli.sigmas * (kt / (2.0 * gamma * unit_energy)).sqrt();
    sweep(modes, positions, mode, peak, cli.frames, cli.linear)
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
    gamma: f64,
    fit_r: Option<f64>,
) -> Result<(), String> {
    let residues = blocks.iter().copied().max().map_or(0, |m| m + 1);
    let frequencies: Vec<f64> = modes.eigenvalues().iter().map(|&l| l.sqrt()).collect();

    println!("elasticrab — {}", cli.input.display());
    println!(
        "  atoms {}, residues {residues}, dropped {}",
        records.len(),
        modes.disconnected().len()
    );
    if cli.voronota {
        println!(
            "  Voronoi tessellation, {} springs, mass-weighted",
            modes.spring_count()
        );
    } else {
        println!(
            "  cutoff {} Å, {} springs, mass-weighted",
            cli.cutoff,
            modes.spring_count()
        );
    }
    match fit_r {
        Some(r) => println!("  gamma {gamma:.4} kJ/mol/Å² (fitted, B-factor r = {r:.3})"),
        None => println!("  gamma {gamma:.4} kJ/mol/Å²"),
    }
    println!("  mode  frequency");
    for (j, frequency) in frequencies.iter().enumerate() {
        println!("  {:>4}  {frequency:.6}", j + 1);
    }

    if let Some(path) = &cli.json {
        let json = report_json(
            cli,
            records.len(),
            residues,
            modes,
            &frequencies,
            gamma,
            fit_r,
        );
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
    gamma: f64,
    fit_r: Option<f64>,
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
    // `network` is always present as the discriminator; `cutoff` only when it applies.
    if cli.voronota {
        let _ = writeln!(s, "  \"network\": \"voronota\",");
    } else {
        let _ = writeln!(s, "  \"network\": \"cutoff\",");
        let _ = writeln!(s, "  \"cutoff\": {},", cli.cutoff);
    }
    let _ = writeln!(s, "  \"springs\": {},", modes.spring_count());
    let _ = writeln!(s, "  \"mass_weighted\": true,");
    let _ = writeln!(s, "  \"gamma\": {gamma},");
    if let Some(r) = fit_r {
        let _ = writeln!(s, "  \"b_factor_correlation\": {r},");
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The tessellation yields a connected, area-weighted spring network on a real
    /// structure: more springs than atoms, every weight finite and positive, every
    /// endpoint in range.
    #[test]
    fn voronota_springs_on_crambin() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/crambin_heavy.pdb");
        let options = ParseOptions {
            exclude_heteroatoms: true,
            ..Default::default()
        };
        let radii = RadiiLookup::new();
        let parsed = parse_file_with_records(Path::new(path), &options, &radii).unwrap();
        let n = parsed.records.len();

        let springs = voronota_springs(&parsed.records, &radii).unwrap();
        assert!(
            springs.len() > n,
            "expected more springs ({}) than atoms ({n})",
            springs.len()
        );
        for s in &springs {
            assert!(s.i < n && s.j < n && s.i != s.j, "bad endpoints: {s:?}");
            assert!(s.weight.is_finite() && s.weight > 0.0, "bad weight: {s:?}");
        }
    }
}
