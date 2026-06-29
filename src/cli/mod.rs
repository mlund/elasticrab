//! The `elasticrab` command-line tool: read a structure, run rigid-block NMA, and
//! either `animate` the lowest modes, morph toward a target (`transition`), or build a
//! Monte-Carlo `energy` table — one named subcommand each, over a shared set of
//! network/solve options. 1-indexed, rigid-body-free modes.

mod io;
mod voromqa;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use elasticrab::{
    transition_iterative, Atom, Builder, IterativeTransition, NormalModes, Spring, Transition,
};
use voronota_ltr::input::{
    build_residue_grouping, parse_file_with_records, AtomRecord, ParseOptions, RadiiLookup,
    Selection,
};
use voronota_ltr::{compute_contacts_only, Ball};

/// Default spring constant (kJ/mol/Å²): the median B-factor-fitted γ over a small
/// high-resolution PDB set (`scripts/calibrate-gamma.sh`). The fit is noisy across
/// structures, so for quantitative work pass `--b-factor-fit` or your own `--gamma`.
const DEFAULT_GAMMA: f64 = 11.5;

/// Molar gas constant R = N_A·k_B, in kJ·mol⁻¹·K⁻¹. Energies here are per-mole
/// (kJ/mol, via γ), so the Boltzmann factor uses RT, not the per-particle kT.
const GAS_CONSTANT_KJ_PER_MOL_K: f64 = 8.314_462_618e-3;

/// Normal-mode analysis of a protein: animate modes, morph toward a target, or
/// build a Monte-Carlo energy table — one named subcommand each.
#[derive(Parser)]
#[command(
    name = "elasticrab",
    version,
    about,
    long_about = None,
    subcommand_required = true,
    arg_required_else_help = true
)]
struct Cli {
    /// The network/solve options shared by every verb — shown here at the top level,
    /// so each `<verb> --help` lists only that verb's own options.
    #[command(flatten)]
    solve: SolveArgs,
    #[command(subcommand)]
    command: Command,
}

/// The network + solve options every verb shares (mass-weighted rigid-block NMA).
#[derive(clap::Args)]
struct SolveArgs {
    /// Input structure (PDB or mmCIF; format auto-detected).
    #[arg(short = 'i', long, value_name = "PATH")]
    input: PathBuf,

    /// Spring interaction cutoff, in ångström.
    #[arg(short = 'c', long, default_value_t = 5.0, value_name = "ANGSTROM")]
    cutoff: f64,

    /// Build springs from a Voronoi tessellation (contact-area weighted). Mutually
    /// exclusive with --cutoff.
    #[arg(long, conflicts_with = "cutoff")]
    voronota: bool,

    /// The N lowest modes (1 = softest).
    #[arg(short = 'n', long = "modes", default_value_t = 1, value_name = "N")]
    modes: usize,

    /// Frames per trajectory (0 = report only).
    #[arg(short = 's', long, default_value_t = 20, value_name = "N")]
    frames: usize,

    /// Use linear displacement, not the nonlinear default (which keeps bonds rigid).
    #[arg(long)]
    linear: bool,

    /// Include HETATM records (ligands, ions). Waters (HOH) are always dropped.
    #[arg(long)]
    hetatm: bool,

    /// Keep only atoms matching a VMD-like selection, e.g. "chain A and name CA".
    #[arg(long, value_name = "EXPR")]
    select: Option<String>,

    /// Output path; `.pdb`/`.xtc` for trajectories, `.nmd` for animate/NMWiz modes.
    #[arg(short = 'o', long, value_name = "PATH")]
    output: Option<PathBuf>,

    /// Write the report as JSON to this file.
    #[arg(long, value_name = "FILE")]
    json: Option<PathBuf>,

    /// Energy scale γ (kJ/mol/Å²) — the spring constant and the energy-table scale.
    /// The default is B-factor-fitted for the spring; tune it for VoroMQA.
    #[arg(short = 'g', long, default_value_t = DEFAULT_GAMMA, value_name = "VALUE")]
    gamma: f64,

    /// Temperature, in kelvin. Sets the fluctuations and the Boltzmann weights.
    #[arg(short = 'T', long, default_value_t = 298.15, value_name = "KELVIN")]
    temperature: f64,

    /// Fit γ to the input's B-factors; overrides --gamma. Runs a dense all-atom solve;
    /// on failure it warns and falls back to --gamma.
    #[arg(long)]
    b_factor_fit: bool,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Animate the lowest modes into PDB/XTC trajectories.
    Animate {
        /// Specific mode to animate (1 = softest); repeatable. Overrides -n.
        #[arg(long = "mode", value_name = "INDEX")]
        mode: Vec<usize>,
        /// Peak displacement RMSD, in ångström.
        #[arg(short = 'a', long, default_value_t = 1.5, value_name = "RMSD")]
        amplitude: f64,
    },

    /// Morph the structure toward a target conformation (NOLB's transition).
    ///
    /// Projects the native→target motion onto the lowest -n modes and writes the morph
    /// (nonlinear unless --linear). The target must have the same atoms in the same order.
    #[command(arg_required_else_help = true)]
    Transition {
        /// Target conformation (PDB/mmCIF).
        #[arg(long, value_name = "PATH")]
        target: PathBuf,
        /// Re-diagonalize the network N times along the path (NOLB's --nIter), for large
        /// changes where the modes drift. 0 (default) is a single nonlinear morph; NOLB
        /// recommends 5. Not available with --linear or --voronota.
        #[arg(long = "n-iter", default_value_t = 0, value_name = "N")]
        n_iter: usize,
    },

    /// Per-frame Monte-Carlo energy table over a thermally-sampled pool.
    ///
    /// Columns: frame, mode, rmsd, energy (γ=1, Å²), energy_kJ_mol, weight — each mode
    /// swept over ±--sigmas of its thermal width.
    #[command(arg_required_else_help = true)]
    Energy {
        /// Output energy-table CSV path.
        #[arg(long, value_name = "PATH")]
        csv: PathBuf,
        /// Sample specific modes (1 = softest); repeatable. Overrides -n.
        #[arg(long = "mode", value_name = "INDEX")]
        mode: Vec<usize>,
        /// Thermal sampling width, in σ.
        #[arg(long, default_value_t = 3.0, value_name = "N")]
        sigmas: f64,
        /// Use the VoroMQA contact-area energy in place of the elastic spring energy
        /// (bundled v1 potential, re-tessellated per frame, scaled by --gamma).
        #[arg(long)]
        voromqa: bool,
        /// Like --voromqa, but with a potential file you supply. Mutually exclusive
        /// with --voromqa.
        #[arg(long, value_name = "FILE", conflicts_with = "voromqa")]
        voromqa_file: Option<PathBuf>,
    },
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
    let solve = &cli.solve;
    match &cli.command {
        Command::Animate { mode, amplitude } => run_animate(solve, mode, *amplitude),
        Command::Transition { target, n_iter } => run_transition(solve, target, *n_iter),
        Command::Energy {
            csv,
            mode,
            sigmas,
            voromqa,
            voromqa_file,
        } => run_energy(solve, csv, mode, *sigmas, *voromqa, voromqa_file.as_deref()),
    }
}

/// The parsed structure and built network inputs every verb needs, before the solve.
struct Prepared {
    records: Vec<AtomRecord>,
    positions: Vec<[f64; 3]>,
    atoms: Vec<Atom>,
    blocks: Vec<usize>,
    springs: Option<Vec<Spring>>,
    radii: RadiiLookup,
}

/// The HETATM/selection parse options for a run.
fn parse_options(solve: &SolveArgs) -> ParseOptions {
    ParseOptions {
        exclude_heteroatoms: !solve.hetatm,
        ..Default::default()
    }
}

/// Reject a non-finite or non-positive numeric option, with a uniform message.
fn require_positive(name: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{name} must be a positive number (got {value})"));
    }
    Ok(())
}

/// Validate the shared options, parse + `--select` the input, and build the network
/// inputs (positions, atoms, residue blocks, tessellation springs) — the prelude every
/// verb runs. The solve is left to each verb (the iterative transition skips it).
fn prepare(solve: &SolveArgs) -> Result<Prepared, String> {
    for (name, value) in [
        ("--cutoff", solve.cutoff),
        ("--gamma", solve.gamma),
        ("--temperature", solve.temperature),
    ] {
        require_positive(name, value)?;
    }

    let options = parse_options(solve);
    // One radii table, shared by the parser and the tessellation (--voronota).
    let radii = RadiiLookup::new();
    let parsed = parse_file_with_records(&solve.input, &options, &radii)
        .map_err(|e| format!("reading {}: {e}", solve.input.display()))?;
    let mut records = parsed.records;
    if let Some(expr) = &solve.select {
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
    let springs = solve
        .voronota
        .then(|| voronota_springs(&records, &radii))
        .transpose()?;

    Ok(Prepared {
        records,
        positions,
        atoms,
        blocks,
        springs,
        radii,
    })
}

/// Solve the mass-weighted rigid-block modes (lowest `k`) for the prepared network.
fn solve_modes(solve: &SolveArgs, prep: &Prepared, k: usize) -> Result<NormalModes, String> {
    with_connectivity(
        NormalModes::builder(&prep.atoms)
            .mass_weighted()
            .k_modes(k)
            .blocks(&prep.blocks),
        solve.cutoff,
        prep.springs.as_deref(),
    )
    .solve()
    .map_err(|e| format!("normal-mode analysis failed: {e}"))
}

/// The shared `animate`/`energy` prelude: solve the requested modes (rejecting any
/// beyond the spectrum), then print the spectrum report. Returns the modes, the
/// 1-indexed set the user asked for, and the effective γ.
fn solve_and_report(
    solve: &SolveArgs,
    prep: &Prepared,
    mode: &[usize],
) -> Result<(NormalModes, Vec<usize>, f64), String> {
    let wanted = wanted_modes(solve.modes, mode)?;
    let k = *wanted.iter().max().expect("wanted is non-empty");
    let modes = solve_modes(solve, prep, k)?;
    for &m in &wanted {
        if m > modes.len() {
            return Err(format!(
                "mode {m} requested but only {} non-zero modes exist",
                modes.len()
            ));
        }
    }
    let (gamma, fit_r) =
        effective_gamma(solve, &prep.atoms, &prep.records, prep.springs.as_deref())?;
    report(solve, &prep.records, &prep.blocks, &modes, gamma, fit_r)?;
    Ok((modes, wanted, gamma))
}

/// `animate`: the spectrum report plus either a PDB/XTC trajectory per requested
/// mode, or one `.nmd` file containing all requested modes for NMWiz.
fn run_animate(solve: &SolveArgs, mode: &[usize], amplitude: f64) -> Result<(), String> {
    let prep = prepare(solve)?;
    let (modes, wanted, _gamma) = solve_and_report(solve, &prep, mode)?;

    if let Some(path) = solve.output.as_deref().filter(|path| is_nmd(path)) {
        guard_input(path, &solve.input)?;
        io::write_nmd(
            path,
            &prep.records,
            &prep.positions,
            &modes,
            &wanted,
            amplitude,
        )?;
        println!("  wrote {}", path.display());
        return Ok(());
    }

    if solve.frames > 0 {
        let multi = wanted.len() > 1;
        for &m in &wanted {
            let path = output_path(solve.output.as_deref(), &solve.input, m, multi);
            guard_input(&path, &solve.input)?;
            let frames = animate(
                &modes,
                &prep.positions,
                m,
                amplitude,
                solve.frames,
                solve.linear,
            )?;
            write_trajectory(&path, &prep.records, &frames)?;
            println!("  wrote {}", path.display());
        }
    }
    Ok(())
}

/// `energy`: the spectrum report plus the merged MC trajectory and energy table.
fn run_energy(
    solve: &SolveArgs,
    csv: &Path,
    mode: &[usize],
    sigmas: f64,
    voromqa: bool,
    voromqa_file: Option<&Path>,
) -> Result<(), String> {
    // Cheap rejections before the (possibly two) eigensolves.
    if solve.frames == 0 {
        return Err("energy needs --frames greater than 0 (nothing to score otherwise)".into());
    }
    require_positive("--sigmas", sigmas)?;

    let prep = prepare(solve)?;
    // Build the scorer before the solve so a malformed --voromqa-file fails fast.
    let potential = match (voromqa, voromqa_file) {
        (true, _) => Some(voromqa::Potential::bundled()),
        (false, Some(path)) => Some(voromqa::Potential::load(path)?),
        (false, None) => None,
    };
    let (modes, wanted, gamma) = solve_and_report(solve, &prep, mode)?;
    write_merged(
        solve,
        &modes,
        &prep,
        &wanted,
        csv,
        gamma,
        sigmas,
        potential.as_ref(),
    )
}

/// `--energy`: merge the native frame plus every mode's frames into one
/// trajectory and write the matching per-frame energy table. The energies are
/// the elastic-network spring energy of each frame (native = 0), comparable
/// across modes because the energy depends only on the coordinates.
// A CLI orchestration sink; the inputs are all distinct and `radii` is shared
// rather than rebuilt, so a context struct would obscure more than it tidies.
#[allow(clippy::too_many_arguments)]
fn write_merged(
    solve: &SolveArgs,
    modes: &NormalModes,
    prep: &Prepared,
    wanted: &[usize],
    csv: &Path,
    gamma: f64,
    sigmas: f64,
    voromqa: Option<&voromqa::Potential>,
) -> Result<(), String> {
    let positions: &[[f64; 3]] = &prep.positions;
    let records: &[AtomRecord] = &prep.records;
    let radii = &prep.radii;
    // (`run_energy` has already rejected `--frames 0` before the solve.)
    // Resolve and check every output path before animating, so a clobbering
    // mistake fails fast and never destroys the input or one output with another.
    let traj = solve.output.as_deref().map_or_else(
        || with_stem(&solve.input, |stem| format!("{stem}_modes.pdb")),
        Path::to_path_buf,
    );
    guard_input(&traj, &solve.input)?;
    guard_input(csv, &solve.input)?;
    if same_path(csv, &traj) {
        return Err(format!(
            "the energy table and the trajectory cannot be the same file ({})",
            csv.display()
        ));
    }

    // Atom types are fixed across frames; compute them once for the scorer.
    let types = voromqa.map(|_| voromqa::atom_types(records));

    // The energy scheme: VoroMQA (when set) or the elastic spring energy. Both are
    // referenced to the native frame so the `energy` column starts at 0.
    let frame_energy = |frame: &[[f64; 3]]| match (voromqa, &types) {
        (Some(p), Some(t)) => p.score(frame, t, records, radii).energy,
        _ => modes.energy(frame),
    };
    // Native energy, scored once; for VoroMQA, surface any coverage gap here (the
    // same atoms every frame).
    let native_energy = match (voromqa, &types) {
        (Some(p), Some(t)) => {
            let s = p.score(positions, t, records, radii);
            if s.skipped > 0 {
                eprintln!(
                    "warning: VoroMQA has no parameters for {}/{} atoms (skipped from the score)",
                    s.skipped, s.total
                );
            }
            s.energy
        }
        _ => modes.energy(positions),
    };

    // A row: real energy γ·E and weight exp(−γ·E / RT), where E is the frame's
    // energy relative to the native (native ⇒ 0 ⇒ weight 1). γ is the shared scale
    // (a tuning knob with the right units) for whichever scheme is active; RT (not
    // kT) because E is molar.
    let rt = GAS_CONSTANT_KJ_PER_MOL_K * solve.temperature;
    let row = |frame, mode, rmsd, energy: f64| {
        let energy_kj = gamma * energy;
        io::EnergyRow {
            frame,
            mode,
            rmsd,
            energy,
            energy_kj,
            weight: (-energy_kj / rt).exp(),
        }
    };

    // Frame 0 is the native structure — the MC rest state, at energy 0.
    let mut frames = vec![positions.to_vec()];
    let mut rows = vec![row(0, 0, 0.0, 0.0)];
    for &m in wanted {
        for frame in thermal_frames(modes, positions, m, gamma, solve, sigmas)? {
            let energy = frame_energy(&frame) - native_energy;
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
    let scheme = if voromqa.is_some() {
        "VoroMQA"
    } else {
        "elastic"
    };
    println!(
        "  wrote {} and {} ({scheme} energy)",
        traj.display(),
        csv.display()
    );
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
const fn with_connectivity<'a>(
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
    solve: &SolveArgs,
    atoms: &[Atom],
    records: &[AtomRecord],
    springs: Option<&[Spring]>,
) -> Result<(f64, Option<f64>), String> {
    if solve.b_factor_fit {
        match fit_gamma(solve, atoms, records, springs) {
            Ok((gamma, r)) => Ok((gamma, Some(r))),
            // A failed fit shouldn't suppress the report and trajectory the user
            // also asked for; warn and fall back to the manual γ.
            Err(message) => {
                eprintln!(
                    "warning: {message}; falling back to --gamma {}",
                    solve.gamma
                );
                Ok((solve.gamma, None))
            }
        }
    } else {
        Ok((solve.gamma, None))
    }
}

/// Fit γ (kJ/mol/Å²) by scaling predicted ANM fluctuations to the input's
/// B-factors; returns `(γ, Pearson r)`. Uses a non-mass-weighted all-atom solve —
/// the correct, mass-independent configurational-fluctuation model — and the
/// through-origin least-squares `γ = Σ B₁² / Σ B₁·B^exp` (since `B ∝ 1/γ`).
fn fit_gamma(
    solve: &SolveArgs,
    atoms: &[Atom],
    records: &[AtomRecord],
    springs: Option<&[Spring]>,
) -> Result<(f64, f64), String> {
    let modes = with_connectivity(NormalModes::builder(atoms), solve.cutoff, springs)
        .solve()
        .map_err(|e| format!("--b-factor-fit: {e}"))?;

    // Predicted B at γ=1, paired with experimental B over the connected atoms
    // (non-zero prediction) that actually carry a B-factor.
    let (mut predicted, mut experimental) = (Vec::new(), Vec::new());
    for (b_pred, record) in modes
        .predicted_b_factors(solve.temperature)
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
fn wanted_modes(modes: usize, mode: &[usize]) -> Result<Vec<usize>, String> {
    let wanted = if mode.is_empty() {
        (1..=modes).collect::<Vec<_>>()
    } else {
        mode.to_vec()
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
    solve: &SolveArgs,
    sigmas: f64,
) -> Result<Vec<Vec<[f64; 3]>>, String> {
    let unit_energy = modes.energy(&displace_at(modes, positions, mode, 1.0, solve.linear)?);
    if !unit_energy.is_finite() || unit_energy <= 0.0 {
        return Err(format!(
            "mode {mode} has no restoring energy to sample thermally"
        ));
    }
    let rt = GAS_CONSTANT_KJ_PER_MOL_K * solve.temperature;
    let peak = sigmas * (rt / (2.0 * gamma * unit_energy)).sqrt();
    sweep(modes, positions, mode, peak, solve.frames, solve.linear)
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
    if is_nmd(path) {
        return Err(".nmd output is only supported by the animate command".into());
    }
    if has_extension(path, "xtc") {
        io::write_xtc(path, frames)
    } else {
        io::write_pdb(path, records, frames)
    }
}

fn is_nmd(path: &Path) -> bool {
    has_extension(path, "nmd")
}

fn has_extension(path: &Path, expected: &str) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(expected))
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

/// Parse a transition target: same atoms in the same order as the native (after any
/// `--select`), returning its coordinates. A count mismatch is an error — aligning
/// differing structures is not yet supported.
fn parse_target(
    solve: &SolveArgs,
    radii: &RadiiLookup,
    native_len: usize,
    target_path: &Path,
) -> Result<Vec<[f64; 3]>, String> {
    let options = parse_options(solve);
    let parsed = parse_file_with_records(target_path, &options, radii)
        .map_err(|e| format!("reading {}: {e}", target_path.display()))?;
    let mut target_records = parsed.records;
    if let Some(expr) = &solve.select {
        let selection =
            Selection::parse(expr).map_err(|e| format!("invalid selection {expr:?}: {e}"))?;
        target_records.retain(|r| selection.matches(r));
    }
    if target_records.len() != native_len {
        return Err(format!(
            "target {} has {} atoms after selection, but the structure has {}; the \
             transition needs the same atoms in the same order",
            target_path.display(),
            target_records.len(),
            native_len
        ));
    }
    Ok(target_records.iter().map(|r| [r.x, r.y, r.z]).collect())
}

/// The morph trajectory path (`--output`, else `<input>_morph.pdb`), guarded against
/// overwriting the input.
fn morph_output_path(solve: &SolveArgs) -> Result<PathBuf, String> {
    let path = match solve.output.as_deref() {
        Some(p) => p.to_path_buf(),
        None => with_stem(&solve.input, |stem| format!("{stem}_morph.pdb")),
    };
    guard_input(&path, &solve.input)?;
    Ok(path)
}

/// `transition`: project the native→target motion onto the lowest -n modes, report it,
/// and write the morph — or the iterative re-diagonalizing path when `n_iter > 0`.
fn run_transition(solve: &SolveArgs, target_path: &Path, n_iter: usize) -> Result<(), String> {
    let k = solve.modes;
    if k == 0 {
        return Err("no modes requested (use -n >= 1)".into());
    }
    // Reject the incompatible iterative-path flags up front, before any parsing or solve.
    if n_iter > 0 {
        if solve.linear {
            return Err("--n-iter is not compatible with --linear (the iterative \
                        transition is nonlinear)"
                .into());
        }
        if solve.voronota {
            return Err(
                "--n-iter is not supported with --voronota (re-tessellating the \
                        network along the path is not yet implemented); use the distance cutoff"
                    .into(),
            );
        }
    }

    let prep = prepare(solve)?;
    let target = parse_target(solve, &prep.radii, prep.positions.len(), target_path)?;

    if n_iter > 0 {
        return run_iterative_transition(solve, &prep, &target, target_path, k, n_iter);
    }

    let modes = solve_modes(solve, &prep, k)?;
    if k > modes.len() {
        return Err(format!(
            "{k} modes requested but only {} non-zero modes exist",
            modes.len()
        ));
    }
    let transition = modes
        .transition(&prep.positions, &target)
        .map_err(|e| format!("transition failed: {e}"))?;
    report_transition(solve, target_path, &transition)?;

    if solve.frames > 0 {
        let path = morph_output_path(solve)?;
        let frames = transition
            .morph(solve.frames, !solve.linear)
            .map_err(|e| format!("morph failed: {e}"))?;
        // The actual endpoint of the written trajectory — for a nonlinear morph this
        // differs from the linear-fit residual the report prints.
        let endpoint = frames
            .last()
            .map_or(0.0, |frame| transition.rmsd_to_target(frame));
        write_trajectory(&path, &prep.records, &frames)?;
        println!(
            "  wrote {} ({} frames, {}, endpoint RMSD {endpoint:.4} Å)",
            path.display(),
            frames.len(),
            if solve.linear { "linear" } else { "nonlinear" }
        );
    }
    Ok(())
}

/// The iterative re-diagonalizing transition (NOLB `--nlin --nIter`): re-solve the
/// network at each intermediate shape so the modes follow the deformation. Restricted
/// to the distance-cutoff network — a Voronoi network would need re-tessellating along
/// the path.
fn run_iterative_transition(
    solve: &SolveArgs,
    prep: &Prepared,
    target: &[[f64; 3]],
    target_path: &Path,
    k: usize,
    n_iter: usize,
) -> Result<(), String> {
    // (--voronota / --linear incompatibility is rejected up front in `run_transition`.)
    if k == 1 {
        // The whole point is a multi-mode morph; one mode barely moves the structure.
        eprintln!(
            "warning: --n-iter with a single mode (-n 1) barely closes the gap; \
             pass e.g. -n 10 for a fuller morph (NOLB uses several modes)"
        );
    }
    // The closure is the whole build config the re-solve needs: new positions, the
    // original masses, the same cutoff and rigid blocks, rebuilt every iteration.
    let masses: Vec<f64> = prep
        .records
        .iter()
        .map(|r| io::element_mass(&r.element))
        .collect();
    let cutoff = solve.cutoff;
    let blocks = &prep.blocks;
    let result = transition_iterative(
        &prep.positions,
        target,
        n_iter,
        solve.frames.max(1),
        |positions| {
            let atoms: Vec<Atom> = positions
                .iter()
                .zip(&masses)
                .map(|(&position, &mass)| Atom { position, mass })
                .collect();
            NormalModes::builder(&atoms)
                .cutoff(cutoff)
                .blocks(blocks)
                .mass_weighted()
                .k_modes(k)
                .solve()
        },
    )
    .map_err(|e| format!("iterative transition failed: {e}"))?;

    report_iterative(solve, n_iter, target_path, &result)?;

    if solve.frames > 0 {
        let path = morph_output_path(solve)?;
        write_trajectory(&path, &prep.records, result.frames())?;
        println!(
            "  wrote {} ({} frames, nonlinear, {} re-diagonalization(s), endpoint RMSD {:.4} Å)",
            path.display(),
            result.frames().len(),
            n_iter,
            result.final_rmsd()
        );
    }
    Ok(())
}

/// The iterative-transition report: the rigid-body-aligned RMSD to the target after
/// each re-diagonalization (NOLB's "RMSD statistics").
fn report_iterative(
    solve: &SolveArgs,
    n_iter: usize,
    target: &Path,
    result: &IterativeTransition,
) -> Result<(), String> {
    println!(
        "elasticrab — {} → {}",
        solve.input.display(),
        target.display()
    );
    println!("  iterative nonlinear transition, {n_iter} re-diagonalization(s)");
    println!("  diag  RMSD (Å)");
    for (i, &r) in result.step_rmsds().iter().enumerate() {
        println!("  {i:>4}  {r:>8.4}");
    }
    println!(
        "  initial RMSD {:.4} Å → final {:.4} Å",
        result.initial_rmsd(),
        result.final_rmsd()
    );

    if let Some(path) = &solve.json {
        let json = iterative_json(solve, n_iter, target, result);
        std::fs::write(path, json).map_err(|e| format!("writing {}: {e}", path.display()))?;
    }
    Ok(())
}

/// Open a transition JSON object with the `input`/`target` path fields the
/// single-shot and iterative reports share.
fn transition_json_header(input: &Path, target: &Path) -> String {
    let mut s = String::from("{\n");
    let _ = writeln!(s, "  \"input\": {},", json_string(&input.to_string_lossy()));
    let _ = writeln!(
        s,
        "  \"target\": {},",
        json_string(&target.to_string_lossy())
    );
    s
}

/// The iterative-transition report as a JSON object (numbers + the escaped paths).
fn iterative_json(
    solve: &SolveArgs,
    n_iter: usize,
    target: &Path,
    result: &IterativeTransition,
) -> String {
    let mut s = transition_json_header(&solve.input, target);
    let _ = writeln!(s, "  \"n_iter\": {n_iter},");
    let _ = writeln!(s, "  \"initial_rmsd\": {},", result.initial_rmsd());
    let _ = writeln!(s, "  \"final_rmsd\": {},", result.final_rmsd());
    let rmsds: Vec<String> = result.step_rmsds().iter().map(f64::to_string).collect();
    let _ = writeln!(s, "  \"step_rmsds\": [{}]", rmsds.join(", "));
    s.push_str("}\n");
    s
}

/// The transition report: per-mode overlap with the native→target motion, the
/// running cumulative overlap, and the RMSD remaining after each mode.
fn report_transition(solve: &SolveArgs, target: &Path, t: &Transition<'_>) -> Result<(), String> {
    let n = t.overlaps().len();
    println!(
        "elasticrab — {} → {}",
        solve.input.display(),
        target.display()
    );
    println!(
        "  {n} mass-weighted modes, initial RMSD {:.4} Å (Cartesian)",
        t.initial_rmsd()
    );
    // The residual column is the linear-fit RMSD after the lowest i+1 modes.
    println!("  mode  overlap  cumulative  residual_rmsd");
    for i in 0..n {
        println!(
            "  {:>4}  {:>7.3}  {:>10.3}  {:>13.4}",
            i + 1,
            t.overlaps()[i],
            t.cumulative_overlap()[i],
            t.residual_rmsd(i + 1)
        );
    }
    println!("  best linear fit {:.4} Å", t.residual_rmsd(n));

    if let Some(path) = &solve.json {
        let json = transition_json(solve, target, t);
        std::fs::write(path, json).map_err(|e| format!("writing {}: {e}", path.display()))?;
    }
    Ok(())
}

/// The transition report as a JSON object (hand-written, like [`report_json`]).
fn transition_json(solve: &SolveArgs, target: &Path, t: &Transition<'_>) -> String {
    let n = t.overlaps().len();
    let mut s = transition_json_header(&solve.input, target);
    let _ = writeln!(s, "  \"initial_rmsd\": {},", t.initial_rmsd());
    let _ = writeln!(s, "  \"final_rmsd\": {},", t.residual_rmsd(n));
    s.push_str("  \"modes\": [\n");
    for i in 0..n {
        let comma = if i + 1 < n { "," } else { "" };
        let _ = writeln!(
            s,
            "    {{\"index\": {}, \"overlap\": {}, \"cumulative_overlap\": {}, \"residual_rmsd\": {}}}{comma}",
            i + 1,
            t.overlaps()[i],
            t.cumulative_overlap()[i],
            t.residual_rmsd(i + 1)
        );
    }
    s.push_str("  ]\n}\n");
    s
}

fn report(
    solve: &SolveArgs,
    records: &[AtomRecord],
    blocks: &[usize],
    modes: &NormalModes,
    gamma: f64,
    fit_r: Option<f64>,
) -> Result<(), String> {
    let residues = blocks.iter().copied().max().map_or(0, |m| m + 1);
    let frequencies: Vec<f64> = modes.eigenvalues().iter().map(|&l| l.sqrt()).collect();

    println!("elasticrab — {}", solve.input.display());
    println!(
        "  atoms {}, residues {residues}, dropped {}",
        records.len(),
        modes.disconnected().len()
    );
    if solve.voronota {
        println!(
            "  Voronoi tessellation, {} springs, mass-weighted",
            modes.spring_count()
        );
    } else {
        println!(
            "  cutoff {} Å, {} springs, mass-weighted",
            solve.cutoff,
            modes.spring_count()
        );
    }
    match fit_r {
        Some(r) => println!("  gamma {gamma:.4} kJ/mol/Å² (fitted, B-factor r = {r:.3})"),
        None => println!("  gamma {gamma:.4} kJ/mol/Å²"),
    }
    // Collectivity κ (NOLB's `--collectivity`) is one cheap O(atoms) pass per mode and
    // is exactly what you need to pick modes, so it is always shown — no flag.
    println!("  mode  frequency  collectivity");
    for (j, frequency) in frequencies.iter().enumerate() {
        println!(
            "  {:>4}  {frequency:.6}  {:>12.4}",
            j + 1,
            modes.collectivity(j)
        );
    }

    if let Some(path) = &solve.json {
        let json = report_json(
            solve,
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
    solve: &SolveArgs,
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
        json_string(&solve.input.to_string_lossy())
    );
    let _ = writeln!(s, "  \"atoms\": {atoms},");
    let _ = writeln!(s, "  \"residues\": {residues},");
    let _ = writeln!(s, "  \"dropped\": [{}],", dropped.join(", "));
    // `network` is always present as the discriminator; `cutoff` only when it applies.
    if solve.voronota {
        let _ = writeln!(s, "  \"network\": \"voronota\",");
    } else {
        let _ = writeln!(s, "  \"network\": \"cutoff\",");
        let _ = writeln!(s, "  \"cutoff\": {},", solve.cutoff);
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
            "    {{\"index\": {}, \"frequency\": {frequency}, \"eigenvalue\": {eigenvalue}, \"collectivity\": {}}}{comma}",
            j + 1,
            modes.collectivity(j)
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

    /// The bundled VoroMQA potential covers every atom of a standard protein (the
    /// `generalize` step maps symmetric/terminal atoms to trained types) and yields
    /// a finite energy.
    #[test]
    fn voromqa_scores_crambin_fully() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/crambin_heavy.pdb");
        let options = ParseOptions {
            exclude_heteroatoms: true,
            ..Default::default()
        };
        let radii = RadiiLookup::new();
        let records = parse_file_with_records(Path::new(path), &options, &radii)
            .unwrap()
            .records;
        let positions: Vec<[f64; 3]> = records.iter().map(|r| [r.x, r.y, r.z]).collect();

        let types = voromqa::atom_types(&records);
        let score = voromqa::Potential::bundled().score(&positions, &types, &records, &radii);
        assert_eq!(score.skipped, 0, "v1 should cover every crambin atom");
        assert_eq!(score.total, records.len());
        assert!(score.energy.is_finite() && score.energy != 0.0);
    }
}
