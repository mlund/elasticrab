//! File output for the CLI: multi-model PDB, XTC (molly), NMD/NMWiz, and CSV.
//! voronota-ltr parses structures but does not write them, so the writers live
//! here; the atomic-mass table is a small lookup voronota does not provide.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};

use molly::{Frame, XTCWriter};
use voronota_ltr::input::AtomRecord;

use elasticrab::NormalModes;

/// A trajectory: one entry per frame, each a per-atom coordinate set (ångström).
pub type Trajectory = [Vec<[f64; 3]>];

/// One row of the per-frame energy table written beside a merged trajectory.
pub struct EnergyRow {
    pub frame: usize,
    /// 1-based mode index; `0` marks the native (input) frame.
    pub mode: usize,
    pub rmsd: f64,
    /// The active scheme's energy (elastic spring, or VoroMQA with `--voromqa`)
    /// *relative to the native structure*, at γ=1, in Å². Native frame = 0.
    pub energy: f64,
    /// Real energy `γ·energy`, in kJ/mol.
    pub energy_kj: f64,
    /// Boltzmann weight `exp(−energy_kj / RT)` (native = 1).
    pub weight: f64,
}

/// Standard atomic weights for the common protein elements; a neutral fallback
/// (carbon) covers anything else, since mass-weighting only rescales the spectrum.
pub fn element_mass(element: &str) -> f64 {
    match element {
        "N" => 14.007,
        "O" => 15.999,
        "S" => 32.06,
        "P" => 30.974,
        _ => 12.011,
    }
}

/// Format an output error with the path that failed.
fn writing(path: &Path, e: &std::io::Error) -> String {
    format!("writing {}: {e}", path.display())
}

/// Write the energy table as CSV (`frame,mode,rmsd,energy,energy_kJ_mol,weight`),
/// one row per frame in the merged trajectory's order, streamed row by row.
pub fn write_csv(path: &Path, rows: &[EnergyRow]) -> Result<(), String> {
    let file = File::create(path).map_err(|e| writing(path, &e))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(b"frame,mode,rmsd,energy,energy_kJ_mol,weight\n")
        .map_err(|e| writing(path, &e))?;
    for r in rows {
        writeln!(
            writer,
            "{},{},{:.6},{:.6},{:.6},{:.6e}",
            r.frame, r.mode, r.rmsd, r.energy, r.energy_kj, r.weight
        )
        .map_err(|e| writing(path, &e))?;
    }
    writer.flush().map_err(|e| writing(path, &e))
}

/// Write a multi-model PDB trajectory: `MODEL`/`ENDMDL` per frame, re-using the
/// records' metadata with each frame's coordinates.
pub fn write_pdb(path: &Path, records: &[AtomRecord], frames: &Trajectory) -> Result<(), String> {
    let file = File::create(path).map_err(|e| writing(path, &e))?;
    let mut writer = BufWriter::new(file);
    // Build and flush one model at a time, so a long trajectory never holds the
    // whole file in memory (the XTC writer streams the same way).
    let mut model = String::new();
    for (index, coords) in frames.iter().enumerate() {
        model.clear();
        let _ = writeln!(model, "MODEL     {:>4}", index + 1);
        for (record, &p) in records.iter().zip(coords) {
            atom_line(&mut model, record, p);
        }
        model.push_str("ENDMDL\n");
        writer
            .write_all(model.as_bytes())
            .map_err(|e| writing(path, &e))?;
    }
    writer.flush().map_err(|e| writing(path, &e))
}

/// Write normal modes in NMD format for VMD's NMWiz plugin. Unlike PDB/XTC this is
/// not a trajectory: it stores the native coordinates once and then the requested
/// mode vectors. `mode_scale_rmsd` is the RMSD reached when NMWiz applies the
/// normalized mode at scale 1.
pub fn write_nmd(
    path: &Path,
    records: &[AtomRecord],
    positions: &[[f64; 3]],
    modes: &NormalModes,
    wanted: &[usize],
    mode_scale_rmsd: f64,
) -> Result<(), String> {
    let file = File::create(path).map_err(|e| writing(path, &e))?;
    let mut writer = BufWriter::new(file);
    let mut line = String::new();

    let nmd_path = absolute_path(path).to_string_lossy().into_owned();
    writeln!(writer, "nmwiz_load {}", tcl_word(&nmd_path)).map_err(|e| writing(path, &e))?;
    writeln!(writer, "name {}", nmd_name(path)).map_err(|e| writing(path, &e))?;
    write_tokens(
        &mut writer,
        path,
        "atomnames",
        records.iter().map(|r| nmd_token(&r.name)),
    )?;
    write_tokens(
        &mut writer,
        path,
        "resnames",
        records.iter().map(|r| nmd_token(&r.res_name)),
    )?;
    write_tokens(
        &mut writer,
        path,
        "resids",
        records.iter().map(|r| r.res_seq.to_string()),
    )?;
    write_tokens(
        &mut writer,
        path,
        "chainids",
        records.iter().map(|r| nmd_token(&r.chain_id)),
    )?;
    write_numbers(
        &mut writer,
        path,
        "bfactors",
        records.iter().map(|r| r.b_factor),
        "{:.2}",
    )?;
    write_numbers(
        &mut writer,
        path,
        "coordinates",
        positions.iter().flat_map(|p| [p[0], p[1], p[2]]),
        "{:.3}",
    )?;

    let scale = mode_scale_rmsd * (records.len() as f64).sqrt();
    for &mode in wanted {
        let i = mode - 1;
        let displacement = modes.mode_displacement(i);
        let norm = displacement_norm(&displacement);
        if norm <= 0.0 || !norm.is_finite() {
            continue;
        }
        line.clear();
        let _ = write!(line, "mode {mode} {:.6}", scale);
        for p in displacement {
            for x in p {
                let _ = write!(line, " {:.6}", x / norm);
            }
        }
        line.push('\n');
        writer
            .write_all(line.as_bytes())
            .map_err(|e| writing(path, &e))?;
    }
    writer.flush().map_err(|e| writing(path, &e))
}

fn absolute_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    }
}

/// Quote a value as one Tcl word. NMD files double as Tcl scripts for VMD, so
/// paths need escaping for spaces, brackets, `$`, and Windows backslashes.
fn tcl_word(value: &str) -> String {
    if value.is_empty() {
        return "{}".to_string();
    }

    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '\\' | ' ' | '"' | '$' | '[' | ']' | '{' | '}' | ';' => {
                out.push('\\');
                out.push(c);
            }
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn nmd_name(path: &Path) -> String {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("elasticrab");
    nmd_token(name)
}

fn nmd_token(value: &str) -> String {
    let token: String = value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if token.is_empty() {
        "_".to_string()
    } else {
        token
    }
}

fn write_tokens(
    writer: &mut BufWriter<File>,
    path: &Path,
    label: &str,
    values: impl IntoIterator<Item = String>,
) -> Result<(), String> {
    write!(writer, "{label}").map_err(|e| writing(path, &e))?;
    for value in values {
        write!(writer, " {value}").map_err(|e| writing(path, &e))?;
    }
    writeln!(writer).map_err(|e| writing(path, &e))
}

fn write_numbers(
    writer: &mut BufWriter<File>,
    path: &Path,
    label: &str,
    values: impl IntoIterator<Item = f64>,
    format: &str,
) -> Result<(), String> {
    write!(writer, "{label}").map_err(|e| writing(path, &e))?;
    for value in values {
        match format {
            "{:.2}" => write!(writer, " {value:.2}"),
            "{:.3}" => write!(writer, " {value:.3}"),
            _ => write!(writer, " {value:.6}"),
        }
        .map_err(|e| writing(path, &e))?;
    }
    writeln!(writer).map_err(|e| writing(path, &e))
}

fn displacement_norm(displacement: &[[f64; 3]]) -> f64 {
    displacement
        .iter()
        .flat_map(|p| p.iter())
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
}

/// Append one fixed-column `ATOM` line at new coordinates. The serial and residue
/// number are wrapped to their field width so a huge structure cannot overflow a
/// column and shift every later field.
fn atom_line(out: &mut String, r: &AtomRecord, p: [f64; 3]) {
    let chain = r.chain_id.chars().next().unwrap_or(' ');
    let icode = r.i_code.chars().next().unwrap_or(' ');
    let _ = writeln!(
        out,
        "ATOM  {:>5} {} {:>3} {}{:>4}{}   {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}",
        r.serial.rem_euclid(100_000),
        pdb_atom_name(&r.name, &r.element),
        r.res_name,
        chain,
        r.res_seq.rem_euclid(10_000),
        icode,
        p[0],
        p[1],
        p[2],
        r.element,
    );
}

/// Place an atom name in its four-column field. By PDB convention a name whose
/// element symbol is one letter is indented a column (` CA `, not `CA  `), so a
/// parser that infers the element from the column reads it correctly.
fn pdb_atom_name(name: &str, element: &str) -> String {
    if element.len() == 1 && name.len() < 4 {
        format!(" {name:<3}")
    } else {
        format!("{name:<4}")
    }
}

/// Write an XTC trajectory. XTC stores coordinates in nanometres, so the
/// ångström inputs are scaled by 1/10; a single molecule has no periodic box, so
/// a cube enclosing the whole trajectory (plus a margin) stands in.
pub fn write_xtc(path: &Path, frames: &Trajectory) -> Result<(), String> {
    let mut writer = XTCWriter::create(path).map_err(|e| writing(path, &e))?;
    let boxvec = enclosing_box_nm(frames);
    for (step, coords) in frames.iter().enumerate() {
        let positions = coords
            .iter()
            .flat_map(|p| [p[0] as f32 / 10.0, p[1] as f32 / 10.0, p[2] as f32 / 10.0])
            .collect();
        let frame = Frame {
            step: step as u32,
            time: step as f32,
            boxvec,
            precision: 1000.0,
            positions,
        };
        writer.write_frame(&frame).map_err(|e| writing(path, &e))?;
    }
    Ok(())
}

/// A diagonal (cubic-ish) box, in nm, enclosing every frame with a 1 nm margin.
fn enclosing_box_nm(frames: &Trajectory) -> [f32; 9] {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for coords in frames {
        for p in coords {
            for c in 0..3 {
                lo[c] = lo[c].min(p[c]);
                hi[c] = hi[c].max(p[c]);
            }
        }
    }
    let side = |c: usize| ((hi[c] - lo[c]) / 10.0 + 2.0) as f32; // nm, +1 nm each side
    [side(0), 0.0, 0.0, 0.0, side(1), 0.0, 0.0, 0.0, side(2)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tcl_word_escapes_loader_paths() {
        assert_eq!(tcl_word("/tmp/modes file.nmd"), "/tmp/modes\\ file.nmd");
        assert_eq!(
            tcl_word("C:\\Users\\Mikael Lund\\modes[1].nmd"),
            "C:\\\\Users\\\\Mikael\\ Lund\\\\modes\\[1\\].nmd"
        );
        assert_eq!(tcl_word("cost$1;safe"), "cost\\$1\\;safe");
    }
}
