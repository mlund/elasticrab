//! File output for the CLI: a multi-model PDB writer and an XTC writer (molly).
//! voronota-ltr parses structures but does not write them, so the writers live
//! here; the atomic-mass table is a small lookup voronota does not provide.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::Path;

use molly::{Frame, XTCWriter};
use voronota_ltr::input::AtomRecord;

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
