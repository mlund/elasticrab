//! VoroMQA contact-area potential — an empirical, knowledge-based energy for
//! Monte-Carlo reweighting of sampled conformations.
//!
//! VoroMQA (Olechnovic & Venclovas) is Boltzmann-inverted from a large PDB set.
//! The energy of a structure is the contact area times a tabulated weight for each
//! atom-type pair and their contact class, plus a one-body solvent (burial) term:
//!
//! ```text
//! E = Σ_ij  A_ij · e(type_i, type_j, class_ij)  +  Σ_a  SAS_a · e(type_a, solvent)
//! ```
//!
//! where `class` is (centrality)×(sequence separation). The contact areas, the
//! centrality flag, and the per-atom solvent-accessible area all come from a single
//! Voronoi tessellation (voronota-ltr), so this is fully in-process.
//!
//! The `e(...)` weights are dimensionless log-odds, so the energy is in arbitrary
//! units (area, Å², times a statistical weight), **not** kJ/mol. It is meaningful
//! only as *differences* between conformations, and for MC reweighting carries one
//! free temperature scale (the undetermined scale of any Boltzmann-inverted score).
//!
//! The bundled potential is **v1** (`voromqa_v1_potential.txt`, embedded verbatim)
//! — the original published VoroMQA atom-level potential. Its contact classes are
//! centrality-only (`central_sep1/2`, `sep1/2`), which the lightweight voronota-lt
//! tessellation supports; v3/v5 additionally need a "peripherial" tag it does not
//! compute, and v4 is a coarse backbone+CB typing. Other revisions can be supplied
//! with `--voromqa-file`. The data is from Voronota
//! (github.com/kliment-olechnovic/voronota, MIT-licensed); see
//! `tests/data/ATTRIBUTION.md`.

use std::collections::HashMap;
use std::path::Path;

use voronota_ltr::input::{AtomRecord, RadiiLookup};
use voronota_ltr::{compute_tessellation, Ball, Results};

use super::VORONOTA_PROBE;

/// The v1 potential, embedded verbatim (byte-identical to upstream).
const BUNDLED_V1: &str = include_str!("voromqa_v1_potential.txt");

/// Index into a pair's `[f64; 4]` for a contact's class. Order matches the parser:
/// `[central_sep1, central_sep2, sep1, sep2]`.
const fn class_index(central: bool, adjacent: bool) -> usize {
    match (central, adjacent) {
        (true, true) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (false, false) => 3,
    }
}

/// A parsed VoroMQA contact-area potential.
pub struct Potential {
    /// Per atom-type-pair contact energies, keyed by the canonical `(min, max)`
    /// type-string pair; classes are positional (see [`class_index`]), a missing
    /// class defaulting to `0.0` (no contribution).
    pairs: HashMap<(String, String), [f64; 4]>,
    /// One-body solvent (burial) energy per atom type — also the registry of
    /// "known" types (every type carries a solvent row).
    solvent: HashMap<String, f64>,
}

/// What scoring a structure produced.
pub struct Score {
    /// The VoroMQA pseudo-energy.
    pub energy: f64,
    /// Atoms whose type is absent from the potential (skipped from both terms).
    pub skipped: usize,
    /// Total atoms scored.
    pub total: usize,
}

impl Potential {
    /// The v1 potential bundled with elasticrab.
    pub fn bundled() -> Self {
        Self::parse(BUNDLED_V1).expect("bundled v1 potential parses")
    }

    /// Load a potential from a file (`--voromqa-file`).
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        Self::parse(&text)
    }

    /// Parse the text potential. Rejects classes outside the centrality-only set,
    /// so a v3/v5 "peripherial" file — which the lightweight tessellation cannot
    /// reproduce — fails loudly instead of scoring wrongly.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut pairs: HashMap<(String, String), [f64; 4]> = HashMap::new();
        let mut solvent = HashMap::new();
        for (n, line) in text.lines().enumerate() {
            let mut it = line.split_whitespace();
            let (t1, t2, class, value) = match (it.next(), it.next(), it.next(), it.next()) {
                (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
                _ => continue, // blank or malformed line
            };
            let energy: f64 = value
                .parse()
                .map_err(|_| format!("line {}: invalid energy {value:?}", n + 1))?;
            if t2 == "c<solvent>" {
                solvent.insert(t1.to_string(), energy);
            } else {
                let idx = match class {
                    "central_sep1" => 0,
                    "central_sep2" => 1,
                    "sep1" => 2,
                    "sep2" => 3,
                    other => {
                        return Err(format!(
                            "line {}: unsupported contact class {other:?}; only the \
                             centrality-only potentials (v1/v2/v4) are supported, not the \
                             peripherial classes of v3/v5",
                            n + 1
                        ))
                    }
                };
                pairs.entry(canonical(t1, t2)).or_insert([0.0; 4])[idx] = energy;
            }
        }
        if pairs.is_empty() || solvent.is_empty() {
            return Err("no VoroMQA entries found (not a potential file?)".into());
        }
        Ok(Self { pairs, solvent })
    }

    /// VoroMQA energy of a conformation. `positions` are the (possibly displaced)
    /// coordinates; `records` supply atom types and residue/chain identity (fixed
    /// across frames); `radii` is the tessellation's per-atom radius lookup.
    pub fn score(
        &self,
        positions: &[[f64; 3]],
        types: &[String],
        records: &[AtomRecord],
        radii: &RadiiLookup,
    ) -> Score {
        let balls: Vec<Ball> = positions
            .iter()
            .zip(records)
            .map(|(p, r)| Ball::new(p[0], p[1], p[2], radii.get_radius(&r.res_name, &r.name)))
            .collect();
        let result = compute_tessellation(&balls, VORONOTA_PROBE, None, None, false);

        let mut energy = 0.0;
        // Two-body contact term.
        for c in &result.contacts {
            let (ta, tb) = (&types[c.id_a], &types[c.id_b]);
            if !self.solvent.contains_key(ta) || !self.solvent.contains_key(tb) {
                continue;
            }
            let Some(adjacent) = sequence_adjacent(&records[c.id_a], &records[c.id_b]) else {
                continue; // same residue — intra-residue contacts are not scored
            };
            let e = self
                .pairs
                .get(&canonical(ta, tb))
                .map_or(0.0, |arr| arr[class_index(c.central, adjacent)]);
            energy += c.area * e;
        }
        // One-body solvent (burial) term, from per-atom solvent-accessible area. An
        // atom with no Voronoi cell is uncontacted, hence fully exposed: fall back
        // to the full inflated sphere so the term never silently drops between
        // frames (which would inject a discontinuity into the score).
        let sas = result.sas_areas();
        let mut skipped = 0;
        for (i, (ty, r)) in types.iter().zip(records).enumerate() {
            if let Some(e) = self.solvent.get(ty) {
                let area = match sas.get(i) {
                    Some(Some(a)) => *a,
                    _ => {
                        let radius = radii.get_radius(&r.res_name, &r.name) + VORONOTA_PROBE;
                        4.0 * std::f64::consts::PI * radius * radius
                    }
                };
                energy += area * e;
            } else {
                skipped += 1;
            }
        }
        Score {
            energy,
            skipped,
            total: records.len(),
        }
    }
}

/// The `R<RES>A<ATOM>` type descriptors for a set of records — computed once per
/// trajectory, since types are fixed while only coordinates vary. Names go through
/// VoroMQA's normalization (so symmetric atoms hit the single trained type).
pub fn atom_types(records: &[AtomRecord]) -> Vec<String> {
    records
        .iter()
        .map(|r| {
            let (res, atom) = generalize(&r.res_name, &r.name);
            format!("R<{res}>A<{atom}>")
        })
        .collect()
}

/// Normalize a residue/atom name to VoroMQA's trained typing — the heavy-atom
/// subset of Voronota's `generalize_crad` (`contacts_scoring_utilities.h`):
/// selenomethionine/-cysteine to the standard residue, the terminal carboxylate
/// oxygen to `O`, and each symmetry-equivalent atom (the carboxylate oxygens, the
/// guanidinium nitrogens, the symmetric aromatic-ring carbons) to one name. The
/// many hydrogen rules there are irrelevant to a heavy-atom model.
fn generalize<'a>(res_name: &'a str, atom_name: &'a str) -> (&'a str, &'a str) {
    let (mut res, mut atom) = (res_name, atom_name);
    match res {
        "MSE" => {
            res = "MET";
            if atom == "SE" {
                atom = "SD";
            }
        }
        "SEC" => {
            res = "CYS";
            if atom == "SE" {
                atom = "SG";
            }
        }
        _ => {}
    }
    if atom == "OXT" {
        atom = "O";
    }
    atom = match (res, atom) {
        ("ARG", "NH2") => "NH1",
        ("ASP", "OD2") => "OD1",
        ("GLU", "OE2") => "OE1",
        ("PHE" | "TYR", "CD2") => "CD1",
        ("PHE" | "TYR", "CE2") => "CE1",
        _ => atom,
    };
    (res, atom)
}

/// Canonical `(min, max)` ordering of a type pair, so lookups are order-independent.
fn canonical(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

/// Whether two atoms' residues are sequence-*adjacent* (separation 1 in the same
/// chain): `Some(true)` ⇒ `sep1`, `Some(false)` ⇒ `sep2` (separation ≥ 2, or a
/// different chain), `None` ⇒ same residue (separation 0, not scored).
fn sequence_adjacent(a: &AtomRecord, b: &AtomRecord) -> Option<bool> {
    if a.chain_id != b.chain_id {
        return Some(false);
    }
    match (a.res_seq - b.res_seq).abs() {
        0 => None,
        1 => Some(true),
        _ => Some(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(chain: &str, seq: i32) -> AtomRecord {
        AtomRecord {
            record_name: "ATOM".into(),
            serial: 0,
            name: "CA".into(),
            alt_loc: String::new(),
            res_name: "ALA".into(),
            chain_id: chain.into(),
            res_seq: seq,
            i_code: String::new(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            element: "C".into(),
            b_factor: 0.0,
        }
    }

    #[test]
    fn generalize_normalizes_symmetric_and_modified_atoms() {
        assert_eq!(generalize("ASP", "OD2"), ("ASP", "OD1"));
        assert_eq!(generalize("GLU", "OE2"), ("GLU", "OE1"));
        assert_eq!(generalize("ARG", "NH2"), ("ARG", "NH1"));
        assert_eq!(generalize("PHE", "CD2"), ("PHE", "CD1"));
        assert_eq!(generalize("TYR", "CE2"), ("TYR", "CE1"));
        assert_eq!(generalize("ALA", "OXT"), ("ALA", "O"));
        assert_eq!(generalize("MSE", "SE"), ("MET", "SD"));
        // LEU's δ-carbons are not symmetry-equivalent, so they stay distinct.
        assert_eq!(generalize("LEU", "CD2"), ("LEU", "CD2"));
        assert_eq!(generalize("ALA", "CB"), ("ALA", "CB"));
    }

    #[test]
    fn class_index_covers_the_four_classes() {
        assert_eq!(class_index(true, true), 0); // central_sep1
        assert_eq!(class_index(true, false), 1); // central_sep2
        assert_eq!(class_index(false, true), 2); // sep1
        assert_eq!(class_index(false, false), 3); // sep2
    }

    #[test]
    fn sequence_separation_classifies() {
        assert_eq!(sequence_adjacent(&rec("A", 1), &rec("A", 2)), Some(true)); // sep1
        assert_eq!(sequence_adjacent(&rec("A", 1), &rec("A", 5)), Some(false)); // sep2
        assert_eq!(sequence_adjacent(&rec("A", 3), &rec("A", 3)), None); // same residue
        assert_eq!(sequence_adjacent(&rec("A", 1), &rec("B", 1)), Some(false)); // other chain
    }

    #[test]
    fn parse_reads_pairs_and_solvent_with_positional_classes() {
        let text = "\
R<ALA>A<C> R<ALA>A<CA> central_sep1 1.0
R<ALA>A<C> R<ALA>A<CA> sep2 2.0
R<ALA>A<C> c<solvent> . 3.0
R<ALA>A<CA> c<solvent> . 4.0";
        let p = Potential::parse(text).unwrap();
        // Positional classes; absent ones default to 0; lookup is order-independent.
        let key = canonical("R<ALA>A<CA>", "R<ALA>A<C>");
        assert_eq!(*p.pairs.get(&key).unwrap(), [1.0, 0.0, 0.0, 2.0]);
        assert_eq!(p.solvent["R<ALA>A<C>"], 3.0);
    }

    #[test]
    fn parse_rejects_peripherial_potentials() {
        let text = "R<ALA>A<C> R<ALA>A<CA> peripherial_sep1 1.0\nR<ALA>A<C> c<solvent> . 3.0";
        assert!(Potential::parse(text).is_err());
    }

    #[test]
    fn bundled_v1_parses_all_types() {
        let p = Potential::bundled();
        assert_eq!(p.solvent.len(), 160, "v1 has 160 atom types");
        assert!(!p.pairs.is_empty());
    }
}
