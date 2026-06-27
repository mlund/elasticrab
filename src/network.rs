//! Pairwise contacts of the elastic network.
//!
//! A contact is created between every pair of atoms closer than the cutoff;
//! these are the springs whose geometry defines the Hessian.

use std::collections::HashMap;

/// A spring between two atoms, carrying the data the Hessian assembly needs so
/// it never has to revisit the coordinates: the endpoint indices, the
/// displacement `j - i`, its squared length, and a relative stiffness `weight`
/// (the effective spring constant is `gamma · weight`; the cutoff path uses 1.0).
pub(crate) struct Contact {
    pub i: usize,
    pub j: usize,
    pub delta: [f64; 3],
    pub dist2: f64,
    pub weight: f64,
}

/// All atom pairs within `cutoff` of each other.
///
/// A uniform cell list: atoms are binned into a grid of cutoff-sized cells, so
/// each atom only has to be compared against the atoms in its own cell and the
/// 26 around it — `O(n)` for the roughly uniform densities of molecular
/// structures, where the brute-force `O(n²)` scan dominates large solvated
/// systems. The result is exactly the brute-force contact set; only the order
/// differs, which no consumer depends on (they all accumulate).
pub(crate) fn contacts(positions: &[[f64; 3]], cutoff: f64) -> Vec<Contact> {
    // A non-positive or NaN cutoff makes the grid spacing meaningless; fall back
    // to the exact pairwise scan, which handles it correctly.
    if cutoff <= 0.0 || cutoff.is_nan() {
        return brute_force(positions, cutoff);
    }

    let grid = Grid::new(positions, cutoff);
    let cutoff2 = cutoff * cutoff;
    let mut out = Vec::new();

    for i in 0..positions.len() {
        grid.for_each_candidate(positions[i], |j| {
            // Each unordered pair is emitted once, by its lower-indexed atom.
            if j > i {
                out.extend(contact_within(positions, i, j, cutoff2));
            }
        });
    }
    out
}

/// Indices of atoms that no contact touches. Such an atom has no spring, so its
/// three coordinates are unconstrained and would add three spurious zero modes;
/// the caller drops them before solving (matching Pepsi's `cAtomGrid`, which
/// flags exactly the zero-neighbour atoms).
pub(crate) fn disconnected_atoms(n_atoms: usize, contacts: &[Contact]) -> Vec<usize> {
    let mut connected = vec![false; n_atoms];
    for c in contacts {
        connected[c.i] = true;
        connected[c.j] = true;
    }
    (0..n_atoms).filter(|&a| !connected[a]).collect()
}

/// `b - a`, the displacement stored on a [`Contact`].
fn displacement(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
}

/// The contact between atoms `i` and `j`, if they lie within the cutoff. Shared
/// by the cell list and the brute-force fallback so the distance test lives once.
fn contact_within(positions: &[[f64; 3]], i: usize, j: usize, cutoff2: f64) -> Option<Contact> {
    let delta = displacement(positions[i], positions[j]);
    let dist2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    (dist2 <= cutoff2).then_some(Contact {
        i,
        j,
        delta,
        dist2,
        weight: 1.0,
    })
}

/// Contacts from an explicit edge list: one spring per [`Spring`](crate::Spring),
/// its geometry taken from the positions and its stiffness from the edge's
/// `weight`. Rejects out-of-range or self-referential edges.
pub(crate) fn contacts_from_edges(
    positions: &[[f64; 3]],
    springs: &[crate::Spring],
) -> Result<Vec<Contact>, crate::Error> {
    let n = positions.len();
    springs
        .iter()
        .map(|s| {
            if s.i >= n || s.j >= n || s.i == s.j {
                return Err(crate::Error::InvalidSpring);
            }
            let delta = displacement(positions[s.i], positions[s.j]);
            let dist2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
            Ok(Contact {
                i: s.i,
                j: s.j,
                delta,
                dist2,
                weight: s.weight,
            })
        })
        .collect()
}

/// A uniform grid of cutoff-sized cells holding atom indices, stored sparsely so
/// an elongated or mostly-empty bounding box costs nothing for the empty space.
struct Grid {
    cells: HashMap<[i64; 3], Vec<usize>>,
    origin: [f64; 3],
    cutoff: f64,
}

impl Grid {
    fn new(positions: &[[f64; 3]], cutoff: f64) -> Self {
        // Anchor the grid at the minimum corner so cell coordinates stay small.
        let mut origin = [f64::INFINITY; 3];
        for p in positions {
            for axis in 0..3 {
                origin[axis] = origin[axis].min(p[axis]);
            }
        }
        let mut grid = Self {
            cells: HashMap::new(),
            origin,
            cutoff,
        };
        for (atom, &p) in positions.iter().enumerate() {
            grid.cells.entry(grid.cell_of(p)).or_default().push(atom);
        }
        grid
    }

    fn cell_of(&self, p: [f64; 3]) -> [i64; 3] {
        std::array::from_fn(|axis| ((p[axis] - self.origin[axis]) / self.cutoff).floor() as i64)
    }

    /// Call `visit` with every atom in the cell of `p` and its 26 neighbours —
    /// a superset of the atoms within `cutoff`, which the caller filters by
    /// actual distance.
    fn for_each_candidate(&self, p: [f64; 3], mut visit: impl FnMut(usize)) {
        let [cx, cy, cz] = self.cell_of(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(atoms) = self.cells.get(&[cx + dx, cy + dy, cz + dz]) {
                        atoms.iter().copied().for_each(&mut visit);
                    }
                }
            }
        }
    }
}

/// Exact `O(n²)` pairwise scan: the fallback for a degenerate cutoff and the
/// oracle the cell list is tested against.
fn brute_force(positions: &[[f64; 3]], cutoff: f64) -> Vec<Contact> {
    let cutoff2 = cutoff * cutoff;
    let mut out = Vec::new();
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            out.extend(contact_within(positions, i, j, cutoff2));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_only_pairs_within_cutoff() {
        // Three colinear atoms at x = 0, 1, 3. With cutoff 1.5 only (0,1) is a
        // contact; (1,2) is 2.0 apart and (0,2) is 3.0 apart.
        let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let c = contacts(&pos, 1.5);
        assert_eq!(c.len(), 1);
        assert_eq!((c[0].i, c[0].j), (0, 1));
        assert_eq!(c[0].dist2, 1.0);
    }

    #[test]
    fn cutoff_is_inclusive() {
        let pos = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        assert_eq!(contacts(&pos, 2.0).len(), 1);
    }

    /// The cell list must return exactly the brute-force contact set. A
    /// deterministic pseudo-random cloud spanning several cells exercises the
    /// 27-cell stencil, including pairs that straddle cell boundaries.
    #[test]
    fn cell_list_matches_brute_force() {
        let positions: Vec<[f64; 3]> = (0..400)
            .map(|i| {
                let f = i as f64;
                [
                    (f * 12.9898).sin() * 20.0,
                    (f * 78.233).sin() * 20.0,
                    (f * 37.719).sin() * 20.0,
                ]
            })
            .collect();

        for &cutoff in &[1.0, 3.5, 8.0] {
            let mut fast: Vec<(usize, usize)> = contacts(&positions, cutoff)
                .iter()
                .map(|c| (c.i, c.j))
                .collect();
            let mut slow: Vec<(usize, usize)> = brute_force(&positions, cutoff)
                .iter()
                .map(|c| (c.i, c.j))
                .collect();
            fast.sort_unstable();
            slow.sort_unstable();
            assert_eq!(fast, slow, "cutoff {cutoff}");
        }
    }

    #[test]
    fn isolated_atom_is_disconnected() {
        // Two atoms bonded, a third far away with no neighbour within cutoff.
        let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [50.0, 0.0, 0.0]];
        let c = contacts(&pos, 1.5);
        assert_eq!(disconnected_atoms(3, &c), vec![2]);
    }

    #[test]
    fn fully_connected_has_no_disconnected_atoms() {
        let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let c = contacts(&pos, 1.5);
        assert!(disconnected_atoms(3, &c).is_empty());
    }
}
