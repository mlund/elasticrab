//! Pairwise contacts of the elastic network.
//!
//! A contact is created between every pair of atoms closer than the cutoff;
//! these are the springs whose geometry defines the Hessian.

/// A spring between two atoms, carrying the data the Hessian assembly needs so
/// it never has to revisit the coordinates: the endpoint indices, the
/// displacement `j - i`, and its squared length.
pub(crate) struct Contact {
    pub i: usize,
    pub j: usize,
    pub delta: [f64; 3],
    pub dist2: f64,
}

/// All atom pairs within `cutoff` of each other.
///
/// Brute-force O(n²): every distinct pair is tested once (`i < j`). This is the
/// deliberate simple choice — it is exact and trivially correct for the system
/// sizes this crate targets. It lives behind this function precisely so a cell
/// list can replace it later without touching anything downstream.
pub(crate) fn contacts(positions: &[[f64; 3]], cutoff: f64) -> Vec<Contact> {
    let cutoff2 = cutoff * cutoff;
    let mut out = Vec::new();

    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let delta = [
                positions[j][0] - positions[i][0],
                positions[j][1] - positions[i][1],
                positions[j][2] - positions[i][2],
            ];
            let dist2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
            if dist2 <= cutoff2 {
                out.push(Contact { i, j, delta, dist2 });
            }
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
}
