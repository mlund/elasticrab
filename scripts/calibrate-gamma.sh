#!/usr/bin/env bash
#
# Calibrate the ANM spring constant gamma against experimental B-factors.
#
# Fits gamma on a small set of high-resolution monomeric structures (downloaded
# from RCSB) and reports the spread, to choose a sensible default for the CLI's
# `--gamma`. The median is robust to the occasional poor fit.
#
#   cargo build --release --features cli
#   scripts/calibrate-gamma.sh
#
set -euo pipefail

IDS=(1UBQ 1AKI 3LZT 2LZM 1PGB 1BPI 1CTF 1IGD 4PTI 1A6M)
BIN="${ELASTICRAB:-./target/release/elasticrab}"

[ -x "$BIN" ] || {
    echo "build first: cargo build --release --features cli" >&2
    exit 1
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

printf '%-6s %14s %8s\n' id gamma r
for id in "${IDS[@]}"; do
    pdb="$TMP/$id.pdb"
    if ! curl -sf "https://files.rcsb.org/download/$id.pdb" -o "$pdb"; then
        echo "$id: download failed" >&2
        continue
    fi
    json="$TMP/$id.json"
    if "$BIN" "$pdb" --b-factor-fit --frames 0 --json "$json" >/dev/null 2>&1; then
        python3 - "$json" "$id" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f'{sys.argv[2]:<6} {d["gamma"]:14.4f} {d.get("b_factor_correlation", float("nan")):8.3f}')
PY
    else
        echo "$id: fit failed" >&2
    fi
done | tee "$TMP/results.txt"

# A poorly-correlated fit (B-factors near-orthogonal to the prediction) makes the
# through-origin gamma blow up, so keep only the well-correlated structures and
# take the median, which is robust to the remaining spread.
python3 - "$TMP/results.txt" <<'PY'
import sys, statistics
R_MIN = 0.3
rows = [l.split() for l in open(sys.argv[1]) if l.strip()]
good = [float(g) for _, g, r in rows if float(r) >= R_MIN]
print(f"\nkept {len(good)}/{len(rows)} fits with r >= {R_MIN}")
if good:
    print(f"median={statistics.median(good):.2f}  "
          f"mean={statistics.mean(good):.2f}  stdev={statistics.pstdev(good):.2f}")
    print(f"Set DEFAULT_GAMMA (kJ/mol/A^2) = {statistics.median(good):.1f}")
PY
