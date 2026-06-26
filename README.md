<p align="center">
  <img src="https://github.com/user-attachments/assets/87ed30fa-b33d-4bb6-859b-17e04c9708cc" alt="Elasticrab logo" width="320">
</p>

A minimal Rust library for **Anisotropic Network Model (ANM)
normal-mode analysis**: give it atoms, get back the vibrational modes of an
elastic network.

```rust
use elasticrab::{Atom, Params, NormalModes};

let atoms = vec![
    Atom { position: [0.0, 0.0, 0.0], mass: 12.0 },
    Atom { position: [3.8, 0.0, 0.0], mass: 12.0 },
    Atom { position: [3.8, 3.8, 0.0], mass: 12.0 },
];

let modes = NormalModes::new(&atoms, &Params::default())?;
let eigenvalues = modes.eigenvalues();   // ascending; first ~6 ≈ 0 (rigid body)
let first_mode  = modes.eigenvector(6);  // per-atom displacement field
let amplitudes  = modes.thermal_amplitudes(300.0);
# Ok::<(), elasticrab::Error>(())
```

## What it does

Every pair of atoms within `cutoff` is joined by a harmonic spring; the `3N×3N`
Hessian of that network is diagonalized into normal modes — the collective,
low-energy motions a structure most readily makes. This is the standard ANM
(uniform spring constant), the same super-element Hessian used by ProDy and
Pepsi-SAXS.

- **Public surface is four items**: `Atom`, `Params`, `NormalModes`, `Error`.
  Neighbour search, Hessian assembly, optional mass-weighting, and the
  eigensolver are all internal.
- **Defaults match the conventional ANM** (15 Å cutoff, γ = 1, unit mass) and
  are validated against ProDy's reference 1UBI Hessian and eigenvalues.
- **Mass-weighting is opt-in** (`Params::mass_weighted`); eigenvalues are then
  squared frequencies `ω²`.
- **Rigid blocks are opt-in** (`NormalModes::with_blocks`): the Rotation-Translation
  Blocks reduction used by Pepsi-SAXS/NOLB, validated against ProDy's reference.

## Scope

Deliberately stops at "frequencies and modes". Structure parsing, hydration
shells, residue coarse-graining, and fitting amplitudes to data belong to the
caller. The eigensolver is **dense** (cost ∝ atom-count³) — ideal for small and
medium systems; a sparse partial solver could replace it later without changing
the public API.

## Testing

`cargo test` runs:

- **Unit & property tests** — Hessian symmetry, rigid-body null space, RTB DOF
  accounting, error paths.
- **Analytic mass-weighting checks** — the diatomic reduced-mass relation
  `ω² = γ(1/m₁ + 1/m₂)` and the equal-mass scaling invariant.
- **ProDy golden tests** — exact spectrum match for the plain ANM (1UBI) and the
  RTB reduction (2GB1).
- **NOLB golden test** — mass-weighted RTB against the authentic engine, via
  vendored crambin fixtures (the binary is not needed at test time).

See [`docs/PEPSI_COMPARISON.md`](docs/PEPSI_COMPARISON.md) for how the crate
relates to Pepsi-SAXS / NOLB.

## License

Apache-2.0. Bundled test fixtures are from ProDy (MIT); see
`tests/data/ATTRIBUTION.md`.
