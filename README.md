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

A harmonic spring joins every pair of atoms within `cutoff`; diagonalizing the
resulting `3N×3N` Hessian gives the normal modes — the collective, low-energy
motions a structure most readily makes. This is the standard ANM (uniform spring
constant), the same model that ProDy and Pepsi-SAXS use.

- **Public surface is four items**: `Atom`, `Params`, `NormalModes`, `Error`.
  Neighbour search, Hessian assembly, optional mass-weighting, and the
  eigensolver are all internal.
- **Defaults match the conventional ANM** (15 Å cutoff, γ = 1, unit mass) and
  reproduce ProDy's reference 1UBI Hessian and eigenvalues.
- **Mass-weighting is opt-in** (`Params::mass_weighted`); eigenvalues are then
  squared frequencies `ω²`.
- **Rigid blocks are opt-in** (`NormalModes::with_blocks`): treat groups of atoms
  as rigid bodies to shrink the eigenproblem (the Rotation-Translation Blocks
  method of Pepsi-SAXS/NOLB), matching ProDy's reference.
- **Partial solving is opt-in** (`sparse` feature + `Params::k_modes`): for large
  systems (e.g. a solvated protein) too big to diagonalize, compute only the
  lowest *k* non-zero modes instead of the full spectrum — for both the plain and
  the rigid-block model.

## Scope

Deliberately stops at "frequencies and modes". Structure parsing, hydration
shells, residue coarse-graining, and fitting amplitudes to data belong to the
caller. The default eigensolver is **dense** (cost ∝ atom-count³), ideal for
small and medium systems. For large systems, the optional `sparse` feature adds
a partial solver for the lowest *k* modes — and also swaps in a SIMD dense
eigensolver (~3× faster) for the full solve.

## Testing

Every result is validated against independent references; `cargo test` runs:

- **Property tests** — Hessian symmetry, the rigid-body null space, rigid-block
  degree-of-freedom accounting, and the error paths.
- **Analytic checks** — closed-form results such as the diatomic reduced-mass
  relation `ω² = γ(1/m₁ + 1/m₂)` and the equal-mass scaling law.
- **ProDy golden tests** — the spectrum matches ProDy's published reference
  exactly, for both the plain ANM (1UBI) and the rigid-block reduction (2GB1).
- **NOLB golden test** — mass-weighted rigid blocks match NOLB, the engine
  Pepsi-SAXS wraps (crambin). The reference values are vendored, so the test
  reproduces without the binary.

See [`docs/PEPSI_COMPARISON.md`](docs/PEPSI_COMPARISON.md) for how the crate
relates to Pepsi-SAXS / NOLB.

## Benchmarks

`cargo bench --features sparse` compares the dense and sparse solvers, with and
without rigid blocks, on real protein structures (lowest 10 modes). Indicative
numbers (one machine; the relative speedups are the point):

| structure | dense | dense + blocks | sparse | sparse + blocks |
|---|---|---|---|---|
| medium (812 atoms) | 1.8 s | 1.0 s | 60 ms | 53 ms |
| large (8015 atoms) | — (too large) | — | 1.5 s | 0.82 s |

On the medium structure the sparse solver runs **~30× faster** than dense, and
**~17× faster** than dense with rigid blocks. On the large one — too big for a
dense solve to fit in memory — the sparse solver finishes in about a second. (The
dense times use the SIMD eigensolver the `sparse` feature enables; the default
build's scalar dense solve is ~3× slower again.)

## License

Apache-2.0. Bundled test fixtures are from ProDy (MIT); see
`tests/data/ATTRIBUTION.md`.

---

Contributors: the architecture and its rationale are in
[`docs/DESIGN.md`](docs/DESIGN.md).
