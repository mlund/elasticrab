<p align="center">
  <img src="https://github.com/user-attachments/assets/87ed30fa-b33d-4bb6-859b-17e04c9708cc" alt="Elasticrab logo" width="320">
</p>

A minimal Rust library for **Anisotropic Network Model (ANM) normal-mode
analysis**: give it atoms, get back the vibrational modes of an elastic network.

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
constant), the model ProDy and Pepsi-SAXS use. The whole public surface is four
items — `Atom`, `Params`, `NormalModes`, `Error` — with defaults (15 Å cutoff,
γ = 1, unit mass) that reproduce ProDy's reference 1UBI spectrum.

Everything beyond the plain dense solve is opt-in:

- **Mass-weighting** (`Params::mass_weighted`): eigenvalues become squared
  frequencies `ω²`.
- **Rigid blocks** (`NormalModes::with_blocks`): treat groups of atoms as rigid
  bodies to shrink the eigenproblem (the Rotation-Translation Blocks method of
  Pepsi-SAXS / NOLB).
- **Partial solving** (`sparse` feature + `Params::k_modes`): for systems too
  large to diagonalize, compute only the lowest *k* non-zero modes — for both the
  plain and the rigid-block model. The feature also swaps in a SIMD dense
  eigensolver (~3× faster) for the full solve.
- **Multi-threading** (`parallel` feature): faster on large dense solves, at the
  cost of bit-for-bit reproducibility.
- **Disconnected atoms** are dropped: an atom with no neighbour within the cutoff
  (say a stray water in a hydration shell) carries no spring, so it is removed
  before solving and listed by `NormalModes::disconnected()` — as Pepsi-SAXS and
  NOLB do.

The crate stops at frequencies and modes: structure parsing, hydration shells,
coarse-graining, and fitting amplitudes to data belong to the caller. The
neighbour search is a cell list, linear in the atom count.

## Validation

`cargo test` reproduces independent references. The spectrum matches ProDy's
published values exactly, for both the plain ANM (1UBI) and the rigid-block
reduction (2GB1). The mass-weighted rigid-block path matches **NOLB** — the
engine Pepsi-SAXS wraps — to about six digits on crambin, including the
disconnected-atom drop: adding an isolated atom leaves the spectrum unchanged,
exactly as NOLB reports it. Property and analytic checks cover Hessian symmetry,
the rigid-body null space, the diatomic reduced-mass relation
`ω² = γ(1/m₁ + 1/m₂)`, and the error paths.

## Visualizing a mode

`NormalModes::displace(positions, i, amplitude)` returns the structure pushed
along mode `i` — sweep `amplitude` to make a trajectory you can watch. The
`animate_pdb` example turns a mode into a multi-model PDB for PyMOL or VMD:

```sh
cargo run --example animate_pdb -- protein.pdb > mode6.pdb   # args: [amplitude] [mode] [frames]
```

## Benchmarks

`cargo bench` compares the solvers on real protein structures — medium (812
atoms) and large (8015 atoms), lowest 10 modes. Indicative numbers (one machine;
relative speedups are the point); the 1-core columns use `--features sparse`, the
10-core columns `--features parallel`:

| solver | medium · 1 core | medium · 10 cores | large · 1 core | large · 10 cores |
|---|---|---|---|---|
| dense | 1.8 s | 0.69 s | — (too large) | — |
| dense + rigid blocks | 1.0 s | 0.90 s | — | — |
| sparse (lowest *k*) | 60 ms | 67 ms | 1.5 s | 1.33 s |
| sparse + rigid blocks | 53 ms | 49 ms | 0.82 s | 0.72 s |

The sparse solvers run ~30× faster than the full dense solve and handle the large
structure dense cannot fit in memory. Multi-threading helps the dense solve but
not the iterative ones, so keep `RAYON_NUM_THREADS` low (1–2) for partial solving.
These figures use the 15 Å cutoff conventional for Cα models; at the ~5 Å cutoff
of all-atom models (as in Pepsi-SAXS) the network is far sparser and the large
partial solve drops to ~0.1 s, where the linear cell list earns its keep.

## License

Apache-2.0. Bundled test fixtures are from ProDy (MIT); see
`tests/data/ATTRIBUTION.md`.
