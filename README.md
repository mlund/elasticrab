<p align="center">
  <img src="https://github.com/user-attachments/assets/87ed30fa-b33d-4bb6-859b-17e04c9708cc" alt="Elasticrab logo" width="320">
</p>
<p align="center">
  <a href="https://github.com/mlund/elasticrab/actions/workflows/ci.yml">
    <img src="https://github.com/mlund/elasticrab/actions/workflows/ci.yml/badge.svg">
  </a>
</p>

A minimal Rust library for **Anisotropic Network Model (ANM) normal-mode
analysis**: give it atoms, get back the vibrational modes of an elastic network.

```rust
use elasticrab::{Atom, NormalModes};

let atoms = vec![
    Atom { position: [0.0, 0.0, 0.0], mass: 12.0 },
    Atom { position: [3.8, 0.0, 0.0], mass: 12.0 },
    Atom { position: [3.8, 3.8, 0.0], mass: 12.0 },
];

let modes = NormalModes::builder(&atoms).cutoff(15.0).solve()?;
let eigenvalues = modes.eigenvalues();   // ascending; first ~6 ≈ 0 (rigid body)
let first_mode  = modes.eigenvector(6);  // per-atom displacement field
let amplitudes  = modes.thermal_amplitudes(300.0);
# Ok::<(), elasticrab::Error>(())
```

## Features

- **ANM normal modes** — dense all-atom solve; optional mass-weighting; defaults match ProDy.
- **Rigid blocks (RTB)** — the Rotation-Translation Blocks reduction of Pepsi-SAXS / NOLB.
- **Partial solver** (`.k_modes()`) — return just the lowest *k* modes; `sparse` makes it scale to large systems (and adds a SIMD dense solver), `parallel` multi-threads.
- **Cell-list neighbour search** — linear in atom count; disconnected atoms are dropped, as Pepsi-SAXS / NOLB do.
- **Mode visualization** — linear and NOLB nonlinear (bond-preserving) displacement.
- **Conformational energy** — `NormalModes::energy()` scores any structure with the network's spring energy, for Boltzmann reweighting of sampled conformations.
- **Command-line tool** (`cli` feature) — the `elasticrab` binary animates modes into PDB/XTC trajectories, with PDB/mmCIF input, VMD-like atom selection, a JSON report, and a per-frame energy table for Monte-Carlo reweighting.
- **Tests** (`cargo test`) — property, analytic, and golden tests: exact ProDy spectra (1UBI, 2GB1) and ~6-digit NOLB agreement (crambin), including the disconnected-atom drop.
- **Fixtures** — vendored reference data (ProDy Hessians and eigenvalues, NOLB frequencies), so tests need no external binary.

## What it does

A harmonic spring joins every pair of atoms within `cutoff`; diagonalizing the
resulting `3N×3N` Hessian gives the normal modes — the collective, low-energy
motions a structure most readily makes. This is the standard ANM (uniform spring
constant), the model ProDy and Pepsi-SAXS use. The public surface is small —
`Atom`, `NormalModes` (configured through `NormalModes::builder`), `Spring`,
`Error` — and the conventional defaults (γ = 1, unit mass) reproduce ProDy's
reference 1UBI spectrum.

Everything beyond the plain dense solve is opt-in, set on the builder:

- **Mass-weighting** (`.mass_weighted()`): eigenvalues become squared
  frequencies `ω²`.
- **Rigid blocks** (`.blocks()`): treat groups of atoms as rigid
  bodies to shrink the eigenproblem (the Rotation-Translation Blocks method of
  Pepsi-SAXS / NOLB).
- **Custom springs** (`.springs()`): supply an explicit list of weighted springs
  (per-edge stiffness `γ·weight`) instead of a distance cutoff — e.g. a
  contact-area-weighted network.
- **Partial solving** (`.k_modes()`): return only the lowest *k* non-zero
  modes — for both the plain and the rigid-block model. The `sparse` feature
  computes them without ever forming the dense Hessian (what scales to large
  systems) and adds a SIMD dense eigensolver (~3× faster) for the full solve;
  without it, `k_modes` falls back to a dense solve.
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
along mode `i` — sweep `amplitude` to make a trajectory you can watch.
`displace_nonlinear` instead moves each rigid block as a rigid body (NOLB's
nonlinear extrapolation), keeping bonds rigid at large amplitude. The
[command-line tool](#command-line-tool) wraps this into ready-made PDB or XTC
trajectories.

## Command-line tool

The `elasticrab` binary runs the analysis and writes mode-animation trajectories
for PyMOL or VMD. Install it from the repository:

```sh
cargo install --git https://github.com/mlund/elasticrab --features cli
```

Then animate the softest modes of a structure (PDB or mmCIF):

```sh
elasticrab protein.pdb -o mode1.pdb                       # softest mode, bond-preserving
elasticrab protein.pdb -n 5 -o anim.xtc                   # five lowest modes -> anim_mode1.xtc …
elasticrab protein.pdb --select "chain A" --json out.json # restrict atoms; structured report
elasticrab protein.pdb -n 5 -o pool.xtc --energy e.csv    # merge modes + per-frame energies
elasticrab protein.pdb --b-factor-fit --frames 0          # fit gamma to the input's B-factors
elasticrab protein.pdb --voronota -o mode1.pdb            # area-weighted Voronoi springs, not a cutoff
elasticrab protein.pdb -n 5 -o p.xtc --energy e.csv --voromqa  # VoroMQA energy, not spring
```

It prints a frequency report to stdout (`--json` writes it to a file); run
`elasticrab --help` for cutoff, amplitude, frame-count, and selection options.
The interface is similar to NOLB.

`--energy` builds a Monte-Carlo conformation pool: it merges every requested mode
into one trajectory (native frame first) and writes a
`frame,mode,rmsd,energy,energy_kJ_mol,weight` table. Each mode is swept over
±`--sigmas` σ (default 3) of its own **thermal amplitude** — sized from γ and the
temperature — so the frames are Boltzmann-relevant, not the much larger
visualization `--amplitude`. Every row's energy is referenced to the native frame
and scaled by `--gamma` ($\gamma$):

$$E = E_\text{frame} - E_\text{native}, \qquad E_\text{kJ/mol} = \gamma\,E, \qquad w = \exp\!\left(-\frac{\gamma\,E}{RT}\right),$$

with $R$ the molar gas constant and $T$ = `--temperature` (the native row is then
$`E=0`$, $`w=1`$). By default $`E_\text{frame}`$ is the spring energy; `--voromqa`
substitutes the VoroMQA score (both area-based) — defined under
[Voronoi tessellation](#voronoi-tessellation). `frame` is the 0-based trajectory
index (a multi-model PDB labels it `MODEL frame+1`).

`--b-factor-fit` calibrates γ physically: it matches the ANM's predicted thermal
fluctuations to the input's crystallographic B-factors and reports the fitted γ
(with the correlation as a quality check), overriding `--gamma` (and falling back
to it if the fit fails). The default γ is
a B-factor-fitted median over a small PDB set (`scripts/calibrate-gamma.sh`);
since the fit is noisy across structures, pass `--b-factor-fit` for quantitative
work.

## Voronoi tessellation

A Laguerre (radical) tessellation of the atoms — each inflated by a 1.4 Å solvent
probe — gives one cell per atom. Two atoms *contact* when their cells share a face,
and the face area $`A_{ij}`$ measures how much they pack together. The contact
network is parameter-free (no cutoff) and occlusion-aware: an atom between two others
screens their contact, so $`A_{ij} = 0`$. `elasticrab` builds it once, in process
([voronota-ltr](https://github.com/mlund/voronota-ltr)), and uses it two ways — for
springs (`--voronota`) and for scoring (`--voromqa`).

**`--voronota` — area-weighted springs.** The default cutoff network springs every
pair within `--cutoff` with equal stiffness. `--voronota` instead springs each
contacting pair, with a per-edge weight set by its contact area:

$$w_{ij} = \frac{A_{ij}}{\bar A}, \qquad \bar A = \frac{1}{N_c} \sum_{(i,j)} A_{ij}$$

($`w_{ij} = 1`$ for the cutoff network; $`N_c`$ is the number of contacts). A larger
shared face is a stiffer spring, and normalizing to unit mean keeps the average
weight at 1, so `--gamma` keeps its meaning. The harmonic energy of a conformation is

$$E_\text{spring} = \tfrac12\,\gamma \sum_{(i,j)} w_{ij}\,\bigl(\lvert \mathbf{r}_{ij} \rvert - d^0_{ij}\bigr)^2,$$

with $`\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i`$ and $`d^0_{ij}`$ the native
separation (so the native structure has $`E_\text{spring} = 0`$). The two networks
feed the same Hessian and energy; only the pair set and the weights $`w_{ij}`$
differ, so their absolute frequencies are not directly comparable. `--voronota` is
mutually exclusive with `--cutoff`.

**`--voromqa` — knowledge-based scoring.** `--voromqa` scores each `--energy` frame
with the [VoroMQA](https://github.com/kliment-olechnovic/voronota) contact-area
potential (bundled v1), Boltzmann-inverted from the PDB, in place of the spring
energy:

$$E_\text{VoroMQA} = \sum_{(i,j)} A_{ij}\,e(t_i, t_j, c_{ij}) \;+\; \sum_a S_a\,e(t_a, \text{solvent}).$$

Here $`t_a`$ is atom $a$'s type, $`e(\cdot)`$ a tabulated energy per unit area, $`S_a`$
the solvent-accessible area of atom $a$ (its one-body burial term), and $`c_{ij}`$ the
contact class — centrality (a buried, face-on contact vs. a peripheral one) crossed
with sequence separation (adjacent residues vs. farther): `central_sep1`,
`central_sep2`, `sep1`, or `sep2`. The harmonic NMA still generates the frames; only
the per-frame score changes, re-tessellated each frame. Because $`E_\text{VoroMQA}`$
is area-based (Å²) like the spring energy, `--gamma` (kJ/mol/Å²) scales it to
`energy_kJ_mol` and `weight` exactly as above — a tuning knob, since γ's
B-factor-fitted default suits the springs. `--voromqa-file <path>` supplies a
different potential; `--voromqa` and `--voromqa-file` are mutually exclusive and both
require `--energy`.

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
