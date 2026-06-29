<p align="center">
  <img src="https://github.com/user-attachments/assets/87ed30fa-b33d-4bb6-859b-17e04c9708cc" alt="Elasticrab logo" width="320">
</p>
<p align="center">
  <a href="https://github.com/mlund/elasticrab/actions/workflows/ci.yml">
    <img src="https://github.com/mlund/elasticrab/actions/workflows/ci.yml/badge.svg">
  </a>
</p>

# Elasticrab

Elasticrab is a command-line tool for protein normal-mode analysis. It reads a
PDB or mmCIF structure, builds an elastic network, computes the lowest
mass-weighted rigid-block modes, and writes trajectories, transition morphs, or
energy tables for downstream analysis.

The project also exposes a small Rust library. The CLI is the main user-facing
tool; the library is for developers who want to embed the same ANM solver in
their own programs.

_Pronunciation:_ ih-LAS-tee-krab (/ɪˈlæs.ti.kræb/)

## Features

- **Protein-focused CLI**: read PDB or mmCIF, select atoms with VMD-like
  expressions, report mode frequencies and collectivity, and write PDB or XTC
  trajectories.
- **Three workflows**: `animate` visualizes modes, `transition` morphs one
  structure toward another, and `energy` builds a thermally sampled trajectory
  with a per-frame weight table.
- **Rigid-block normal modes**: the CLI groups atoms by residue, uses
  mass-weighted Rotation-Translation Blocks (RTB), and uses NOLB-style nonlinear
  displacement by default to preserve bonds within each block.
- **Two network models**: use a conventional distance cutoff, or use Voronoi
  tessellation to make contact-area-weighted springs without choosing a cutoff.
- **Voronoi-based energies**: the `energy` workflow can score frames with the
  elastic spring energy or with the bundled VoroMQA v1 contact-area potential.
- **Reproducible outputs**: text reports go to stdout, JSON reports are optional,
  and energy tables are written as CSV.
- **Tests as a feature**: the test suite checks the ANM Hessian and spectra
  against [ProDy](https://github.com/prody/ProDy), RTB spectra against ProDy,
  mass-weighted RTB frequencies and
  collectivities against [NOLB](https://team.inria.fr/nano-d/software/nolb-normal-modes/)
  references, transition behavior, CLI grammar, and disconnected-atom handling.

## Installation

Elasticrab is written in Rust. [Install Rust](https://rust-lang.org/tools/install/)
first, then install the CLI from the repository:

```sh
cargo install --git https://github.com/mlund/elasticrab --features cli
elasticrab --version
elasticrab --help
```

For a local checkout:

```sh
git clone https://github.com/mlund/elasticrab
cd elasticrab
cargo install --path . --features cli
```

The `cli` feature includes the partial sparse solver. For multithreaded solver
kernels, build with `--features "cli parallel"` instead.

For development and validation:

```sh
cargo test
cargo test --features cli
cargo test --features sparse
cargo bench --features sparse
```

## Quick Start

Print a report for the softest mode without writing a trajectory:

```sh
elasticrab -i protein.pdb -s 0 animate
```

Animate the five lowest modes. When several modes are requested, Elasticrab
inserts `_mode1`, `_mode2`, ... before the output extension:

```sh
elasticrab -i protein.pdb -n 5 -o modes.xtc animate
```

Use a Voronoi tessellation instead of a distance cutoff:

```sh
elasticrab -i protein.pdb --voronota -s 0 animate
```

Restrict the analysis to a selection:

```sh
elasticrab -i protein.pdb --select "chain A and name CA" -n 3 animate
```

Morph a native structure toward a target conformation:

```sh
elasticrab -i native.pdb -n 10 -o morph.pdb transition --target target.pdb
```

Build a thermally sampled trajectory and an energy table:

```sh
elasticrab -i protein.pdb -n 5 -s 20 -o pool.pdb energy --csv energy.csv
```

Use Voronoi springs for the modes and VoroMQA for the frame energies:

```sh
elasticrab -i protein.pdb --voronota -n 5 -s 20 -o pool.pdb \
  energy --csv energy.csv --voromqa
```

Shared options such as `--input`, `--modes`, `--frames`, `--select`,
`--voronota`, and `--output` come before the command. Command-specific options,
such as `--target`, `--mode`, `--csv`, and `--voromqa`, come after the command.

Run these for the complete option list:

```sh
elasticrab --help
elasticrab animate --help
elasticrab transition --help
elasticrab energy --help
```

## Input Model

Elasticrab reads PDB and mmCIF files. The CLI uses atomic coordinates and element
names, groups atoms into one rigid block per residue, and always drops waters. HETATM
records are excluded unless `--hetatm` is set.

Atom selections use VMD-like expressions, for example:

```sh
elasticrab -i protein.pdb --select "chain A and name CA" animate
```

The `transition` command requires the native and target files to contain the
same atoms in the same order after any selection. Elasticrab rigid-body aligns
the target to the native structure before projecting the internal deformation
onto the modes.

Disconnected atoms are dropped from the solve. A disconnected atom has no spring
within the chosen network, so it contributes no normal mode or spring energy.
The report lists how many atoms were dropped.

## Command Reference

### `animate`

`animate` writes one trajectory per requested mode. By default it uses the
softest mode, writes 20 frames, and uses nonlinear rigid-block displacement.

```sh
elasticrab -i protein.pdb -o mode1.pdb animate
elasticrab -i protein.pdb -n 5 -o modes.xtc animate
elasticrab -i protein.pdb animate --mode 3 --amplitude 2.0
```

Important options:

- `-n, --modes N`: use modes 1 through `N`.
- `--mode INDEX`: animate a specific 1-based mode. Repeat it to request several
  modes.
- `-s, --frames N`: number of frames. Use `0` for report only.
- `-a, --amplitude RMSD`: peak displacement RMSD in Å.
- `--linear`: use linear displacement instead of the nonlinear bond-preserving
  displacement.

### `transition`

`transition` projects a native-to-target conformational change onto the lowest
modes and writes a morph trajectory.

```sh
elasticrab -i native.pdb -n 10 -o morph.pdb transition --target target.pdb
```

For large changes, re-diagonalize along the path:

```sh
elasticrab -i native.pdb -n 10 -o morph.pdb transition \
  --target target.pdb --n-iter 5
```

The report gives the initial RMSD, each mode's overlap with the target motion,
the cumulative overlap, and the residual RMSD after each mode. `--n-iter` is
nonlinear and currently uses the cutoff network; it is not available with
`--linear` or `--voronota`.

### `energy`

`energy` writes one merged trajectory and one CSV file. Frame 0 is the native
structure. The following frames sample each requested mode over plus or minus
`--sigmas` of its thermal width.

```sh
elasticrab -i protein.pdb -n 5 -s 20 -o pool.pdb energy --csv energy.csv
```

The CSV columns are:

| column | meaning |
|---|---|
| `frame` | 0-based trajectory frame |
| `mode` | sampled mode; `0` for the native frame |
| `rmsd` | RMSD from the native structure |
| `energy` | native-referenced energy before applying `--gamma` |
| `energy_kJ_mol` | `--gamma` times `energy` |
| `weight` | Boltzmann weight relative to the native frame |

Elasticrab uses

$$
\Delta E = E_\text{frame} - E_\text{native}
$$

$$
E_\text{kJ/mol} = \gamma \Delta E
$$

$$
w = \exp\left(-\frac{\gamma \Delta E}{RT}\right)
$$

where $R$ is the molar gas constant and $T$ is `--temperature`.

By default, `energy` uses the elastic spring energy. With `--voromqa`, it
re-tessellates every frame and uses the bundled VoroMQA v1 contact-area
potential instead:

```sh
elasticrab -i protein.pdb --voronota -n 5 -s 20 -o pool.pdb \
  energy --csv energy.csv --voromqa
```

`--voromqa-file PATH` supplies a different potential file. VoroMQA energies are
pseudo-energies, not calibrated physical energies. For VoroMQA, treat `--gamma`
as a tuning scale for the weights.

## Network Models

### Distance Cutoff

The default network connects every atom pair separated by at most `--cutoff` Å.
The CLI default is 5 Å, which is the all-atom cutoff used for the NOLB-style
workflow.

```sh
elasticrab -i protein.pdb --cutoff 5.0 animate
```

Every cutoff spring has unit relative weight, so the global spring constant
`--gamma` sets the common stiffness scale.

### Voronoi Tessellation

Voronoi tessellation is Elasticrab's distinctive network option. It replaces a
distance cutoff with contact geometry.

With `--voronota`, Elasticrab calls `voronota-ltr` in process. It represents
each selected atom as a ball with its parsed coordinates and Voronota radius,
then computes radical-tessellation contacts with a 1.4 Å solvent probe. Each
returned contact contains two atom indices, `id_a` and `id_b`, and the shared
cell-face area, $A_{ij}$. Elasticrab creates one elastic spring for each
returned contact, using `id_a` and `id_b` as the spring endpoints.

Contact area is a physical proxy for mechanical coupling. In a coarse elastic
model, a broad packing interface should resist relative displacement more than a
small grazing contact. Elasticrab therefore uses contact area as a relative
stiffness, not as a first-principles force constant. It first computes the mean
contact area

$$
\bar{A} = \frac{1}{N_c}\sum_{(i,j)} A_{ij}
$$

where $N_c$ is the number of `voronota-ltr` contacts. It then assigns the
dimensionless spring weight

$$
w_{ij} = \frac{A_{ij}}{\bar{A}}
$$

so the effective spring constant is $k_{ij}=\gamma w_{ij}$. This normalization
gives the Voronoi network a mean spring weight of 1, so `--gamma` keeps the same
role as in the cutoff model.

The spring rest length is the native distance between the two atoms. During
Hessian assembly, each Voronoi spring contributes the usual ANM block with the
area-derived weight:

$$
H_{ij} = -\frac{\gamma w_{ij}}{d_{ij}^2}\,\Delta\mathbf{r}_{ij}\Delta\mathbf{r}_{ij}^{T}
$$

where $\Delta\mathbf{r}_{ij}=\mathbf{r}_j^0-\mathbf{r}_i^0$ and
$d_{ij}=|\Delta\mathbf{r}_{ij}|$. The diagonal blocks receive the opposite row
sums. This construction makes the network occlusion-aware: an atom between two
others can remove or reduce their shared Voronoi face, whereas a distance cutoff
would still connect them. The connectivity still differs from the cutoff
network, so absolute frequencies from the two models should not be compared
directly.

Voronoi tessellation matters in two places:

- **Spring construction** with `--voronota`: contacts define the spring graph,
  and contact areas define relative spring stiffness.
- **Energy scoring** with `energy --voromqa`: each frame is tessellated again
  and scored with a contact-area potential.

## Methodology

Elasticrab uses the Anisotropic Network Model (ANM). Each atom is a point in a
spring network. For a conformation with coordinates $\mathbf{r}$, the elastic
spring energy is

$$
E_\text{spring} =
\frac{1}{2}\gamma
\sum_{(i,j)} w_{ij}
\left(\left|\mathbf{r}_j - \mathbf{r}_i\right| - d_{ij}^0\right)^2
$$

where $d_{ij}^0$ is the native distance for spring $(i,j)$, $w_{ij}$ is the
relative spring weight, and $\gamma$ is the global spring constant. In the
cutoff network, $w_{ij}=1$. In the Voronoi network, $w_{ij}=A_{ij}/\bar{A}$.

The Hessian is the second derivative of this energy at the native structure:

$$
H = \left.\frac{\partial^2 E_\text{spring}}{\partial \mathbf{r}\,\partial \mathbf{r}}\right|_{\mathbf{r}=\mathbf{r}^0}
$$

The CLI uses mass-weighted modes, so it diagonalizes

$$
M^{-1/2} H M^{-1/2} \mathbf{u}_k = \lambda_k \mathbf{u}_k
$$

The reported frequency is proportional to $\sqrt{\lambda_k}$. In an ANM, the
absolute scale depends on $\gamma$, so mode shapes, frequency ratios, and
relative amplitudes are usually more useful than absolute frequencies.

The CLI also uses Rotation-Translation Blocks. Each residue is treated as a
rigid body with translational and rotational degrees of freedom. If $P$ maps
rigid-block coordinates to Cartesian displacements and $H_m$ is the mass-weighted
Hessian, the reduced problem is

$$
H_\text{RTB} = P^\mathrm{T} H_m P
$$

The solved modes are lifted back to per-atom displacements for reporting and
trajectory writing.

Nonlinear displacement is NOLB-inspired rigid-block extrapolation. It unweights
the reduced translational and angular velocities as in NOLB, then applies each
block as one rigid motion, so bonds within a block remain fixed even for large
amplitudes. Elasticrab rotates each block about its center of mass and then
translates it. NOLB's full nonlinear update additionally folds translation
perpendicular to the rotation axis into a rotation about a shifted center.

The collectivity report uses the Brüschweiler collectivity $\kappa$:

$$
\kappa =
\frac{1}{N}
\exp\left(-\sum_i p_i \ln p_i\right)
$$

where $p_i$ is the normalized squared displacement of atom $i$ in the physical
mode. $\kappa=1$ means all atoms participate equally; $\kappa=1/N$ means the
mode is localized to one atom.

Predicted crystallographic B-factors use

$$
B_i = \frac{8\pi^2}{3}\left<\Delta r_i^2\right>
$$

with

$$
\left<\Delta r_i^2\right> =
RT \sum_k \frac{\left|\mathbf{v}_k(i)\right|^2}{\lambda_k}
$$

over non-zero modes. `--b-factor-fit` uses a separate non-mass-weighted all-atom
solve for this fit, because B-factors describe configurational fluctuations.
It reports the fitted $\gamma$ and the Pearson correlation. If the fit fails,
Elasticrab warns and falls back to `--gamma`.

### VoroMQA Energy

`energy --voromqa` uses the bundled VoroMQA v1 atom-level potential. The score is
a sum of two contact-area terms:

$$
E_\text{VoroMQA} =
\sum_{(i,j)} A_{ij}\,e(t_i,t_j,c_{ij}) +
\sum_i S_i\,e(t_i,\text{solvent})
$$

Here $t_i$ is the atom type, $A_{ij}$ is the Voronoi contact area, $c_{ij}$ is
the contact class, and $S_i$ is the solvent-accessible area.

Elasticrab applies the following inclusion rules:

- A pair contact is scored only if both atom types are present in the potential.
- Same-chain contacts with residue-number separation 0 or 1 are skipped. These
  are same-residue and sequence-adjacent contacts, whose area is dominated by
  covalent geometry.
- Same-chain contacts with residue-number separation 2 or more are scored.
  Inter-chain contacts are scored regardless of residue numbers.
- Scored pair contacts use the `central_sep2` class when the `voronota-ltr`
  contact is central, and `sep2` otherwise. The `sep1` classes are not used
  after sequence-adjacent contacts are skipped.
- A known atom type with no matching pair/class coefficient contributes zero for
  that pair contact.

The solvent term includes every atom whose type has a solvent coefficient. Atoms
without a coefficient are skipped from both the pair and solvent terms, and the
CLI prints a warning. If an atom has no Voronoi cell in a frame, Elasticrab
treats it as fully exposed and uses $S_i=4\pi(r_i+1.4)^2$. Custom potential
files must use the centrality-only classes `central_sep1`, `central_sep2`,
`sep1`, and `sep2`; files with peripheral classes are rejected.

Elasticrab evaluates this score in process with `voronota-ltr`, not with the
full Voronota executable. The bundled coefficients were derived for full
Voronota areas, so the absolute pseudo-energy is approximate. The intended use
is native-referenced reweighting, where a mostly systematic offset should cancel
between $E_\text{frame}$ and $E_\text{native}$.

## Output Files

Trajectory format is chosen from the output extension:

- `.pdb`: multi-model PDB.
- `.xtc`: XTC trajectory.
- any other extension: PDB.

If no output path is given, Elasticrab writes beside the input structure:

- `animate`: `<input>_mode1.pdb`, or one file per mode.
- `transition`: `<input>_morph.pdb`.
- `energy`: `<input>_modes.pdb` plus the required CSV path.

Elasticrab refuses to overwrite the input structure.

## Validation

The tests are part of the intended scientific surface of the project. They check
the numerical method against independent tools and lock down CLI behavior.

- **ProDy ANM**: the 1UBI C-alpha Hessian reconstructs ProDy's reference Hessian
  with maximum difference below `1e-5`; eigenvalues match to `1e-4`.
- **ProDy RTB**: the RTB-reduced spectrum for a truncated 2GB1 C-alpha model
  matches ProDy's reference spectrum to `1e-5`.
- **NOLB / Pepsi-SAXS path**: mass-weighted RTB frequencies for crambin are
  proportional to NOLB's frequencies within `1e-3` after one global scale factor.
- **Collectivity**: Brüschweiler collectivities for crambin match NOLB's
  reported values within `0.03`.
- **Transitions**: single-shot and iterative nonlinear transitions reduce RMSD
  and are checked against NOLB reference regimes.
- **Edge cases**: tests cover rigid-body zero modes, disconnected atoms,
  explicit spring weights, invalid inputs, CLI help, mode selection, and energy
  CSV generation.

Run the full checked CLI suite with:

```sh
cargo test --features cli
```

## Performance Notes

The library has dense and partial sparse solver paths. The dense solve is useful
for small and medium systems. For large systems, use the partial solver through
`-n, --modes`, which returns only the lowest non-zero modes. The CLI feature
includes the sparse solver.

`parallel` enables multithreaded kernels. It can speed up large dense solves, but
parallel floating-point reductions are not bit-for-bit identical to serial
results. For partial sparse solves, one or two threads are often enough.

Benchmarks live in `benches/scaling.rs` and run with:

```sh
cargo bench --features sparse
```

## Rust API

The Rust API is intentionally small: `Atom`, `Spring`, `NormalModes`, `Builder`,
and error types. Use it when you already have coordinates and want normal modes
inside another Rust program. Structure parsing and CLI conveniences live outside
the core library.

```rust
use elasticrab::{Atom, NormalModes};

let atoms = vec![
    Atom { position: [0.0, 0.0, 0.0], mass: 12.0 },
    Atom { position: [3.8, 0.0, 0.0], mass: 12.0 },
    Atom { position: [3.8, 3.8, 0.0], mass: 12.0 },
];

let modes = NormalModes::builder(&atoms)
    .cutoff(15.0)
    .k_modes(3)
    .solve()
    .unwrap();
```

Library documentation is published at <https://docs.rs/elasticrab>.

## License

Elasticrab is licensed under Apache-2.0. Bundled ProDy fixtures and the VoroMQA
potential are MIT-licensed; see `tests/data/ATTRIBUTION.md`.
