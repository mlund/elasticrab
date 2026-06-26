# Why Pepsi-SAXS is not used as a weighted reference oracle

`elasticrab` validates its **unit-mass** ANM against ProDy (an exact, per-atom
match). It was natural to ask whether the Pepsi-SAXS binary could serve as the
reference oracle for the **mass-weighted** path. Empirically, it cannot — the
two compute structurally different models. This note records what was found by
running `~/bin/Pepsi-SAXS` (v2.6, 29 Jan 2020).

## How Pepsi's NMA was invoked

```
# theoretical curve, fed back as the "experimental" curve to enable --opt
Pepsi-SAXS crambin.pdb -o theo.out
Pepsi-SAXS crambin.pdb exp.dat --opt --modes 10 --useMasses
```

`--opt` (flexible optimization) enables the NMA path; it requires an
experimental curve. `--useMasses` turns on mass-weighting (off by default in
this build). `GAMMA` is fixed at 1.0; the cutoff defaults to 5 Å.

## What the log revealed (crambin, PDB 1EJG: 46 residues, 327 atoms)

```
All-atom Hessian size :     981 x 981      # 3 × 327 atoms — NO hydration nodes added
Reduced Hessian size :      276 x 276      # 6 DOF × 46 residues
Warning: Did not converge, increasing nModes to ...
```

Three findings:

1. **Per-residue RTB reduction.** The all-atom Hessian (981 = 3·327) is reduced
   to 276 = 6·46 before solving — six rigid-body DOF per residue, i.e. one
   Rotation-Translation Block per residue. The block file format keys on
   *residue* IDs (`cSAS.cpp` block reader), so blocks can never be finer than a
   residue. Pepsi therefore reports **RTB modes**, not per-atom ANM modes.

2. **Iterative/sparse solver.** "Did not converge, increasing nModes" shows an
   iterative partial eigensolver (lowest-*k* modes), not a dense full
   decomposition. (This also answers an open design question: Pepsi is *not*
   dense.)

3. **No hydration nodes in this build's NMA.** 981 = 3·327 equals the heavy-atom
   count exactly, so — unlike the newer Pepsi source tree — this binary does not
   add water pseudo-atoms to the elastic network.

## Why a direct comparison is impossible

- **CA-only input segfaults.** Forcing one atom per residue (a Cα model) makes
  each RTB block a single point with zero rotational inertia; the inertia-tensor
  inverse-square-root in the block construction divides by zero → `SIGSEGV`.
  So Pepsi cannot be coerced into a per-atom spectrum.
- **RTB ≠ ANM.** elasticrab is a per-atom ANM; Pepsi's reported frequencies come
  from a per-residue RTB projection of a mass-weighted all-atom Hessian. The
  eigenvalues are not the same quantities, so no tolerance makes them agree.

Reproducing Pepsi's numbers would require implementing all-atom Hessian
construction **plus** RTB projection **plus** mass-weighting in elasticrab —
i.e. re-implementing Pepsi's (external NOLB) engine, which is explicitly out of
scope for this library.

## What we validate instead

- **Unit-mass path** → exact golden test vs ProDy's 1UBI Hessian + eigenvalues.
- **Mass-weighted path** → analytic checks: the diatomic reduced-mass relation
  `ω² = γ(1/m₁ + 1/m₂)`, and the equal-mass invariant `spectrum(M=mI) =
  spectrum(unit)/m`.

If per-residue RTB is wanted in the future, the right reference oracle is
ProDy's own `RTB` class (it ships `rtb2gb1` fixtures), not the Pepsi binary.

## RTB is now implemented — and the oracle is ProDy, not the binary

`NormalModes::with_blocks` adds the per-residue RTB reduction. It is validated by
`tests/prody_rtb.rs`: the eigenvalue spectrum of our reduced Hessian `Pᵀ H P`
matches ProDy's reference `rtb2gb1_hessian.coo` exactly. Spectra are compared
(not the matrices) because a block's rotational basis is only defined up to
orientation, so `Pᵀ H P` is basis-dependent while its eigenvalues are not.

### Why not assert against the NOLB binary directly

NOLB (`~/Downloads/NOLB`, the authentic Grudinin engine Pepsi wraps) *can*
generate RTB fixtures — it is ARPACK-based (lowest-k, confirmed), takes a rigid
block file (`--blocks`, new format `chain:start:end` intervals, e.g. `A1:8`), and
emits frequencies as JSON (`-j`, under `["Doing NMA"]["Frequencies"]["value"]`).
For crambin it gives a clean run (`-n 10 -c 5`):

```
0.006593 0.006788 0.008845 0.009876 0.010521 0.011923 0.012043 0.013044 0.013821 0.013881
```

But an *exact* cross-check is brittle, because matching NOLB means matching every
one of its engine conventions, several of them surprising:
- it reads **742 atoms** for crambin (it keeps hydrogens; Pepsi kept 327);
- it reports a **null-space of 2**, not the textbook 6;
- its own mass set and frequency definition (`√λ` up to an unknown constant).

A scale-invariant comparison doesn't rescue this, since relative frequency
spacing depends on those mass/hydrogen choices. So NOLB confirms the *model* we
implemented and is recorded here as a reference, but the committed assertion uses
ProDy's exact, self-contained `rtb2gb1` fixtures.
