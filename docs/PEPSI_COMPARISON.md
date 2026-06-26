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
