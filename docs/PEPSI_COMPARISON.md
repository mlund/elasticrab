# How elasticrab relates to Pepsi-SAXS and NOLB

Pepsi-SAXS computes normal modes with the NOLB engine (Grudinin's NOn-Linear
rigid Block method). NOLB does **not** run a plain per-atom ANM: it reduces a
mass-weighted all-atom Hessian to per-residue rigid blocks (RTB) before solving.
elasticrab implements that same RTB model and reproduces both ProDy's and NOLB's
reference spectra exactly. This note records the evidence, gathered by running
`~/bin/Pepsi-SAXS` (v2.6, 29 Jan 2020) and `~/Downloads/NOLB` (v1.1).

## What Pepsi and NOLB compute

Running Pepsi's NMA on crambin (PDB 1EJG, 46 residues, 327 heavy atoms):

```
# feed Pepsi's own theoretical curve back as the "experimental" curve to enable --opt
Pepsi-SAXS crambin.pdb -o theo.out
Pepsi-SAXS crambin.pdb exp.dat --opt --modes 10 --useMasses
```

`--opt` enables the NMA path and needs a curve; `--useMasses` turns on
mass-weighting; γ = 1 and the cutoff defaults to 5 Å. The log shows:

```
All-atom Hessian size :     981 x 981      # 3 × 327 atoms
Reduced Hessian size :      276 x 276      # 6 DOF × 46 residues
Warning: Did not converge, increasing nModes to ...
```

Three things follow:

1. **Per-residue RTB reduction.** The 981×981 all-atom Hessian reduces to
   276 = 6·46 before solving — six rigid-body DOF per residue. The block-file
   reader keys on residue IDs (`cSAS.cpp`), so blocks never go finer than a
   residue. Pepsi reports RTB modes, not per-atom ANM modes.
2. **Iterative solver.** "Did not converge, increasing nModes" reveals an
   iterative partial eigensolver (lowest-*k* modes), not a dense decomposition.
3. **No hydration nodes.** 981 = 3·327 matches the heavy-atom count exactly, so
   this build adds no water pseudo-atoms to the network (unlike the newer source
   tree).

## Why a per-atom ANM cannot be compared naively

elasticrab's plain ANM and Pepsi's RTB are different models, so their
frequencies are different quantities and no tolerance makes them agree. Forcing
Pepsi onto a per-atom grid fails outright: a Cα model gives each RTB block a
single point with zero rotational inertia, and the inertia inverse-square-root
then divides by zero (`SIGSEGV`). The answer is not to coerce Pepsi but to
implement RTB in elasticrab.

## elasticrab implements RTB — validated against ProDy and NOLB

`NormalModes::with_blocks` adds the per-residue RTB reduction, checked two ways:

- **`tests/prody_rtb.rs` (unit-mass, exact).** The eigenvalue spectrum of our
  reduced Hessian `Pᵀ H P` matches ProDy's reference `rtb2gb1_hessian.coo`. We
  compare spectra, not matrices: a block's rotational basis is fixed only up to
  orientation, so `Pᵀ H P` is basis-dependent while its eigenvalues are not.
- **`tests/nolb_rtb.rs` (mass-weighted, the engine Pepsi wraps).** Our spectrum
  matches NOLB's frequencies for heavy-atom crambin.

### Reconciling with NOLB

The NOLB User Guide (v1.1) settles every convention: equations §1.2–1.5 spell
out exactly the model we implemented — mass-weighted stiffness
`K_w = M^{-1/2} K M^{-1/2}`, frequency `√λ`, translation columns `√(mₖ/M_b)`,
rotation columns `√mₖ · I^{-1/2} · [rₖ − r_COM]×`, reduced as `Pᵀ K_w P`.
elasticrab's RTB is therefore identical to NOLB's by construction.

Two bookkeeping differences remained, both now resolved:

- **Hydrogens.** NOLB reads 742 atoms for crambin because it keeps hydrogens.
  The test feeds NOLB and elasticrab the same heavy-atom structure
  (`crambin_heavy.pdb`, 327 atoms), removing the difference.
- **Unit constant.** NOLB scales its frequency by a fixed factor (empirically
  1/√1000), so the test compares the spectra for proportionality, not absolute
  value.

Agreement is then essentially exact: across the 10 lowest non-zero modes, the
elasticrab/NOLB ratio is constant to ~6 digits. elasticrab's raw `√λ` (0.08155,
0.08853, …) also reproduces the Pepsi frequencies (0.0815544, 0.0885307) to five
figures, since Pepsi reports `√λ` without NOLB's constant. The reference
frequencies are vendored (`nolb_crambin_freqs.txt`), so the test never runs the
binary.
