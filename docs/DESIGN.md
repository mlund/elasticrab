# Design notes

Principles behind elasticrab's architecture, for contributors. The README stays
user-facing; the implementation detail lives here.

- **Deep, minimal interface.** Four public items (`Atom`, `Params`,
  `NormalModes`, `Error`), unchanged across three features — `mass_weighted`,
  `k_modes`, and `with_blocks` each absorbed a feature behind an existing
  parameter, changing internals, not signatures.
- **Validate exactly, against independent code.** Golden tests against ProDy
  (1UBI, 2GB1) and NOLB — the engine Pepsi-SAXS wraps — for mass-weighted RTB.
  Where the Pepsi binary's conventions were hidden, they were derived, not
  guessed; see [`PEPSI_COMPARISON.md`](PEPSI_COMPARISON.md).
- **Pay for capability only when used.** The default build is nalgebra-only and
  dense; the partial solvers and `faer` sit behind the `sparse` feature (`faer`
  trimmed, `divan` dev-only). `k_modes` without it errors rather than silently
  running slow. Since `sparse` pulls faer anyway, the dense `eigen::solve` also
  uses faer's SIMD eigensolver there (rayon off → deterministic); nalgebra's
  scalar solver remains the default-build fallback.
- **Algorithm follows the evidence.** Dense for exactness; shift-invert Lanczos
  where the soft modes resist plain Lanczos; regular-mode Lanczos for matrix-free
  RTB, skipping the factorization that dominates large systems — NOLB's trade.
- **Validated layers.** Dense → RTB → sparse partial → matrix-free RTB, each
  cross-checked then simplified, reusing one Lanczos loop, assembler, and
  projection.
