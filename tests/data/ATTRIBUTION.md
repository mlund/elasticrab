# Test fixtures

These files are vendored from **ProDy** (https://github.com/prody/ProDy),
`prody/tests/datafiles/`, and are used here solely as a reference oracle for the
ANM implementation.

- `1ubi_ca.pdb` — Cα-only structure of ubiquitin (PDB 1UBI).
- `anm1ubi_hessian.coo` — reference ANM Hessian (cutoff 15 Å, γ = 1), sparse COO,
  1-indexed `i j value`, symmetric (only one triangle stored).
- `anm1ubi_evalues.dat` — reference ANM eigenvalues (lowest 36), `index value`.

ProDy is distributed under the MIT License:

> Copyright (C) 2010-2014 University of Pittsburgh
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

The `elasticrab` crate itself is licensed under Apache-2.0; bundling these
MIT-licensed fixtures is compatible with that license.
