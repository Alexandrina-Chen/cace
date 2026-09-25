# ASE and external LES compatibility

## Scope and versions

This change updates stored-label reads, E0 fitting, ASE calculator/evaluator
interfaces, serialization, dependency metadata and MD examples. CACE's
representation, energy heads, derivative modules and LES implementation are
unchanged. External LES remains optional.

Validation used:

- CACE base: `3f1a66552684d7f31710d5f9b98e8d56d53eb6a0`.
- External LES: `35b971a650cf30adb84fef7138761e81aa47b3a2` from
  https://github.com/ChengUCB/les.
- Python 3.12.9, Torch 2.8.0, NumPy 1.26.4, matscipy 1.0.0.
- ASE 3.22.1 and 3.29.0.

## Interface changes

- Reference reads honor exact configured keys and metadata precedence, then
  use matching valid stored calculator results without evaluating the
  calculator or applying constraints. Requested stale results raise an error;
  missing optional labels remain absent. Arrays and key mappings are copied.
- E0 fitting uses the same reader and configured CLI energy key. The CLI
  recomputes offsets rather than reusing an unkeyed `avge0.pkl` cache.
- Calculator graphs use calculator-free atom copies. Recognized conservative
  models expose offset-inclusive, converted `free_energy`; direct-force models,
  force-only external-field corrections and uncertified compositions do not.
  Stress is advertised only when enabled; scalar charge output has shape `(N,)`.
- Evaluator input forms share device transfer, offsets, stress requests and
  unit conversion. File output is optional and requires batch size one.
  Custom prediction names preserve valid standard references; standard names
  replace corresponding references without duplicate storage or stale
  `free_energy`. Graph serialization accepts omitted predictions.
- The ASE requirement is `ase>=3.22.1`. MD examples use `atoms.calc` and
  `temperature_K`. Full-object model paths must be trusted: loading explicitly
  uses `weights_only=False` for compatibility with Torch 2.8.

## Reproduce the portable tests

Run from the repository root in a dedicated Python 3.12.9 environment:

```bash
python -m pip install torch==2.8.0 numpy==1.26.4 matscipy==1.0.0 ase==3.22.1 pytest -e .
python -m pip install 'les @ git+https://github.com/ChengUCB/les.git@35b971a650cf30adb84fef7138761e81aa47b3a2'
OMP_NUM_THREADS=1 python -m pytest -q tests/test_stored_labels.py tests/test_*_compat.py
python -m pip install ase==3.29.0
OMP_NUM_THREADS=1 python -m pytest -q tests/test_stored_labels.py tests/test_*_compat.py
python -m pip check
```

The modern MD module is skipped on ASE 3.22.1. On an allocated CUDA device,
the same portable integration tests can be run without a scheduler script:

```bash
CACE_TEST_DEVICE=cuda OMP_NUM_THREADS=1 python -m pytest -q tests/test_external_les_compat.py tests/test_modern_md_compat.py
```

## Results and limits

| Check | CPU result |
| --- | --- |
| ASE 3.22.1 focused suite | 42 passed |
| ASE 3.29.0 suite, including modern MD | 46 passed; four expected velocity-initialization deprecation warnings |
| Final ASE 3.29.0 dependency check | No broken requirements |
| Fresh CACE and scalar external LES | Float32/float64 finite E/F/S, SR+LR addition, optimizer coverage, finite gradients, external-head update and calculator/evaluator save/load passed |
| Float64 external LES derivatives | All 15 force and 6 symmetric-strain components passed central differences |
| CUDA execution | Not yet validated |

Synthetic H/B/C/N/F tests use a 6-A cutoff, six trainable Bessel functions,
mixed radial width 12, embedding width 4, `max_l=max_nu=3`, one Bchi step
and energy-head widths `[32,16]`. Dtype is set before model construction;
lazy heads are materialized on the target device before optimizer creation.
The unchanged wrapper uses the default scalar external head `[32,16]`,
scaling 0.1, legacy Ewald backend, `sigma=1`, `dl=2` and self removal.
Descriptor-list, supplied-charge and optional response forwarding are tested.

Central differences use step `1e-5`; force tolerances are
`rtol=2e-5, atol=2e-7`, and stress tolerances `rtol=2e-5, atol=2e-8`.
A separate local audit of two existing checkpoints (CACE and internal-Ewald
CACELES) found exactly equal native E/F/S between current-base ASE 3.22.1 and
patched ASE 3.29.0. It compared an original frame, a `0.001 A` displacement
and uniform position/cell scaling `1.0001`, with `rtol=atol=1e-6`.
Those private fixtures and operational scripts are not part of this repository;
that audit does not establish external-LES compatibility.

Neighbor parity covers default `true_self_interaction=False`, periodic,
nonperiodic and orthorhombic partial-PBC cases. The existing self-interaction
routes differ when enabled: matscipy omits zero-shift self edges.
Graph-to-Atoms retains its legacy all-or-none PBC inference because graphs
lack the original partial-PBC metadata.

Scalar E/F/S is the acceptance target; multipolar training, BEC-only execution
and compiled LES backends are outside this claim. Three-step, 0.1-fs NHC-NVT
and isotropic MTK-NPT tests check conserved-energy calls, finite states and
orthorhombic shape preservation. They do not establish production stability
or equilibrium density. Existing benchmark examples remain thermostat-only.
