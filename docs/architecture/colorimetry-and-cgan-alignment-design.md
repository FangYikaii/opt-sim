# Colorimetry And cGAN Alignment Design

## Status

- Status: Proposed
- Date: 2026-04-25
- Related PRD: [cGAN / optics alignment PRD](../planning/cgan-optics-data-alignment-prd.md)

## Overview

This document turns the PRD into an implementation-ready technical design for three closely related changes:

1. Replace pseudo D65 and pseudo tristimulus functions with real reference data from `refer_data/`.
2. Refactor the cGAN preprocessing and checkpoint format so thickness uses normalization while Lab uses standardization.
3. Extend retrieval so both `euclidean_lab` and `delta_e_2000` are supported and measurable.

The design goal is to improve physical and colorimetric fidelity without causing uncontrolled breakage in the existing runtime inverse-design and paper-reproduction paths.

## Problem Statement

The current repository has a working end-to-end pipeline, but the colorimetry and training metadata are still too approximate for trustworthy paper-style comparison:

- `backend/app/algorithms/optics.py` hardcodes pseudo `_X_BAR`, `_Y_BAR`, `_Z_BAR`, and `_ILLUMINANT_D65`.
- `backend/app/algorithms/cgan.py` uses min-max normalization for both `Lab` and thickness.
- Saved checkpoints do not explicitly encode scaling strategy or checkpoint schema version.
- Retrieval in both runtime inference and reproduction scripts is based on Lab Euclidean distance instead of a configurable metric.

These problems are coupled. Once real reference data and new scaling are introduced, older checkpoints can become semantically invalid even if they still load.

## Goals

- Centralize colorimetry reference loading into one reusable module.
- Keep the public optics function names stable where possible.
- Make scaling strategy explicit, serializable, and versioned.
- Add retrieval metric selection without forcing an immediate UI or API redesign.
- Preserve a safe migration path from current checkpoint files.

## Non-Goals

- No redesign of the thin-film transfer-matrix core.
- No change to the current Ag-SiO2-Ag design vector dimensionality.
- No first-pass switch to a 1 nm wavelength grid.
- No mandatory frontend setting for retrieval metric in this design phase.

## Design Constraints

- `refer_data/D65.csv` is the source of truth for illuminant D65.
- `refer_data/tristimulus.csv` is the source of truth for tristimulus values.
- The current optics grid remains `380-780 nm` in `5 nm` steps.
- Runtime inverse design should keep working even if no new checkpoint has been trained yet.
- Existing code paths should fail loudly on incompatible checkpoint metadata instead of silently producing wrong samples.

## Architecture Changes

### 1. New Colorimetry Data Module

Add a dedicated module:

- `backend/app/algorithms/colorimetry_data.py`

This module owns:

- reading `refer_data/D65.csv`
- reading `refer_data/tristimulus.csv`
- UTF-8 BOM handling
- wavelength validation
- deterministic resampling from 1 nm D65 data to the optics wavelength grid
- cached in-memory reference arrays

### 2. optics.py Becomes A Consumer, Not The Owner

`backend/app/algorithms/optics.py` should stop embedding color matching data. It should:

- keep the spectral simulation logic
- request reference colorimetry arrays from `colorimetry_data.py`
- wrap `colour-science` for `XYZ -> Lab`, `XYZ -> sRGB`, and `Delta E 2000`

This keeps the upper layer API stable while removing hidden pseudo data definitions.

### 3. cGAN Scaling Becomes Structured Metadata

`backend/app/algorithms/cgan.py` should separate:

- Lab scaling
- design scaling
- training hyperparameters
- checkpoint compatibility metadata

The current checkpoint layout is effectively unversioned. The new design introduces an explicit checkpoint schema version and explicit scaling descriptors.

### 4. Retrieval Metric Becomes A Named Contract

Both runtime inverse design and reproduction scripts should use the same metric names:

- `euclidean_lab`
- `delta_e_2000`

This avoids hidden divergence between demo runtime behavior and offline evaluation behavior.

## Module Boundaries

### colorimetry_data.py

Primary responsibility:

- authoritative loading and validation of reference colorimetry data

Public contract:

```python
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ColorimetryReference:
    wavelengths_nm: np.ndarray
    x_bar: np.ndarray
    y_bar: np.ndarray
    z_bar: np.ndarray
    d65: np.ndarray


def get_colorimetry_reference(
    wavelengths_nm: np.ndarray | None = None,
) -> ColorimetryReference:
    ...
```

Rules:

- If `wavelengths_nm` is omitted, return the native optics grid reference.
- If `wavelengths_nm` is provided, resample deterministically.
- Loading failures raise structured `ValueError` or `FileNotFoundError`.

### optics.py

Primary responsibility:

- thin-film response simulation
- spectrum integration against loaded reference data
- stable wrappers for color conversions and color-difference metrics

Public contract to preserve:

```python
def spectrum_to_xyz(spectrum: np.ndarray) -> np.ndarray: ...
def xyz_to_lab(xyz: np.ndarray) -> np.ndarray: ...
def xyz_to_srgb_hex(xyz: np.ndarray) -> str: ...
def delta_e_2000(lab_a: np.ndarray, lab_b: np.ndarray) -> float: ...
```

Internal rule:

- direct access to hardcoded `_ILLUMINANT_D65`, `_X_BAR`, `_Y_BAR`, `_Z_BAR` is removed

### cgan.py

Primary responsibility:

- scaling
- training
- checkpoint save/load
- sampling from checkpoints

New data contracts:

```python
from dataclasses import dataclass
from typing import Literal
import numpy as np


ScalingType = Literal["min_max", "standardization"]


@dataclass(frozen=True)
class MinMaxScalingStats:
    scaling_type: Literal["min_max"]
    min_values: np.ndarray
    max_values: np.ndarray


@dataclass(frozen=True)
class StandardScalingStats:
    scaling_type: Literal["standardization"]
    mean_values: np.ndarray
    std_values: np.ndarray


@dataclass(frozen=True)
class CganTrainingConfig:
    generator_learning_rate: float
    discriminator_learning_rate: float
    steps_per_batch: int
    noise_dim: int
    seed: int


@dataclass
class CganBundle:
    generator: Generator
    lab_regressor: LabRegressor
    lab_scaling: StandardScalingStats
    design_scaling: MinMaxScalingStats
    training_config: CganTrainingConfig
    device: str
    losses: list[dict[str, float]]
    best_generator_state_dict: dict[str, torch.Tensor] | None = None
    selected_checkpoint_epoch: int | None = None
    selected_checkpoint_metric_name: str | None = None
    selected_checkpoint_metric_value: float | None = None
```

Compatibility note:

- Existing field names like `lab_min`, `lab_max`, `design_min`, `design_max` can remain as transitional computed properties during migration if that simplifies call sites.

### inverse_design.py

Primary responsibility:

- runtime candidate retrieval
- runtime cGAN candidate generation
- ranking and refinement

New contract:

```python
from typing import Literal

RetrievalMetric = Literal["euclidean_lab", "delta_e_2000"]


def run_inverse_design(
    target_hex: str,
    top_k: int = 3,
    theta_deg: float = 0.0,
    polarization: Polarization = "unpolarized",
    retrieval_metric: RetrievalMetric = "euclidean_lab",
) -> InverseDesignResult:
    ...
```

Internal rule:

- `NearestNeighbors` remains acceptable only for the `euclidean_lab` branch
- `delta_e_2000` branch uses direct scoring over the current small grid

### train_cgan_reproduction.py

Primary responsibility:

- offline training
- offline evaluation
- artifact export

New CLI contract:

```text
--generator-learning-rate
--discriminator-learning-rate
--steps-per-batch
--retrieval-metric
```

Default values:

- `--generator-learning-rate=1e-3`
- `--discriminator-learning-rate=2e-4`
- `--steps-per-batch=1`
- `--retrieval-metric=euclidean_lab`

The default retrieval metric stays conservative during migration. The design intentionally supports A/B comparison before flipping defaults.

## Data Design

### Reference CSV Semantics

`refer_data/D65.csv`

- no header
- 2 columns
- schema: `wavelength_nm, d65_value`
- expected coverage: `380-780` inclusive, 1 nm step

`refer_data/tristimulus.csv`

- no header
- 5 columns
- schema: `wavelength_nm, x_bar, y_bar, z_bar, d65_value`
- expected coverage: `380-780` inclusive, 5 nm step

Usage policy:

- D65 comes from `D65.csv`
- the fifth column in `tristimulus.csv` is validation-only data

### Resampling Strategy

Because the optics grid remains at 5 nm while `D65.csv` is at 1 nm:

- use linear interpolation onto `WAVELENGTHS_NM`
- validate that resampled 5 nm D65 values closely match column 5 in `tristimulus.csv`
- treat mismatch above tolerance as a data-quality error

Recommended tolerance:

- absolute difference per sample <= `1e-3` after interpolation, unless real file analysis shows a justified looser threshold

### Spectrum Integration Contract

`spectrum_to_xyz` should compute:

```text
weighted = spectrum * d65
X = sum(weighted * x_bar) / sum(d65 * y_bar)
Y = sum(weighted * y_bar) / sum(d65 * y_bar)
Z = sum(weighted * z_bar) / sum(d65 * y_bar)
```

White-point derivation:

```text
ref_white = [
  sum(d65 * x_bar) / sum(d65 * y_bar),
  1.0,
  sum(d65 * z_bar) / sum(d65 * y_bar),
]
```

The important design point is that these arrays now come from the data module, not inline pseudo functions.

## `colour-science` Integration Design

Use `colour-science` as an implementation dependency, but keep project-local wrappers so upper layers do not depend on raw library calls.

Recommended wrapper behavior:

- `xyz_to_lab(xyz)`:
  - normalize against the same reference white used in `spectrum_to_xyz`
  - return Lab in the same shape as today
- `xyz_to_srgb_hex(xyz)`:
  - convert XYZ to sRGB
  - clip to `[0, 1]`
  - encode as lowercase hex
- `delta_e_2000(lab_a, lab_b)`:
  - call library implementation
  - return Python `float`

Reason for wrappers:

- library API changes stay local
- tests can target stable internal functions
- runtime code remains clean and unsurprising

## Scaling Design

### Lab Scaling

Lab uses standardization because:

- Lab dimensions are centered and signed
- min-max on bounded observed samples can make small dataset range choices overly influential
- standardization is easier to compare across training subsets

Rule:

```text
lab_scaled = (lab - lab_mean) / max(lab_std, epsilon)
```

### Design Scaling

Thickness remains per-dimension normalization because:

- the three design values are bounded by explicit fabrication windows
- generator outputs already use `Sigmoid`, which maps naturally to normalized design coordinates

Rule:

```text
design_scaled = (design - design_min) / max(design_max - design_min, epsilon)
```

### Scaling Helper Interfaces

Keep scaling helpers explicit and type-specific instead of a single generic helper with hidden branches.

Recommended internal helpers:

```python
def normalize_design(values: np.ndarray, stats: MinMaxScalingStats) -> np.ndarray: ...
def denormalize_design(values: np.ndarray, stats: MinMaxScalingStats) -> np.ndarray: ...
def standardize_lab(values: np.ndarray, stats: StandardScalingStats) -> np.ndarray: ...
def destandardize_lab(values: np.ndarray, stats: StandardScalingStats) -> np.ndarray: ...
```

## Checkpoint Format Design

### Current Problem

Current checkpoints are structurally loadable but semantically ambiguous. They do not say:

- which scaling rule was used
- whether Lab is min-max or standardized
- which training hyperparameters were used
- whether the file belongs to the old or new schema

### New Checkpoint Schema

Introduce:

```text
checkpoint_format_version = 2
```

Recommended serialized shape:

```python
{
    "checkpoint_format_version": 2,
    "generator_state_dict": ...,
    "lab_regressor_state_dict": ...,
    "lab_scaling_type": "standardization",
    "lab_mean": [...],
    "lab_std": [...],
    "design_scaling_type": "normalization",
    "design_min": [...],
    "design_max": [...],
    "generator_learning_rate": 1e-3,
    "discriminator_learning_rate": 2e-4,
    "steps_per_batch": 1,
    "noise_dim": 2,
    "seed": 42,
    "device": "...",
    "selected_checkpoint_epoch": ...,
    "selected_checkpoint_metric_name": ...,
    "selected_checkpoint_metric_value": ...,
}
```

### Backward Compatibility Strategy

Supported formats:

- version 2: new explicit format
- legacy unversioned format: treated as version 1 compatibility path

Loader behavior:

1. If `checkpoint_format_version == 2`, use explicit scaling metadata.
2. If version is missing but legacy fields `lab_min`, `lab_max`, `design_min`, `design_max` exist, load as version 1 compatibility mode.
3. If neither structure is recognized, raise a readable error.

Compatibility policy:

- version 1 checkpoints may be loaded only for legacy inference paths that still understand min-max Lab
- if the runtime pipeline has been fully switched to standardized Lab, version 1 checkpoints should raise an incompatibility error and instruct retraining

This avoids the worst failure mode: loading old weights into a new scaling regime with no warning.

## Retrieval Design

### Metric Enumeration

Use the same names everywhere:

- `euclidean_lab`
- `delta_e_2000`

### Runtime Retrieval Algorithm

For `euclidean_lab`:

- continue using `NearestNeighbors`

For `delta_e_2000`:

- compute `delta_e_2000(candidate_lab, target_lab)` for each row in the current retrieval grid
- select the top K lowest scores

Given the current grid size, full scoring is acceptable and simpler than building a custom metric index.

### Future Scale Strategy

If retrieval sample count grows substantially:

1. coarse candidate selection by Lab Euclidean distance
2. final reranking by `delta_e_2000`

That design is compatible with this naming contract and does not require an API change later.

## Artifact And Metrics Design

The following metadata should appear in `metrics.json` and be easy to inspect:

```python
{
    "colorimetry": {
        "illuminant_source": "refer_data/D65.csv",
        "tristimulus_source": "refer_data/tristimulus.csv",
        "wavelength_grid_nm": [380.0, 385.0, ...],
        "conversion_backend": "colour-science",
    },
    "scaling": {
        "lab": "standardization",
        "design": "normalization",
    },
    "training": {
        "generator_learning_rate": ...,
        "discriminator_learning_rate": ...,
        "steps_per_batch": ...,
    },
    "retrieval_metric": "euclidean_lab" | "delta_e_2000",
}
```

This metadata is not just documentation. It is required for comparing experimental runs fairly.

## Migration Plan

### Step 1

Introduce `colorimetry_data.py` and tests without changing public optics function names.

### Step 2

Replace pseudo colorimetry data inside `optics.py` with loaded reference arrays.

### Step 3

Add `colour-science` wrappers and rebaseline optics tests.

### Step 4

Refactor cGAN scaling helpers and bundle metadata while preserving current call sites with transitional adapters if needed.

### Step 5

Introduce checkpoint version 2 save/load support.

### Step 6

Add retrieval metric option to runtime inference and reproduction scripts.

### Step 7

Update docs, smoke commands, artifact inspection, and operator guidance.

This order minimizes simultaneous breakage across optics, training, and inference.

## Testing Strategy

### New Tests

- `backend/tests/test_colorimetry_data.py`
  - validates CSV loading
  - validates BOM handling
  - validates D65 interpolation onto the optics grid
- optics tests
  - white-point sanity
  - flat-spectrum XYZ/Lab stability
  - `delta_e_2000` wrapper sanity
- cGAN tests
  - checkpoint v2 save/load
  - legacy checkpoint compatibility behavior
  - Lab standardization metadata is persisted
- retrieval tests
  - `euclidean_lab` branch returns deterministic candidates
  - `delta_e_2000` branch returns deterministic candidates

### Regression Checks

- the runtime inverse-design path still returns top K candidates
- training smoke runs still export:
  - `loss_history.csv`
  - `candidate_samples.csv`
  - `retrieval_metric_comparison.json`
  - `metrics.json`
  - `generator_checkpoint.pt`

## Risks And Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Real colorimetry changes all current color outputs | High | Add explicit regression tests and document the intentional baseline shift |
| `colour-science` expects different XYZ scaling assumptions | High | Keep wrappers local and write tests against known normalized XYZ behavior |
| Legacy checkpoints become misleading | High | Add checkpoint versioning and fail loudly on incompatible metadata |
| Retrieval by `delta_e_2000` is slower | Medium | Use full scoring now, coarse-to-fine later if needed |
| Too many module changes land at once | Medium | Keep public optics function names stable and migrate in ordered steps |

## Open Questions

- Should runtime inference expose `retrieval_metric` through the API immediately, or stay internal until A/B results are collected?
- Should version 1 checkpoints be loadable only for offline artifact inspection, or also for runtime sampling during transition?
- Do we want to surface scaling and retrieval metadata in the frontend algorithm overview once implementation lands?
