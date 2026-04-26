# ADR-002: Use Real Colorimetry Data, `colour-science`, And Versioned cGAN Scaling Metadata

## Status

Accepted

## Date

2026-04-25

## Context

The repository currently runs a working Ag-SiO2-Ag inverse-design pipeline, but several core behaviors are still prototype-grade:

- color matching functions and D65 are embedded as pseudo Gaussian approximations in `backend/app/algorithms/optics.py`
- Lab and thickness are both scaled with min-max normalization in `backend/app/algorithms/cgan.py`
- cGAN checkpoints do not record scaling strategy or checkpoint schema version
- nearest retrieval is effectively tied to Lab Euclidean distance

This makes paper-style comparison and future experiment tracking harder than it needs to be. Once real colorimetry data and a new Lab scaling rule are introduced, the repository also needs a reliable way to distinguish old and new checkpoints.

## Decision

Adopt the following decisions as the new baseline:

1. Use `refer_data/D65.csv` as the source of truth for illuminant D65.
2. Use `refer_data/tristimulus.csv` as the source of truth for tristimulus values.
3. Move colorimetry reference loading into a dedicated reusable module.
4. Use `colour-science` behind local wrapper functions for:
   - `XYZ -> Lab`
   - `XYZ -> sRGB`
   - `Delta E 2000`
5. Scale Lab with standardization and thickness with min-max normalization.
6. Introduce explicit checkpoint schema versioning and explicit scaling metadata.
7. Expose explicit cGAN training hyperparameters as:
   - `generator_learning_rate`
   - `discriminator_learning_rate`
   - `steps_per_batch`
8. Standardize retrieval metric names as:
   - `euclidean_lab`
   - `delta_e_2000`

## Alternatives Considered

### Keep The Current Pseudo Colorimetry

- Pros: no dependency changes and no baseline shift
- Cons: lower physical fidelity, weaker paper comparison, and hidden approximation debt
- Rejected: the repository is far enough along that real reference data is now more valuable than approximation convenience

### Reimplement All Color Science Locally

- Pros: fewer third-party dependencies and full control
- Cons: more maintenance burden and higher risk of silent formula drift
- Rejected: mature library behavior is preferable, but should be wrapped locally to avoid spreading dependency details

### Keep Min-Max Scaling For Lab

- Pros: simpler checkpoint migration and fewer code changes
- Cons: Lab ranges become dataset-bound and less stable across subsets
- Rejected: Lab behaves better as a centered standardized feature, while thickness remains naturally bounded and compatible with min-max scaling

### Overwrite Old Checkpoints In Place

- Pros: less loader code
- Cons: silent incompatibility risk and loss of experiment traceability
- Rejected: schema versioning is safer and cheaper than debugging incorrect samples later

### Switch Runtime Retrieval Directly To `delta_e_2000`

- Pros: better perceptual metric on paper
- Cons: removes the current baseline before comparative evidence is collected
- Deferred: support both metrics first, then change defaults only after comparison

## Consequences

- Color outputs and Delta E values will intentionally shift once real reference data is used.
- Existing checkpoints need explicit compatibility handling and may require retraining.
- Experimental runs will become easier to compare because metrics and artifacts will record:
  - colorimetry data source
  - scaling strategy
  - checkpoint schema
  - training hyperparameters
  - retrieval metric
- The repository gains one additional dependency, `colour-science`, but keeps its usage localized.
- Retrieval behavior can now be compared under a perceptual metric without deleting the existing Euclidean baseline.
