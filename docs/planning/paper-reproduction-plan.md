# Paper Reproduction Plan

## Goal

Prioritize a faithful reproduction of the reference paper workflow before adapting the broader problem statement or investing in frontend integration.

Reference paper:

- `需求/10.1515_nanoph-2022-0095.pdf`

Reference theory:

- `需求/Principles of Optics 60th Anniversary Edition by Max Born, Emil Wolf (z-lib.org).pdf`

## Execution Order

1. Reproduce the paper algorithm end to end.
2. Validate training, testing-set inference, and analysis outputs against the paper-style metrics.
3. Map the reproduced pipeline onto the project-specific problem description.
4. Adapt backend services to the reproduced pipeline.
5. Finish frontend and workspace integration.

## Reproduction Scope

### Stage 1: Theory alignment

- Map Born and Wolf sections `1.6.2-1.6.3` to the thin-film transfer-matrix implementation.
- Keep the multilayer optics model as the physics source of truth.
- Record the assumptions used in the current implementation:
  normal incidence, compact Ag-SiO2-Ag stack, spectrum-to-Lab conversion, and bounded thickness normalization.

### Stage 2: Dataset and forward model

- Support the paper dataset files (`training set.csv`, `testing set.csv`) directly.
- Keep a synthetic-data mode for smoke testing and local debugging.
- Validate the parameter ranges and normalization used by the paper:
  `d1` and `d3` in `0-50 nm`, `d2` in `0-1000 nm`.

### Stage 3: Model reproduction

- Train the forward `Lab regressor`.
- Train the cGAN-style inverse model with:
  generator, evaluator, latent sampling, hinge-style adversarial loss, and Lab regression loss.
- Export training losses and summary metrics in a way that can be compared to the paper.

### Stage 4: Testing-set inference and analysis

- Produce testing-set distribution comparisons.
- Report paper-style metrics including:
  `JSD`, best `ΔE`, solution-group counts, and ground-truth proximity via `Δd2`.
- Use DBSCAN-based solution-group counting for multi-solution analysis.

### Stage 5: Project adaptation

- Reconnect the reproduced inverse-design engine to the backend service flow.
- Update candidate ranking, artifact generation, and timeline text to reflect the reproduced algorithm.
- Only after this stage is stable, continue with broader problem adaptation and frontend debugging.

## Current Status

- The repository now supports:
  paper CSV loading, JSD calculation, DBSCAN solution-group counting, synthetic and paper dataset modes, and a paper-style cGAN training loop.
- Remaining work:
  improve physical fidelity toward the transmissive Fabry-Pérot setup, integrate reproduced metrics into service artifacts, and refresh user-facing documentation and API behavior.
