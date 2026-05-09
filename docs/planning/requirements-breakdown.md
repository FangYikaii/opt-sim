# Requirements Breakdown: AI Assisted Microstructure Design

## Overview

This document breaks the project into implementable, verifiable tasks. The recommended implementation order is foundation first, then vertical slices that connect physics, inverse design, Agent orchestration, and UI.

The May 2026 requirement update adds `prd/3414685.3417802.pdf` as a second product branch. The project must now preserve the existing structural-color implementation while adding a `neural-holography` design mode for phase-only SLM, camera-in-the-loop calibration, and HoloNet-style real-time inference planning.

## Dependency Graph

```text
Project docs and contracts
  -> Physics engine
      -> Synthetic dataset generation
          -> Inverse design baseline
              -> Candidate ranking and validation
                  -> Agent workflow
                      -> Vue frontend integration
                      -> Export and scaling

Neural holography branch
  -> Design-mode contract
      -> CGH/CITL planning workspace
          -> Wave propagation and phase optimization
              -> CITL calibration
                  -> HoloNet-style inference
                      -> Phase-map export
```

## Phase 0: Planning Foundation

### Task 0.1: Create Technical Baseline Documents

**Description:** Document the physical model, AI role, Agent workflow, architecture, and implementation phases.

**Acceptance criteria:**

- [ ] Technical proposal describes the thin-film model, inverse design pipeline, Agent responsibilities, and frontend/backend stack.
- [ ] Architecture decisions are recorded in ADR format.
- [ ] UI visual direction is captured in `DESIGN.md`.

**Verification:**

- [ ] Review generated docs for consistency with `/需求` and the two reference PDFs.

**Dependencies:** None

**Estimated scope:** Small

### Task 0.2: Define Data Contracts

**Description:** Define JSON schemas or Pydantic models for target color, view image, material stack, design vector, candidate solution, simulation result, and export manifest.

**Acceptance criteria:**

- [ ] Each major object has required fields, units, and validation rules.
- [ ] Contracts include physical units such as nm, um, degrees, and color space names.
- [ ] Contracts support both single-color MVP and future 100 x 100 multi-view batches.

**Verification:**

- [ ] Unit tests validate accepted and rejected example payloads.

**Dependencies:** Task 0.1

**Estimated scope:** Medium

## Phase 1: Physics Core

### Task 1.1: Implement Material Data Loader

**Description:** Load refractive index data for air, PET/quartz, SiO2, Ag, Ti, and user-defined materials.

**Acceptance criteria:**

- [ ] Material n/k values can be queried by wavelength.
- [ ] Interpolation is deterministic and handles wavelength range errors.
- [ ] Built-in material metadata includes source and valid wavelength range.

**Verification:**

- [ ] Unit tests cover interpolation and out-of-range validation.

**Dependencies:** Task 0.2

**Estimated scope:** Medium

### Task 1.2: Implement Thin-Film Transfer Matrix Simulator

**Description:** Implement a Python TMM simulator for multilayer films.

**Acceptance criteria:**

- [ ] Supports normal incidence.
- [ ] Computes reflectance and/or transmittance spectra over 380-780 nm.
- [ ] Supports at least single-layer dielectric and paper-style Ag-SiO2-Ag stack.
- [ ] Exposes stable API for batch simulation.

**Verification:**

- [ ] Unit test energy behavior for non-absorbing simple films.
- [ ] Regression test validates thickness sweep produces periodic color/spectrum changes.

**Dependencies:** Task 1.1

**Estimated scope:** Medium

### Task 1.3: Implement Spectrum-To-Color Conversion

**Description:** Convert simulated spectra to CIE XYZ, Lab, and sRGB.

**Acceptance criteria:**

- [ ] Supports D65 illuminant by default.
- [ ] Produces clipped and unclipped sRGB.
- [ ] Computes CIEDE2000 or a documented fallback DeltaE metric.

**Verification:**

- [ ] Unit tests cover known white/flat spectrum behavior and DeltaE sanity checks.

**Dependencies:** Task 1.2

**Estimated scope:** Medium

### Checkpoint: Physics

- [ ] A script can sweep film thickness and save a plot/table of simulated colors.
- [ ] Physics tests pass.
- [ ] The API shape is stable enough for dataset generation.

## Phase 2: Inverse Design Baseline

### Task 2.1: Generate Synthetic Dataset

**Description:** Sample the design space and simulate color responses for training/retrieval.

**Acceptance criteria:**

- [ ] Supports configurable sampling ranges and sample counts.
- [ ] Stores design vectors, spectra summary, Lab, sRGB, and process metadata.
- [ ] Can create a smoke dataset and a 50k-sample MVP dataset.

**Verification:**

- [ ] Dataset generation command completes for 1k samples.
- [ ] Dataset summary reports coverage and invalid samples.

**Dependencies:** Phase 1 checkpoint

**Estimated scope:** Medium

### Task 2.2: Implement Candidate Retrieval

**Description:** Return top K design candidates for a target Lab/sRGB by searching the synthetic dataset.

**Acceptance criteria:**

- [ ] Supports target input in sRGB and Lab.
- [ ] Returns top K candidates with DeltaE and design parameters.
- [ ] Supports constraints such as min/max thickness and height quantization.

**Verification:**

- [ ] Query known dataset samples and confirm nearest candidates include the original or near-original design.

**Dependencies:** Task 2.1

**Estimated scope:** Small

### Task 2.3: Train Small MLP Inverse Model

**Description:** Train a small MLP that predicts candidate design parameters from target color.

**Acceptance criteria:**

- [ ] Training script loads generated dataset.
- [ ] Model predicts normalized design vectors.
- [ ] Evaluation reports DeltaE after forward simulation, not just parameter MSE.

**Verification:**

- [ ] Training works on smoke dataset.
- [ ] Evaluation report compares MLP-only vs retrieval baseline.

**Dependencies:** Task 2.1

**Estimated scope:** Medium

### Task 2.4: Multi-Solution Ranking and Robustness Scoring

**Description:** Cluster and rank candidate solutions using optical error, manufacturability, and process sensitivity.

**Acceptance criteria:**

- [ ] Returns multiple solution groups when available.
- [ ] Computes sensitivity under plus/minus thickness or height error.
- [ ] Produces a human-readable rationale for the selected candidate.

**Verification:**

- [ ] Tests confirm robust candidate can outrank fragile low-DeltaE candidate when configured.

**Dependencies:** Task 2.2

**Estimated scope:** Medium

### Checkpoint: Inverse Design

- [ ] A target sRGB/Lab query returns ranked candidate designs.
- [ ] Each candidate can be forward-simulated and compared to target color.
- [ ] Single target query completes under 1 second after dataset/model load.

## Phase 3: Backend API and Agent

### Task 3.1: Scaffold FastAPI Backend

**Description:** Create backend application with endpoints for health, simulation, inverse design, and run management.

**Acceptance criteria:**

- [ ] API has typed request/response models.
- [ ] Simulation endpoint accepts a design vector and returns color/spectrum result.
- [ ] Inverse endpoint accepts target color and constraints and returns candidates.

**Verification:**

- [ ] API tests cover health, simulation, and inverse endpoints.

**Dependencies:** Phase 2 checkpoint

**Estimated scope:** Medium

### Task 3.2: Add Agent Orchestrator

**Description:** Implement a rule-first Agent workflow that turns user requirements into design runs.

**Acceptance criteria:**

- [ ] Agent creates a structured requirement summary.
- [ ] Agent can trigger inverse design and forward simulation tools.
- [ ] Agent asks for confirmation before expensive export.
- [ ] Agent records timeline events for the UI.

**Verification:**

- [ ] End-to-end test runs a scripted user request and produces a run timeline.

**Dependencies:** Task 3.1

**Estimated scope:** Medium

### Task 3.3: Add Progress Streaming

**Description:** Stream Agent and job progress to the frontend.

**Acceptance criteria:**

- [ ] Supports WebSocket or Server-Sent Events.
- [ ] Streams tool start, progress, warning, result, and approval-needed events.
- [ ] Reconnect or polling fallback can retrieve latest run state.

**Verification:**

- [ ] Integration test receives ordered progress events for a sample run.

**Dependencies:** Task 3.2

**Estimated scope:** Medium

### Checkpoint: Backend

- [ ] API can serve a complete single-target design run.
- [ ] Agent timeline includes requirement parsing, inverse design, simulation, and approval event.
- [ ] API can serve both `structural-color` and `neural-holography` design modes without breaking old requests.
- [ ] Backend tests pass.

## Phase 4: Vue Human-Computer Interface

### Task 4.1: Scaffold Vue Workspace

**Description:** Build Vue 3 + Vite + TypeScript app shell using the project `DESIGN.md`.

**Acceptance criteria:**

- [ ] Three-panel Claude Code inspired layout exists.
- [ ] Responsive behavior works on desktop and tablet-sized screens.
- [ ] Core visual tokens are implemented as CSS variables.

**Verification:**

- [ ] App builds successfully.
- [ ] Manual browser check confirms layout and visual direction.

**Dependencies:** Task 3.1 can be mocked initially

**Estimated scope:** Medium

### Task 4.2: Build Agent Chat and Run Timeline

**Description:** Add chat input, Agent messages, tool-call cards, warnings, and approval cards.

**Acceptance criteria:**

- [ ] User can submit a text requirement.
- [ ] Timeline renders streamed backend events.
- [ ] Approval card can confirm or revise a run.

**Verification:**

- [ ] Mocked timeline story covers all event types.
- [ ] Integration check with backend streaming endpoint.

**Dependencies:** Task 4.1, Task 3.3

**Estimated scope:** Medium

### Task 4.3: Build Simulation Inspector

**Description:** Show target vs simulated color, spectra, candidate table, and process constraint checks.

**Acceptance criteria:**

- [ ] Candidate list displays design parameters, DeltaE, sensitivity, and process status.
- [ ] Selecting a candidate updates preview and explanation.
- [ ] Constraint failures are visible and actionable.

**Verification:**

- [ ] UI test or manual check with sample backend response.

**Dependencies:** Task 4.1, Phase 2 checkpoint

**Estimated scope:** Medium

### Checkpoint: UI

- [ ] User can run a single-color design flow from the UI.
- [ ] User can select structural-color or neural-holography mode before submitting.
- [ ] Workspace shows requirement source, output kind, calibration mode, and runtime target for the active mode.
- [ ] Candidate selection and approval are understandable.
- [ ] UI remains responsive during backend progress streaming.

## Phase H: Neural Holography / CITL Extension

### Task H0: Fuse New PRD Into Product Contracts

**Description:** Add explicit design-mode branching for `prd/3414685.3417802.pdf` while preserving the existing thin-film route.

**Acceptance criteria:**

- [ ] `DesignRequest` accepts `designMode` with default `structural-color`.
- [ ] `neural-holography` responses include source paper, output kind, calibration mode, and runtime target.
- [ ] Runtime artifacts persist the active mode and source paper.
- [ ] Vue UI exposes mode selection and displays mode context.

**Verification:**

- [ ] Backend API test creates a neural holography run and validates CITL fields.
- [ ] Frontend build passes with regenerated OpenAPI types.

**Dependencies:** Phase 3 API and Phase 4 workspace shell

**Estimated scope:** Small

### Task H1: Implement CGH Simulation Core

**Description:** Add phase-only SLM simulation and baseline CGH algorithms.

**Acceptance criteria:**

- [ ] Supports angular spectrum or Fresnel propagation for small target images.
- [ ] Implements GS, WH, and SGD baseline route interfaces.
- [ ] Reports PSNR and SSIM against target images.
- [ ] Stores phase-map previews and replay metadata.

**Verification:**

- [ ] Unit tests cover propagation shape, energy sanity, and deterministic replay for a smoke image.
- [ ] Regression test compares baseline algorithm metrics on a fixed target.

**Dependencies:** Task H0

**Estimated scope:** Large, split by algorithm

### Task H2: Add CITL Calibration Data Model

**Description:** Model camera-in-the-loop calibration datasets and calibrated propagation proxy metadata.

**Acceptance criteria:**

- [ ] Calibration record includes source intensity, per-pixel phase nonlinearity, aberration proxy, camera setup, and version.
- [ ] Workspace flags missing calibration data before phase-map export.
- [ ] Artifact manifest records calibration version used for a run.

**Verification:**

- [ ] API tests validate missing/ready calibration states.
- [ ] Serialization test round-trips calibration manifest.

**Dependencies:** Task H1 can be mocked initially

**Estimated scope:** Medium

### Task H3: Add HoloNet-Style Real-Time Route

**Description:** Add a deployable route for a neural phase generator trained against the calibrated proxy.

**Acceptance criteria:**

- [ ] Workspace distinguishes iterative CITL quality optimization from real-time neural inference.
- [ ] Model metadata records target resolution, checkpoint, runtime, and validation metrics.
- [ ] Deployment remains blocked unless captured holdout quality is available.

**Verification:**

- [ ] API test exposes HoloNet route and block/pass state.
- [ ] Benchmark script reports inference latency on a smoke target.

**Dependencies:** Task H2

**Estimated scope:** Large

### Task H4: Phase-Map Export

**Description:** Export wrapped phase-map frames and calibration manifests.

**Acceptance criteria:**

- [ ] Export output is phase-map based, not grayscale relief height.
- [ ] Export requires explicit approval.
- [ ] Manifest records SLM resolution, wavelength/channel setup, calibration version, and replay metrics.

**Verification:**

- [ ] Export test validates phase-map range and manifest fields.
- [ ] Manual workspace check confirms approval is required.

**Dependencies:** Task H2

**Estimated scope:** Medium

## Phase 5: Export and Scaling

### Task 5.1: Grayscale Preview Export

**Description:** Generate a downsampled grayscale height map preview for selected candidate/design field.

**Acceptance criteria:**

- [ ] Preview image uses the selected height range and quantization.
- [ ] Metadata records height units and mapping.
- [ ] UI can display and download preview.

**Verification:**

- [ ] Export test validates pixel range and metadata.

**Dependencies:** Phase 4 checkpoint

**Estimated scope:** Small

### Task 5.2: Tiled High-Resolution Export

**Description:** Implement chunked export for 160k x 320k grayscale map.

**Acceptance criteria:**

- [ ] Export does not require loading the full map into memory.
- [ ] Output can be tiled TIFF, tiled PNG folder, or another agreed manufacturing format.
- [ ] Export requires explicit user confirmation.

**Verification:**

- [ ] Test with smaller tiled export confirms tile naming, stitching metadata, and memory limits.

**Dependencies:** Task 5.1

**Estimated scope:** Large, split further during implementation

### Task 5.3: 100 x 100 Multi-View Batch Design

**Description:** Extend single-target inverse design to a 100 x 100 target field across multiple viewing angles.

**Acceptance criteria:**

- [ ] Batch pipeline accepts target view stack.
- [ ] Neighbor continuity and quantization constraints are applied.
- [ ] Runtime and memory are reported.

**Verification:**

- [ ] Smoke run on 10 x 10 x few angles.
- [ ] Scaled dry-run estimates full 100 x 100 runtime.

**Dependencies:** Task 5.1 and multi-angle physics extension

**Estimated scope:** Large, split further during implementation

## Phase 6: Advanced Models and Physics

### Task 6.1: Oblique Incidence TE/TM Support

**Description:** Extend the TMM simulator to angle-dependent TE/TM and unpolarized response.

**Acceptance criteria:**

- [ ] Supports theta angle grid.
- [ ] Handles total internal reflection and complex angles safely.
- [ ] Produces angle-dependent color previews.

**Verification:**

- [ ] Regression tests compare normal-incidence behavior with existing simulator.

**Dependencies:** Phase 1 checkpoint

**Estimated scope:** Medium

### Task 6.2: cGAN or MDN Multi-Solution Model

**Description:** Add a model that explicitly represents one-to-many inverse design if retrieval plus MLP is insufficient.

**Acceptance criteria:**

- [ ] Model returns diverse candidate groups.
- [ ] Evaluation measures solution-group count, DeltaE, and coverage.
- [ ] Baseline comparison shows clear benefit before making it default.

**Verification:**

- [ ] Reproduce a paper-inspired three-layer film experiment on generated data.

**Dependencies:** Phase 2 checkpoint

**Estimated scope:** Large, split further during implementation

## Open Questions

- Which optical mode is primary for the product: reflective film, transmissive film, or both?
- What exact material stack should be assumed for the PET/UV resin process?
- What manufacturing file format is required by the grayscale lithography tool?
- Are the 100 x 100 view images per viewing angle, or 100 x 100 angular samples?
- Should the first deliverable optimize for visual proof-of-concept or direct manufacturing readiness?

## Near-Term Recommended Sequence

1. Implement Phase 1 physics core.
2. Validate against simple thickness sweeps and the reference paper's qualitative behavior.
3. Implement retrieval baseline before MLP.
4. Build API around the stable single-color flow.
5. Build Vue workspace with mocked data, then connect live backend.
6. Add batch and export scaling only after single-target flow is credible.
