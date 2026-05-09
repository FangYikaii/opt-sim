# AI Assisted Microstructure Design Technical Proposal

## 1. Project Goal

Build a pure-code AI assisted microstructure design system for structural color films. The system receives target visual requirements, including 100 x 100 multi-view color images and text constraints, then produces a manufacturable 160k x 320k grayscale relief map with height range 0-7 um. The design must pass fast forward simulation, satisfy optical continuity and process constraints, and provide an interactive human-in-the-loop workflow.

The first implementation will focus on an explainable MVP:

- Use thin-film interference and transfer matrix methods as the forward physical model.
- Use a small inverse design model and candidate search to map target color to manufacturable film or relief parameters.
- Use an AI Agent to orchestrate requirement parsing, design generation, simulation, validation, and export.
- Provide a Vue frontend with a Claude Code inspired workspace UI.
- Use Python for backend services, physics computation, AI training/inference, and export.

The updated scope now includes a second explicit mode from `prd/3414685.3417802.pdf`:

- `structural-color`: the existing thin-film/cGAN route for Ag-SiO2-Ag and manufacturing-oriented color candidates.
- `neural-holography`: a phase-only SLM route for CGH baselines, camera-in-the-loop calibration, and HoloNet-style real-time 1080p inference planning.

## 2. Input Requirements From Existing Materials

The `/需求` assets describe three core tasks:

- Multi-view completion: Given N discrete viewing-angle images and designer text, generate a complete 100 x 100 image set across all viewing angles while satisfying optical realizability and inter-view continuity.
- Inverse microstructure design: Given the 100 x 100 view images and manufacturing constraints, generate a surface contour grayscale map with 160k x 320k resolution and 0-7 um height range. The output should reproduce the target effect after forward simulation and actual processing, with compute time under 0.5 h.
- AI microstructure design Agent: Integrate design and simulation into an Agent that accepts zero-view images and requirement descriptions, generates multi-view simulation images, asks for confirmation, then generates the final grayscale lithography map.

The process image indicates the physical manufacturing context:

- PET substrate and UV-curable resin.
- Grayscale lithography to create a microstructured master.
- Embossing or imprinting a microstructure relief film.
- Target phone-scale film around 8 x 18 cm.
- Grayscale exposure precision around 0.25-0.5 um.
- Surface relief height range around 0-7 um.

## 3. Reference Implementation Direction

### 3.1 Paper Reproduction Scope

Reference paper: `需求/10.1515_nanoph-2022-0095.pdf`, "Inverse design of structural color: finding multiple solutions via conditional generative adversarial networks".

Key ideas to reproduce or adapt:

- Structural color inverse design has a one-to-many nature: one target color may correspond to multiple microstructure or film configurations.
- The paper uses a Fabry-Perot cavity color filter, with Ag-SiO2-Ag thickness parameters as a compact design vector.
- The paper generates forward data using transfer matrix methods, converts spectra to color spaces, and trains an inverse model to find multiple design groups.
- Multiple candidate solutions are useful because manufacturing constraints can reject optically valid but fragile designs.
- Candidate evaluation should include both optical error, such as CIEDE2000 color difference, and fabrication robustness, such as sensitivity to thickness error.

Current implementation priority:

- Start with a full paper reproduction, including the cGAN workflow, before adapting the broader project problem.
- Build and validate the thin-film forward simulator as the physics basis of that reproduction.
- Support dense sampling, nearest-neighbor retrieval, and lightweight regressors as debugging baselines, but not as the main milestone.
- Preserve multiple solutions by returning ranked candidate sets instead of a single prediction.
- After the paper workflow is stable, adapt it to the project-specific inverse-design and agent workflow.

### 3.2 Optics Theory Reference

Reference book: `需求/Principles of Optics 60th Anniversary Edition by Max Born, Emil Wolf (z-lib.org).pdf`.

Relevant section from the provided screenshots and local text extraction:

- Homogeneous dielectric film characteristic matrix.
- Reflection and transmission coefficients for stratified media.
- Transfer matrix multiplication for a stack of films.

Implementation implication:

- The backend should expose a transfer matrix method (TMM) implementation for TE/TM and unpolarized light.
- For MVP, normal incidence can be the default because it is faster, easier to validate, and matches the referenced paper's basic thin-film setup.
- Oblique incidence and multi-view response should be added as the second stage to support angle-dependent effects.

### 3.3 Neural Holography Reference

Reference paper: `prd/3414685.3417802.pdf`, "Neural Holography with Camera-in-the-loop Training".

Key ideas to integrate:

- Computer-generated holography has a runtime versus image-quality tradeoff: direct methods are fast, iterative methods can be higher quality, and HoloNet targets real-time quality after training.
- Simulation-only wave propagation can fail on real hardware because of model mismatch.
- Camera-in-the-loop optimization directly compares target images with physically captured holographic replay.
- A differentiable calibrated proxy should model source intensity variation, SLM phase nonlinearity, and optical aberrations.
- Phase-only SLM outputs are a different artifact class from grayscale relief height maps.

Implementation implication:

- Keep structural-color and neural-holography as separate design modes in the API and UI.
- Add planning routes for simulation-only CGH, CITL optimization, and HoloNet-style deployment before implementing full optical training.
- Gate phase-map export on calibration readiness and captured replay metrics such as PSNR/SSIM.

## 4. System Architecture

```text
Vue Frontend
  |
  | REST/WebSocket
  v
Python API Backend
  |
  +-- Agent Orchestrator
  |     +-- Requirement parser
  |     +-- Design planner
  |     +-- Simulation runner
  |     +-- Human confirmation manager
  |     +-- Export coordinator
  |
  +-- Physics Engine
  |     +-- Thin-film TMM simulator
  |     +-- Spectrum to XYZ/Lab/sRGB conversion
  |     +-- Multi-angle renderer
  |     +-- Process constraint checker
  |     +-- CGH wave propagation model
  |     +-- SLM phase-map simulator
  |     +-- CITL calibration proxy
  |
  +-- Inverse Design Engine
  |     +-- Sampling dataset generator
  |     +-- Candidate retrieval baseline
  |     +-- MLP inverse model
  |     +-- Multi-solution clustering/ranking
  |
  +-- Export Engine
        +-- 100 x 100 design preview export
        +-- tiled high-resolution grayscale map export
        +-- phase-map and calibration-manifest export
        +-- metadata/report export
```

Recommended Python stack for MVP:

- API: FastAPI.
- Numerics: NumPy, SciPy.
- ML: PyTorch or scikit-learn. PyTorch is preferred if cGAN is planned later.
- Color science: `colour-science` if available; otherwise implement a small CIE 1931 + D65 conversion module.
- Image IO and high-resolution tiling: Pillow, tifffile, zarr or chunked NumPy memmap.
- Data validation: Pydantic.

Recommended frontend stack:

- Vue 3 + Vite + TypeScript.
- Pinia for app/session state if needed.
- Canvas/WebGL for previews when the design image count becomes large.
- WebSocket or Server-Sent Events for Agent progress streaming.

## 5. Physics Model

### 5.1 Design Vector

Start with a compact design vector:

```text
D = [material_stack_id, d1, d2, ..., dn, relief_height, local_fill_or_pitch_optional]
```

MVP can use:

```text
D = [d_film]
```

or for paper-style reproduction:

```text
D = [d_Ag_bottom, d_SiO2, d_Ag_top]
```

Project-specific relief mapping should use:

```text
height_nm_or_um -> local optical path difference -> target color / angle response
```

### 5.2 Forward Simulation

For a multilayer film at wavelength lambda and incidence angle theta:

1. Compute complex refractive indices for each material.
2. Compute per-layer phase thickness.
3. Build the characteristic matrix for each layer.
4. Multiply matrices across the stack.
5. Compute reflection/transmission amplitudes and intensities.
6. Convert spectral reflectance/transmittance to CIE XYZ, Lab, and sRGB under selected illuminant.

MVP assumptions:

- Wavelength range: 380-780 nm.
- Sampling interval: 5 nm initially; 1-2 nm for validation.
- Incident medium: air.
- Substrate: PET or quartz depending on selected mode.
- Normal incidence initially.
- Unpolarized color can average TE and TM for non-normal incidence.

### 5.3 Multi-View Extension

Multi-view support adds angle sampling:

```text
view_id -> (theta, phi, observer/illumination condition)
```

For thin-film interference, color shift mainly depends on incidence/view angle through optical path length. The first multi-view version should model polar angle theta. Azimuth phi can remain unused until anisotropic or periodic microstructures are introduced.

### 5.4 Process Constraints

Core manufacturing constraints:

- Height range: 0-7 um.
- Grayscale precision: 0.25-0.5 um.
- Minimum feature slope and continuity constraints.
- Film size: 8 x 18 cm target, final grayscale map 160k x 320k.
- Tiling is mandatory; the full image should not be held as a dense in-memory RGB array if avoidable.

Recommended checks:

- Height clipping and quantization.
- Local gradient limit.
- Smoothness penalty for adjacent pixels.
- Connected-region cleanup for isolated spikes.
- Sensitivity check: recompute color after plus/minus process error.

## 6. AI Inverse Design

### 6.1 Why AI Is Used

The thin-film forward model is fast enough and does not need AI acceleration. AI should be used for inverse design and workflow assistance:

- Convert target color or target multi-view behavior into candidate physical parameters.
- Approximate a many-to-one inverse mapping without requiring a full optimizer for every pixel.
- Rank multiple solutions by optical error, manufacturability, robustness, and continuity.
- Assist users through an Agent that interprets design intent and explains tradeoffs.

### 6.2 MVP Inverse Pipeline

```text
target color / target Lab
  -> query sampled library for candidate parameters
  -> optional MLP predicts seed parameters
  -> local refinement using forward simulator
  -> cluster equivalent solutions
  -> rank by score
  -> return top K manufacturable candidates
```

Recommended score:

```text
score = w_color * deltaE
      + w_process * process_penalty
      + w_sensitivity * deltaE_under_error
      + w_continuity * neighbor_discontinuity
```

### 6.3 Training Data

Generate synthetic data by sampling the design space:

- Sample layer thicknesses or relief heights within manufacturing bounds.
- Simulate spectra and convert to Lab/sRGB.
- Store design vector, spectra summary, Lab, sRGB, angle metadata, and process metrics.

Initial dataset sizes:

- Smoke dataset: 1k-5k samples.
- MVP dataset: 50k samples, matching the referenced paper scale.
- Multi-angle dataset: 50k samples x selected angles if compute permits.

### 6.4 Model Choices

Baseline:

- KD-tree or approximate nearest-neighbor retrieval in Lab space.
- Small MLP from Lab to normalized design parameters.
- Optional MLP from design parameters to Lab as a learned evaluator, mainly for fast candidate filtering.

Later:

- Mixture Density Network for explicit multi-modal outputs.
- cGAN based on the referenced paper if diverse solution groups are needed beyond retrieval.
- Conditional diffusion or flow model if the design vector becomes high-dimensional.

## 7. AI Agent Workflow

### 7.1 Agent Responsibilities

The Agent is not a replacement for the physics engine. It is an orchestrator:

- Parse user goals, target images, view count, material/process constraints, and output format.
- Ask focused clarification questions only when requirements are physically ambiguous.
- Plan a design run with selected simulator, inverse method, angle grid, and quality thresholds.
- Launch dataset generation, inverse prediction, simulation, ranking, and export jobs.
- Stream progress and intermediate results to the UI.
- Explain failures, tradeoffs, and candidate selection.
- Request human confirmation before expensive export or final grayscale generation.

### 7.2 Agent State Machine

```text
Draft requirements
  -> Validate physical feasibility
  -> Generate/complete multi-view targets
  -> Run inverse design candidates
  -> Run forward simulation
  -> Present comparison and risk report
  -> Human confirms or revises
  -> Generate high-resolution grayscale map
  -> Export package
```

### 7.3 Agent Tools

Internal tool-like functions:

- `parse_requirement(text, images)`
- `simulate_stack(design, angles, illuminant)`
- `inverse_design(target_lab, constraints, top_k)`
- `complete_views(seed_views, description, view_grid)`
- `rank_candidates(candidates, target, constraints)`
- `export_grayscale_map(design_field, output_spec)`
- `plan_holography_run(target_images, slm_spec, calibration_state)`
- `optimize_phase_citl(target_image, calibrated_proxy, capture_config)`
- `export_phase_map(phase_frames, calibration_manifest)`
- `write_report(run_id)`

### 7.4 Design Mode Branching

The Agent must branch early by `designMode`:

- `structural-color` uses target color/image, TMM simulation, inverse-design candidates, DeltaE, manufacturability, and grayscale relief export planning.
- `neural-holography` uses target image anchors, CGH algorithm selection, SLM phase-map outputs, CITL calibration readiness, PSNR/SSIM replay gates, and phase-map export planning.

This branching prevents the system from mixing thin-film thickness constraints with SLM phase-map requirements.

## 8. Human-Computer Interaction

The interface should feel like a Claude Code style scientific design workspace:

- Left panel: project files, runs, uploaded target images, exported artifacts.
- Center panel: chat/Agent timeline with tool calls, progress, warnings, and approval cards.
- Right panel: live simulation inspector with color patches, spectra, angle previews, candidate tables, and constraint checks.
- Bottom/side terminal-like drawer: raw logs, generated parameters, job status.

Key interaction flows:

- Upload target images and type design requirements.
- Agent extracts requirements and displays a structured checklist.
- User confirms constraints and starts a design run.
- Candidate solutions are shown as cards with DeltaE, height range, sensitivity, and process risk.
- User chooses a candidate or lets the Agent select the best robust solution.
- User confirms high-resolution export.

## 9. Output Artifacts

Each design run should export:

- `preview_views/`: simulated multi-angle preview images.
- `height_map_preview.png`: downsampled relief preview.
- `height_map_full.tif` or tiled output folder for 160k x 320k grayscale data.
- `design_params.json`: materials, layer thicknesses, angle grid, scoring weights.
- `simulation_report.md`: target comparison, DeltaE metrics, process constraints, selected candidate rationale.
- `run_manifest.json`: reproducibility metadata and software versions.

Neural holography runs should export a different package:

- `phase_maps/`: wrapped phase patterns for the selected SLM/channel setup.
- `calibration_manifest.json`: SLM response, source intensity, aberration proxy, camera setup, and calibration version.
- `replay_report.md`: target versus simulated/captured PSNR, SSIM, runtime, and known model mismatch.
- `holonet_checkpoint/`: optional trained real-time model artifacts when HoloNet-style inference is available.

## 10. MVP Acceptance Criteria

Physics:

- Forward TMM computes spectra for a simple single-layer or three-layer film.
- Spectrum-to-color conversion produces plausible sRGB/Lab values.
- A small validation script reproduces color variation as thickness changes.

Inverse design:

- Given a target sRGB/Lab color, return top K candidate designs.
- Each candidate includes predicted color, DeltaE, process validity, and sensitivity.
- At least one baseline design can be found in under 1 second for a single target color after dataset generation.

Agent:

- User can submit a text requirement and target color/image.
- Agent produces a structured plan, runs inverse design, runs simulation, and asks for confirmation before export.
- Agent streams progress to the frontend.

Frontend:

- Vue app provides Claude Code style three-panel workspace.
- User can inspect target vs simulated color, candidate parameters, and constraint status.
- User can trigger export and download artifacts.
- User can choose `structural-color` or `neural-holography` mode and see the active requirement source, output kind, calibration mode, and runtime target.

Export:

- Generate a downsampled grayscale preview.
- Generate a tiled high-resolution output plan. Full 160k x 320k export can be deferred to a chunked writer if storage/time is large.
- For neural holography, generate a phase-map export plan only after CITL calibration and replay-quality gates are reviewed.

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---:|---|
| Thin-film model may not reproduce all target 3D relief effects | High | Start with model-valid design cases; mark out-of-scope anisotropic/diffractive effects; add RCWA/FDTD later only if needed |
| 160k x 320k export is extremely large | High | Use tiled/chunked output, lazy generation, preview-first workflow, and explicit user confirmation |
| One-to-many inverse mapping collapses to one solution | Medium | Return retrieval candidates, cluster solutions, then add MDN/cGAN if needed |
| Material refractive index data missing or inaccurate | Medium | Version material datasets and allow user-provided n/k tables |
| Multi-view target completion can generate physically impossible colors | Medium | Validate every generated view through the forward simulator and report impossible regions |
| Runtime exceeds 0.5 h for full map | Medium | Cache per-color candidates, quantize color/height palettes, tile export, and avoid per-pixel optimization |

## 12. Phased Roadmap

### Phase 0: Documentation and Design Basis

- Finalize technical proposal and task breakdown.
- Create initial UI design rules.
- Record architectural decisions.

### Phase 1: Physics Core

- Implement TMM forward simulator.
- Implement color conversion and material data loading.
- Add validation notebooks/scripts for thickness sweeps.

### Phase 2: Inverse Design Baseline

- Generate synthetic dataset.
- Implement candidate retrieval and MLP inverse model.
- Implement multi-solution ranking and robustness scoring.

### Phase 3: Agent and API

- Build FastAPI service.
- Add run/session model.
- Add Agent orchestration and progress events.

### Phase 4: Vue Workspace

- Build Claude Code style UI shell.
- Add upload, chat, run timeline, candidate inspector, and preview panels.

### Phase 5: Export and Scaling

- Add grayscale preview and tiled high-resolution export.
- Add 100 x 100 multi-view batch processing.
- Add caching and runtime profiling.

### Phase 6: Advanced Multi-Solution and Multi-View

- Add cGAN/MDN if baseline diversity is insufficient.
- Add oblique incidence TE/TM support.
- Add stricter continuity and manufacturability optimization.
