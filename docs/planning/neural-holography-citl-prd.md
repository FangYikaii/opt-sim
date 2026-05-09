# PRD: Neural Holography Camera-In-The-Loop Extension

## Objective

Fuse the new requirement source `prd/3414685.3417802.pdf`, "Neural Holography with Camera-in-the-loop Training", into the existing Opt-Sim platform without discarding the current structural-color and cGAN reproduction work.

The updated platform must support two explicit design modes:

- `structural-color`: the current Ag-SiO2-Ag thin-film inverse-design workflow.
- `neural-holography`: a computer-generated holography workflow for phase-only SLM outputs, camera-in-the-loop calibration, and HoloNet-style real-time inference planning.

Success means users can create and review a neural holography design run from the same UI/API surface, see the new source paper and calibration requirements, and receive a verifiable route before any expensive phase-map export.

## Requirement Delta

| Area | Previous project baseline | New requirement from `3414685.3417802.pdf` | Required change |
|---|---|---|---|
| Physical output | Grayscale relief or thin-film stack parameters | Phase-only SLM hologram patterns | Add phase-map output mode and metadata |
| Forward model | Thin-film TMM, spectra, color conversion | Wave propagation from SLM to image plane | Add CGH propagation model concepts and planning gates |
| Inverse method | Candidate retrieval, MLP, cGAN multi-solution design | Iterative GS/WH/SGD and HoloNet neural inference | Add holography algorithm route options |
| Calibration | Material data and process sensitivity | Camera-in-the-loop captures of source intensity, phase nonlinearity, aberrations | Add CITL calibration checklist and runtime state |
| Quality metrics | DeltaE, manufacturability, process drift | PSNR, SSIM, captured replay quality, runtime | Add holography metrics in candidates |
| Runtime target | Offline design and high-resolution export | Real-time 1080p inference after proxy training | Add `runtimeTarget` and HoloNet deployment route |

## Scope

### In Scope For Current Slice

- Extend API request contract with `designMode`.
- Keep `structural-color` behavior backwards compatible by default.
- Add `neural-holography` run creation with:
  - source paper reference,
  - output kind,
  - calibration mode,
  - runtime target,
  - CGH/CITL/HoloNet candidate routes,
  - CITL constraint checklist,
  - phase-map export estimate.
- Surface the new fields in the Vue workspace.
- Update OpenAPI and generated frontend types.
- Add regression tests for the new mode.

### Out Of Scope For Current Slice

- Full wave-propagation numerical solver.
- Real camera capture integration.
- HoloNet training.
- Bench PSNR/SSIM evaluation from captured images.
- Production phase-map file generation.

These are planned as follow-up implementation phases below.

## User Stories

- As an optics engineer, I can choose whether a run is a structural-color run or a neural holography run before submitting requirements.
- As a reviewer, I can see that a neural holography run is based on `3414685.3417802.pdf`, not the structural-color cGAN paper.
- As a bench engineer, I can see which calibration data is required before trusting phase-map output.
- As a project owner, I can distinguish simulation-only CGH baselines, CITL optimization, and HoloNet deployment as separate candidate routes.

## Contracts

### Request

`POST /api/agent/design-run`

```json
{
  "requirementText": "Plan a camera-in-the-loop holographic display workflow for a 1080p target.",
  "targetHex": "#6f8fd8",
  "topK": 3,
  "thetaDeg": 0,
  "polarization": "unpolarized",
  "designMode": "neural-holography"
}
```

`designMode` defaults to `structural-color` for backwards compatibility.

### Workspace Fields

`WorkspaceDraft` includes:

- `designMode`
- `referenceSource`
- `outputKind`
- `calibrationMode`
- `runtimeTarget`

These fields must persist to workspace artifacts and be visible in the frontend.

## Implementation Plan

1. Add contract fields and schema defaults.
2. Branch the Agent workflow by `designMode`.
3. Add deterministic neural holography planning candidates and constraints.
4. Update runtime artifact metadata.
5. Add workspace UI fields and home-page design mode selector.
6. Regenerate OpenAPI and frontend API types.
7. Run backend tests and frontend build.

## Follow-Up Phases

### Phase H1: CGH Simulation Core

- Implement angular spectrum or Fresnel propagation.
- Implement GS, WH, and SGD baseline phase optimization.
- Add PSNR/SSIM simulation metrics.
- Verify on small static images before 1080p.

### Phase H2: CITL Calibration

- Define camera capture dataset schema.
- Estimate source intensity variation, SLM phase nonlinearity, and optical aberration proxy parameters.
- Store calibration version and bench conditions per run.
- Verify with captured holdout images.

### Phase H3: HoloNet-Style Inference

- Train a direct neural phase generator against the calibrated differentiable proxy.
- Report runtime at target resolution.
- Gate deployment on captured quality, not simulation-only quality.

### Phase H4: Phase-Map Export

- Export wrapped phase maps, calibration manifests, and replay reports.
- Add explicit approval before export.
- Include wavelength/channel and SLM metadata.

## Acceptance Criteria

- Backend accepts `designMode: "neural-holography"` and returns a complete workspace response.
- Existing requests without `designMode` still use structural-color mode.
- Neural holography workspace response references `3414685.3417802.pdf`.
- Candidate routes include CITL optimization, HoloNet-style inference, and simulation-only baseline.
- Constraints include target image readiness, CITL calibration, SLM phase output, and viewing condition checks.
- Frontend build passes after generated API types are updated.
- Backend regression tests pass in the project `opt_sim` environment.
