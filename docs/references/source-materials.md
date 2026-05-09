# Source Materials

This document records the source materials currently used to define the project scope and technical direction.

## Requirement Assets

Location: `需求/`

- `需求/问题描述.png`: Defines the three major problem areas: multi-view completion, inverse microstructure design, and AI microstructure design Agent.
- `需求/工艺.png`: Describes the manufacturing context: PET substrate, UV-curable resin, grayscale lithography, embossing/imprinting, phone-scale film, 0.25-0.5 um grayscale precision, and 0-7 um relief height.

## Paper Reference

File: `需求/10.1515_nanoph-2022-0095.pdf`

Title: "Inverse design of structural color: finding multiple solutions via conditional generative adversarial networks"

How it is used:

- Provides the paper reproduction reference.
- Establishes the one-to-many nature of structural color inverse design.
- Demonstrates a Fabry-Perot cavity color filter with Ag-SiO2-Ag layer thicknesses.
- Shows a workflow using transfer matrix simulation, color conversion, inverse design, multi-solution grouping, and fabrication-oriented candidate selection.

Current execution interpretation:

- Reproduce the full paper workflow first, not just the data-generation idea.
- Include training, testing-set inference, distribution analysis, and multi-solution grouping.
- Use retrieval and simplified baselines only as supporting smoke-test tools, not as the primary milestone.
- Preserve multiple candidate solutions because manufacturability matters.

## Neural Holography Reference

File: `prd/3414685.3417802.pdf`

Title: "Neural Holography with Camera-in-the-loop Training"

How it is used:

- Adds a second product mode for computer-generated holography rather than structural-color film design.
- Establishes phase-only SLM hologram generation as a required output class.
- Introduces camera-in-the-loop optimization as the high-fidelity route for matching physical replay quality.
- Introduces a calibrated differentiable propagation proxy that models source intensity, SLM phase nonlinearity, and optical aberrations.
- Introduces HoloNet-style neural inference as the real-time 1080p route after calibration.

Current execution interpretation:

- Do not replace the structural-color/cGAN reproduction route; expose a separate `neural-holography` design mode.
- Treat GS, Wirtinger Holography, SGD, CITL optimization, and HoloNet as distinct algorithm routes with different quality/runtime tradeoffs.
- Gate any final phase-map export on explicit calibration and replay-quality evidence.
- Use PSNR/SSIM and captured replay quality for future holography metrics, while the current implementation slice records the planning and calibration contract.

## Optics Theory Reference

File: `需求/Principles of Optics 60th Anniversary Edition by Max Born, Emil Wolf (z-lib.org).pdf`

How it is used:

- Provides the transfer matrix theory for wave propagation in stratified media.
- Supports the homogeneous dielectric film characteristic matrix.
- Supports reflection and transmission coefficient derivations.

MVP interpretation:

- Implement a thin-film transfer matrix method as the backend physics source of truth.
- Start with normal incidence and add TE/TM oblique incidence later.
