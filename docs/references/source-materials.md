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

## Optics Theory Reference

File: `需求/Principles of Optics 60th Anniversary Edition by Max Born, Emil Wolf (z-lib.org).pdf`

How it is used:

- Provides the transfer matrix theory for wave propagation in stratified media.
- Supports the homogeneous dielectric film characteristic matrix.
- Supports reflection and transmission coefficient derivations.

MVP interpretation:

- Implement a thin-film transfer matrix method as the backend physics source of truth.
- Start with normal incidence and add TE/TM oblique incidence later.
