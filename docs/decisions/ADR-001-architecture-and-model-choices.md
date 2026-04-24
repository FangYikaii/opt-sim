# ADR-001: Use Thin-Film Physics, Python Backend, Vue Frontend, and Agent Orchestration

## Status

Accepted

## Date

2026-04-23

## Context

The project aims to build an AI assisted microstructure design system for structural color films. The initial requirements specify:

- Use thin-film interference physics to realize structural color.
- Use AI mainly for inverse design rather than accelerating expensive simulation.
- Implement the project in pure code without complex third-party commercial software.
- Include an AI Agent.
- Include a human-computer interaction page.
- Use Python for backend and Vue for frontend.
- Reference `需求/10.1515_nanoph-2022-0095.pdf` and Born & Wolf optics theory, especially transfer matrix formulations for stratified media.

The referenced Nanophotonics paper demonstrates that structural color inverse design is one-to-many: many film thickness combinations can produce the same or similar target color. The paper uses a cGAN to preserve multiple solution groups for an Ag-SiO2-Ag Fabry-Perot color filter. The project requirement also includes manufacturing constraints, high-resolution grayscale export, and user confirmation loops, so a pure black-box inverse model is not sufficient.

## Decision

Use the following architecture:

- Physics model: thin-film interference using transfer matrix method as the primary forward simulator.
- AI role: inverse design, candidate generation, candidate ranking, and Agent assistance.
- Initial inverse method: dataset sampling + candidate retrieval + small MLP. Add cGAN/MDN later only if the baseline cannot provide enough diverse solutions.
- Backend: Python with FastAPI, NumPy/SciPy, PyTorch or scikit-learn, Pydantic, and chunked image/export tooling.
- Frontend: Vue 3 + Vite + TypeScript.
- Agent: rule-first orchestrator that calls physics, inverse design, simulation, validation, and export tools; it should stream progress and request human confirmation before expensive or irreversible operations.
- UI direction: Claude Code inspired scientific design workspace using `awesome-design-md` cues from Claude and developer-tool interfaces.

## Alternatives Considered

### Full Electromagnetic Simulation First

Examples include FDTD, FEM, RCWA, or commercial solvers.

- Pros: Can model diffraction, periodic microstructures, anisotropy, and complex relief effects.
- Cons: Slower, more complex, more dependencies, and conflicts with the requirement for pure-code fast implementation.
- Rejected for MVP: Thin-film interference is explicitly chosen and is enough for a fast explainable baseline.

### cGAN First

- Pros: Closely follows the paper and directly targets one-to-many inverse design.
- Cons: More training complexity, harder debugging, and unnecessary if retrieval from a dense analytic dataset already returns multiple viable solutions.
- Deferred: Add after baseline evaluation if solution diversity is insufficient.

### Frontend in React

- Pros: Large ecosystem and strong agent familiarity.
- Cons: User explicitly selected Vue.
- Rejected: Use Vue 3.

### Backend in Node.js

- Pros: Same language as frontend and easy WebSocket integration.
- Cons: Scientific computing, optics, and ML ecosystem are stronger in Python.
- Rejected: Use Python.

### No Agent, Only Forms

- Pros: Simpler implementation.
- Cons: User explicitly requires an AI Agent, and the workflow benefits from requirement parsing, staged confirmation, and explanation of physical tradeoffs.
- Rejected: Include an Agent orchestrator.

## Consequences

- The first deliverable can be useful without training a large model.
- The physics core becomes the source of truth for both AI evaluation and UI previews.
- Candidate ranking can incorporate manufacturing robustness, not just color error.
- The system can grow from single-color MVP to 100 x 100 multi-view batch design.
- High-resolution export must be designed carefully with chunked/tiled output because 160k x 320k is too large for naive in-memory processing.
- Future advanced models must be judged against retrieval/MLP baselines before being made default.
