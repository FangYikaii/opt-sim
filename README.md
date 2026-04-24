# Opt-Sim

AI assisted microstructure design prototype for structural color films.

The current repository stage is requirements and technical planning. The selected direction is a pure-code Python + Vue system:

- Thin-film interference and transfer matrix methods for fast forward simulation.
- AI inverse design for mapping target colors or multi-view targets to manufacturable microstructure parameters.
- Agent orchestration for requirement parsing, simulation, candidate ranking, human confirmation, and export.
- Claude Code inspired human-computer interaction workspace.

## Documents

- [Documentation index](docs/README.md): organized map of all project documents.
- [Algorithm operations guide](docs/planning/algorithm-operations-guide.md): current algorithm status, GPU/training interpretation, backend/frontend usage, and step-by-step operations.
- [Technical proposal](docs/architecture/technical-proposal.md): system architecture, physics model, AI inverse design, Agent workflow, UI direction, outputs, risks, and roadmap.
- [Requirements breakdown](docs/planning/requirements-breakdown.md): phased implementation tasks with acceptance criteria and verification steps.
- [ADR-001](docs/decisions/ADR-001-architecture-and-model-choices.md): architecture and model choice rationale.
- [DESIGN.md](DESIGN.md): UI design direction for the future Vue frontend.
- [Source materials](docs/references/source-materials.md): requirement assets and reference PDF notes.

## Reference Materials

- `需求/10.1515_nanoph-2022-0095.pdf`: inverse design of structural color via cGAN, used as the paper reproduction and multi-solution inverse-design reference.
- `需求/Principles of Optics 60th Anniversary Edition by Max Born, Emil Wolf (z-lib.org).pdf`: transfer matrix and stratified media theory reference.
- `需求/`: original requirement images and process context.

## Recommended Next Step

Prioritize the reference-paper reproduction track first:

- complete the full paper-style algorithm chain,
- verify training, testing-set inference, and analysis outputs,
- then adapt the reproduced pipeline to the broader problem statement,
- and only after that finish backend/frontend integration.

See [paper reproduction plan](docs/planning/paper-reproduction-plan.md).
