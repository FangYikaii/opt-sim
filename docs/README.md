# Documentation Index

This folder is the single entry point for project documentation. Use it to find the current product plan, technical architecture, task breakdown, design rules, decisions, and reference notes.

## Reading Order

1. [Project README](../README.md): quick project overview and current next step.
2. [Technical proposal](architecture/technical-proposal.md): full system plan and engineering direction.
3. [Requirements breakdown](planning/requirements-breakdown.md): phased tasks, acceptance criteria, and verification plan.
4. [cGAN / optics alignment PRD](planning/cgan-optics-data-alignment-prd.md): planned alignment for real D65 / tristimulus data, scaling, colour-science integration, and retrieval metrics.
5. [Colorimetry and cGAN alignment design](architecture/colorimetry-and-cgan-alignment-design.md): module boundaries, interfaces, checkpoint migration, and retrieval-metric design.
6. [ADR-001](decisions/ADR-001-architecture-and-model-choices.md): why the current architecture and model choices were accepted.
7. [ADR-002](decisions/ADR-002-use-real-colorimetry-data-and-versioned-cgan-scaling.md): why real colorimetry data, `colour-science`, and versioned scaling metadata are the new baseline.
8. [UI design guide](../DESIGN.md): Claude Code inspired visual and interaction direction.
9. [Workspace UX spec](design/workspace-ux-spec.md): implementable information architecture, flows, states, and responsive behavior.
10. [Component spec](design/component-spec.md): core workspace module definitions and interaction details.
11. [Algorithm operations guide](planning/algorithm-operations-guide.md): how to interpret current algorithm quality, run the stack, and use the new overview UI.

## Directory Structure

```text
docs/
  README.md                 Documentation index and structure guide
  architecture/             Technical architecture and system design
  planning/                 Requirements, task plans, milestones
  decisions/                Architecture Decision Records
  design/                   Product UX notes, wireframes, visual specs
  references/               Notes extracted from papers, books, and source materials
```

## Document Types

### Architecture

Use `docs/architecture/` for documents that explain how the system works.

Examples:

- Physics model design.
- Backend/frontend architecture.
- Agent tool architecture.
- Data and export pipeline design.

### Planning

Use `docs/planning/` for execution plans and project management materials.

Examples:

- Requirements breakdown.
- Milestones.
- Sprint plans.
- Acceptance criteria.

### Decisions

Use `docs/decisions/` for ADR files. Each ADR should explain context, decision, alternatives, and consequences.

Naming format:

```text
ADR-001-short-title.md
ADR-002-short-title.md
```

### Design

Use `docs/design/` for UX details that are more specific than the root `DESIGN.md`.

Examples:

- Page-level interaction flows.
- Wireframes.
- Component behavior.
- User journey notes.
- Workspace interaction specs.
- Component contracts for implementation.

The root [DESIGN.md](../DESIGN.md) remains the top-level visual design system that coding agents should read before implementing UI.

### References

Use `docs/references/` for notes extracted from source materials. Keep large PDFs and raw requirement images at the project root or their current source folders unless we later create an `assets/` archive.

Examples:

- Paper summary.
- Optics formula notes.
- Requirement image transcription.
- Material data source notes.

## Maintenance Rules

- Keep the root [README.md](../README.md) short and navigational.
- Keep detailed implementation plans inside `docs/`.
- Update an ADR when a major technical direction changes; do not silently overwrite historical rationale.
- Prefer relative links between docs.
- Add new documents to this index when they become important entry points.

## Naming Conventions

- Use lowercase kebab-case for normal documentation files.
- Keep names descriptive but short.
- Put sequence numbers only on ADR files or when strict reading order matters.

Examples:

- `technical-proposal.md`
- `physics-engine-design.md`
- `milestone-plan-phase-1.md`
- `agent-workflow-notes.md`

Avoid:

- `final_v2_new.md`
- `notes123.md`
- `temp.md`
