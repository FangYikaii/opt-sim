# Workspace UX Spec

## Purpose

This document turns the high-level visual direction in [DESIGN.md](../../DESIGN.md) into an implementable product UX plan for the MVP Vue workspace.

The goal is not a marketing homepage. The primary surface is a scientific design workspace where users collaborate with an Agent, inspect simulation evidence, compare manufacturable candidates, and explicitly approve expensive export steps.

## Product Intent

The workspace should feel like a calm engineering cockpit:

- Text-forward, dense, and trustworthy.
- Focused on decision support instead of visual spectacle.
- Transparent about uncertainty, constraints, and long-running jobs.
- Optimized for repeated design-review loops rather than one-shot generation.

## Primary User Jobs

### 1. Start a Design Run

The user provides:

- Text requirements.
- Optional target color or target image assets.
- Optional manufacturing constraints.

The system responds with:

- Parsed requirement summary.
- Missing information warnings.
- A ready-to-run plan with simulation scope and cost/risk hints.

### 2. Review Candidate Solutions

The user needs to:

- Compare multiple inverse-design candidates.
- Understand target versus simulated behavior.
- See process robustness, not only DeltaE.
- Pick a candidate or ask the Agent to revise.

### 3. Approve Export

The user needs confidence before triggering expensive output generation:

- Dimensions, tile count, file size, and format must be explicit.
- The selected candidate and export parameters must remain visible.
- Confirmation must be a deliberate action, not an automatic side effect.

## Information Architecture

The MVP workspace uses one primary application shell with four cooperating regions.

### Top Bar

Persistent global context:

- Project name.
- Current run selector.
- Run status.
- Primary action.
- Secondary global actions such as settings and artifact download entry points.

### Left Rail

Navigation and project context:

- Project files and source assets.
- Uploaded targets.
- Design runs.
- Exported artifacts.

This rail should answer: "What am I working on, and what previous runs/artifacts exist?"

### Center Panel

The center is the main decision surface:

- Agent conversation.
- Timeline events.
- Tool-call progress.
- Approval cards.
- Run summaries.

This panel should answer: "What is the system doing, why, and what does it need from me?"

### Right Inspector

The evidence panel:

- Simulation preview.
- Candidate comparison.
- Spectra and color evidence.
- Constraint checklist.
- Ranking rationale.

This panel should answer: "Do I trust this candidate, and why was it ranked this way?"

### Bottom Drawer

Secondary but important detail:

- Logs.
- Raw JSON.
- Generated commands or manifests.
- Export progress detail.

This drawer should answer: "What exactly happened under the hood?"

## Primary Screens Within The Same Shell

The shell stays stable while the content shifts by run state.

### Empty Project

Used before any run exists.

Content priorities:

- Concise product framing.
- Target upload entry.
- Requirement input.
- Example prompt chips.
- Recent artifacts if present.

### Draft Run

Used after inputs are added but before execution.

Content priorities:

- Parsed requirement checklist.
- Feasibility warnings.
- Editable constraints.
- Run plan summary.
- Primary action to start the run.

### Active Run

Used during simulation, retrieval, ranking, or export.

Content priorities:

- Streaming timeline.
- Step progress.
- Intermediate candidate groups.
- Warnings and blockers.
- Live inspector updates.

### Review State

Used once candidates are ready.

Content priorities:

- Candidate comparison.
- Selected candidate state.
- Ranked rationale.
- Process sensitivity evidence.
- Revision and approval actions.

### Export State

Used before and during high-resolution output generation.

Content priorities:

- Export confirmation summary.
- Cost and size estimates.
- Progress by tile or stage.
- Artifact links once complete.

## End-To-End User Flow

### Flow A: Text + Target Color To Candidate Approval

1. User creates a run and enters design intent.
2. Agent parses the requirement and displays structured fields.
3. User reviews or adjusts constraints.
4. User starts the run.
5. Agent streams simulation and inverse-design events.
6. Ranked candidates appear in the inspector.
7. User compares candidates and selects one.
8. Agent summarizes tradeoffs and asks for export confirmation.

### Flow B: Target Images To Preview Validation

1. User uploads seed images.
2. Agent identifies missing views and assumptions.
3. Agent proposes a completion/simulation plan.
4. Timeline shows generation and validation events.
5. Inspector shows target, simulated, and error previews.
6. User either revises requirements or confirms continuation.

### Flow C: Export Confirmation

1. User enters export from a reviewed candidate.
2. Export panel shows dimensions, file size, tile strategy, and risk.
3. User confirms.
4. Timeline and bottom drawer stream export progress.
5. Artifacts become available in the left rail and export panel.

## Run State Model

These states should be represented consistently in top bar badges, left-rail run rows, and timeline milestones.

- `Draft`
- `Validating`
- `Simulating`
- `Ranking`
- `Needs approval`
- `Exporting`
- `Complete`
- `Failed`

Each state should have:

- Human-readable label.
- Semantic color token.
- Optional icon.
- Optional explanatory tooltip.

## Panel Behaviors

### Left Rail Behavior

- Width target: 280 px.
- Sections can collapse independently.
- Active run remains pinned near the top.
- Artifact rows should expose type, timestamp, and status at a glance.

### Center Panel Behavior

- Default scroll anchor stays near the latest timeline event.
- Approval cards should pin until resolved.
- Tool calls should collapse by default after completion.
- Long-running steps should show elapsed time and latest progress text.

### Right Inspector Behavior

- Context follows the current selection.
- If no candidate is selected, show the top-ranked candidate by default.
- Selection made in a candidate list must propagate to all inspector modules.
- Evidence modules should preserve scroll position while the timeline updates.

### Bottom Drawer Behavior

- Closed by default on first load.
- Auto-open on failure states or raw export generation if useful.
- Tabs should include `Logs`, `JSON`, and `Commands`.

## Responsive Strategy

### Desktop

Three-column layout with bottom drawer:

- Left rail visible.
- Center timeline primary.
- Right inspector persistent.

### Tablet

Two-column emphasis:

- Left rail becomes a drawer.
- Center remains primary.
- Inspector switches to tabbed sections inside the right area.

### Mobile

Single-column workflow:

- Top bar remains compact and sticky.
- Left rail becomes a sheet.
- Main content uses tabs: `Agent`, `Preview`, `Candidates`, `Export`.
- Approval cards must stay prominent and reachable without deep scrolling.

## Content Priority Rules

When space is constrained, keep these visible first:

1. Current run status.
2. Approval-needed messages.
3. Selected candidate summary.
4. Target versus simulated evidence.
5. Export warnings.

Hide or collapse these first:

- Raw logs.
- Long historical tool outputs.
- Secondary metadata.
- Low-priority project navigation groups.

## Empty, Loading, And Failure States

### Empty

- Use guided copy, not generic emptiness.
- Provide one clear next action.
- Include example requirement formats.

### Loading

- Prefer skeletons or structured placeholders over spinners alone.
- Keep labels visible so users know what is loading.
- Show progress text whenever the backend can provide it.

### Failure

- Show what failed, where, and what the user can do next.
- Provide retry or revise actions near the failure message.
- Keep prior successful results visible if they are still relevant.

## Interaction Principles

- Evidence before recommendation.
- Warnings before irreversible actions.
- Units always visible.
- Selection state must be obvious without relying only on color.
- The Agent should explain tradeoffs in short engineering language.
- The UI should never imply certainty when simulation confidence is limited.

## Accessibility Notes

- Timeline events must be keyboard navigable.
- Status should never rely on color alone.
- Swatches require text labels and numeric values.
- Charts need textual summaries for key values such as peak shift or DeltaE.
- Approval actions should have clear focus order and disabled-state rationale.

## Implementation Guidance For Vue MVP

Suggested top-level route structure:

```text
/                       Workspace shell with current project/run
/runs/:runId            Specific run focus
/artifacts/:artifactId  Artifact detail or download context
```

Suggested shell component breakdown:

```text
AppShell
  TopBar
  LeftRail
  AgentTimeline
  InspectorPanel
  BottomDrawer
```

The shell should be implemented first, even with mocked data, because the interaction model depends more on stable layout and state transitions than on final backend wiring.
