# Component Spec

## Purpose

This document defines the MVP components for the Opt-Sim workspace so implementation can begin without re-deciding field structure and interaction behavior on each screen.

Component design follows the product rules in [DESIGN.md](../../DESIGN.md) and the layout rules in [workspace-ux-spec.md](./workspace-ux-spec.md).

## Shared Data Display Rules

Apply these rules across all components:

- Always show physical units.
- Prefer labels plus values instead of unlabeled numeric chips.
- Use monospace for parameters, IDs, status tags, and compact evidence values.
- Keep semantic status text visible alongside color.
- When values are approximate or simulated under error, label them explicitly.

## Top Bar

### Purpose

Maintains global project and run context.

### Required Content

- Project title: `Opt-Sim Microstructure Agent`
- Current run selector
- Run status badge
- Primary action button
- Secondary actions: settings, artifact entry, optional reconnect/retry

### States

- No run selected
- Draft run
- Active run
- Approval needed
- Exporting
- Complete

### Interaction Notes

- Changing the run updates center and right panels together.
- Primary action label changes with state, for example `Start design run` or `Approve export`.
- Status badge should include iconography for warning/failure states.

## Left Rail

### Sections

- `Targets`
- `Runs`
- `Exports`
- `Project files`

### Item Content

Run item:

- Run name or generated timestamp label
- State badge
- Last updated time
- Optional warning dot

Target item:

- Thumbnail or color swatch
- Name
- Type such as `color`, `image`, `multi-view`

Artifact item:

- Artifact name
- Type
- Completion status

### Interaction Notes

- Active item uses stronger border and background, not just text color.
- Sections can collapse.
- Rail should support long project histories without overwhelming the current run.

## Requirement Composer

### Purpose

Captures the initial brief before the run starts.

### Fields

- Requirement text area
- Optional target color input
- Optional image upload
- Constraint chips or compact fields
- Example prompt shortcuts

### Validation

- Missing target is allowed if the workflow supports text-only exploration.
- Invalid units or unsupported formats should be flagged inline.
- Upload state must show progress and completion feedback.

### Output

Structured requirement draft to feed the Agent parser.

## Requirement Summary Card

### Purpose

Shows the Agent-parsed interpretation before execution.

### Sections

- Target summary
- Manufacturing constraints
- Planned simulation scope
- Open assumptions
- Risk warnings

### Actions

- Confirm
- Edit requirements
- Start run

### Behavior

- Open assumptions are visually distinct from confirmed facts.
- If feasibility is questionable, the card should surface that before run start.

## Agent Timeline

### Purpose

Acts as the core collaborative workspace for the run.

### Event Types

- Agent message
- Tool call started
- Tool progress
- Tool result
- Warning
- Approval request
- Export completion
- Failure

### Event Card Anatomy

- Role or event label
- Timestamp
- Main summary sentence
- Optional expandable details
- Optional action row

### Behavior

- New events animate in subtly.
- Completed tool calls collapse to compact summaries.
- Approval requests remain pinned until resolved.
- Warnings use high contrast but should not dominate unrelated content.

## Tool Call Block

### Purpose

Compact representation of backend actions triggered by the Agent.

### Required Fields

- Tool name
- Current phase
- Elapsed time
- Short status text

### Optional Fields

- Input summary
- Output summary
- Progress percentage
- Link to raw JSON in the drawer

### Visual Treatment

- Monospace-heavy compact card
- Quiet surface treatment
- Stronger border when active

## Candidate List

### Purpose

Shows ranked inverse-design results in a scan-friendly way.

### Required Candidate Fields

- Candidate ID
- Rank
- Key design parameters with units
- DeltaE
- Constraint status
- Process sensitivity summary
- Selection affordance

### Optional Fields

- Material stack label
- Group label for clustered solutions
- Manufacturability score
- Predicted export risk

### Behavior

- Supports compare mode for 2-3 candidates.
- Selected card updates the inspector immediately.
- If a candidate becomes invalid after a constraint change, mark it instead of silently removing it.

## Candidate Card

### Purpose

Dense evidence summary for one candidate.

### Content Order

1. Header with ID, rank, and state
2. Design parameters
3. Color swatches
4. Key metrics
5. Risk and constraint summary
6. Actions

### Required Metrics

- `DeltaE`
- Thickness or height range
- Sensitivity under process error
- Constraint pass/fail

### Required Actions

- `Select`
- `Compare`
- `Inspect`

### Selected State

- Accent border
- Brighter background
- Explicit text such as `Selected`

## Simulation Inspector

### Purpose

Displays the evidence behind the selected or highlighted candidate.

### Sections

- Target vs simulated swatches
- Spectrum chart
- Angle preview grid
- Constraint checklist
- Ranking rationale

### Default Behavior

- Follows selected candidate
- Falls back to top-ranked candidate
- Preserves state while new timeline events stream in

## Color Swatch Module

### Required Swatches

- Target
- Simulated
- Process error `+`
- Process error `-`

### Required Labels

- Color role
- Lab or sRGB values
- Delta summary where relevant

### Accessibility

- Never rely on color alone.
- Every swatch needs a text label and numeric representation.

## Spectrum Chart

### Purpose

Shows spectral evidence without overwhelming the user.

### Requirements

- Wavelength axis from 380-780 nm for MVP
- Thin grid lines
- Calm chart colors
- Hover or focus detail if implemented

### Supporting Text

- Short summary sentence under the chart
- Highlight if spectral deviations are concentrated in a narrow band

## Angle Preview Grid

### Purpose

Prepares the product for multi-view simulation review.

### MVP Content

- A small grid of selected view previews
- Angle labels with units
- Optional placeholders when multi-view is not yet available

### Behavior

- Keep the selected view highlighted.
- Do not fake unavailable angles as if they were simulated.

## Constraint Checklist

### Purpose

Makes manufacturability visible rather than hidden in settings.

### Items

- Height range
- Quantization/precision
- Thickness bounds
- Smoothness or continuity
- Export feasibility

### States

- Pass
- Warning
- Fail
- Unknown

### Behavior

- Each row should explain why it passed or failed in plain engineering language.

## Ranking Rationale Panel

### Purpose

Explains why a candidate is above or below another candidate.

### Content

- Score summary
- Major tradeoffs
- Dominant penalties
- Reason candidate is recommended or not recommended

### Copy Style

- Short declarative sentences
- Avoid opaque ML language

## Approval Card

### Purpose

Collects deliberate confirmation before export or other costly steps.

### Required Content

- What action is being approved
- Why approval is needed
- Estimated output dimensions
- Estimated file size
- Tile plan
- Selected candidate reference

### Actions

- `Approve export`
- `Revise`
- `Cancel`

### Behavior

- The consequences of approval must be explicit.
- Disabled approve states need a reason.

## Export Progress Panel

### Purpose

Tracks long-running output generation.

### Required Fields

- Current stage
- Tile counter
- Percent complete
- Latest generated artifact
- Estimated remaining time if available

### Behavior

- Progress updates should be stream-friendly.
- Completed tiles may appear in the drawer or artifact list progressively.

## Bottom Drawer

### Tabs

- `Logs`
- `JSON`
- `Commands`

### Purpose

Provides inspectability for advanced users without crowding the main workspace.

### Behavior

- Remembers last-opened tab per run if easy to implement.
- Can deep-link from a timeline tool block to a JSON/log entry.

## Recommended MVP Implementation Order

1. `AppShell` with top bar, left rail, center, right, and bottom regions.
2. `RequirementComposer` and `RequirementSummaryCard`.
3. `AgentTimeline` with `ToolCallBlock` and `ApprovalCard`.
4. `CandidateList` and `CandidateCard`.
5. `SimulationInspector` with swatches, checklist, and placeholder chart.
6. `ExportProgressPanel` and bottom drawer detail tabs.

## Suggested Frontend Types

The following entities deserve explicit TypeScript types early:

- `RunSummary`
- `RunStatus`
- `TimelineEvent`
- `CandidateSolution`
- `SimulationEvidence`
- `ConstraintCheck`
- `ExportEstimate`
- `ArtifactSummary`

These types should align with the backend contracts planned in [requirements-breakdown.md](../planning/requirements-breakdown.md).
