# DESIGN.md: Claude Code Inspired Microstructure Design Workspace

## Visual Theme and Atmosphere

Build a scientific AI design workspace inspired by Claude Code and warm Claude editorial cues. The interface should feel like a calm engineering cockpit: precise, text-forward, trustworthy, and good for long design sessions.

Use a developer-tool layout rather than a generic AI landing page. The product is a working environment for optics simulation, inverse design, Agent reasoning, and manufacturing export.

## Design Principles

- Prefer dense but readable panels over oversized marketing cards.
- Put the Agent timeline at the center, because the user collaborates with the system through staged decisions.
- Keep simulation evidence visible: target color, simulated color, spectra, DeltaE, process risk, and export status.
- Make uncertainty explicit with warnings, checklists, and candidate score explanations.
- Use warm accents sparingly to highlight confirmations, selected candidates, and Agent reasoning.

## Color Palette

```css
:root {
  --color-bg: #171512;
  --color-bg-elevated: #211f1b;
  --color-bg-panel: #26231e;
  --color-bg-soft: #f4efe7;
  --color-text: #f3eee7;
  --color-text-muted: #b8ada0;
  --color-text-inverse: #221f1b;
  --color-border: #3a352d;
  --color-border-strong: #5a5145;
  --color-accent: #d97757;
  --color-accent-soft: #f2c2ad;
  --color-accent-deep: #9f4f35;
  --color-ok: #82b366;
  --color-warning: #d6a84f;
  --color-danger: #d56b62;
  --color-data-blue: #8bb8d8;
  --color-spectrum-green: #91c7a9;
}
```

## Typography

Use a warm text family for conversation and a crisp monospace for parameters.

- Interface sans: `Avenir Next`, `Nunito Sans`, or `Geist`.
- Monospace: `JetBrains Mono`, `Geist Mono`, or `IBM Plex Mono`.
- Avoid default Arial/System-only styling unless forced by environment.

Type scale:

- App title: 20-24 px, 600 weight, slight negative letter spacing.
- Panel heading: 13-15 px, 600 weight, uppercase optional for tool sections.
- Body: 14-16 px, 400 weight, high line height around 1.55.
- Parameter values: 12-14 px monospace.
- Status labels: 11-12 px monospace, uppercase, letter spacing 0.06em.

## Layout

Primary desktop layout:

```text
Top bar: project name, run selector, global actions
Left rail: files, uploaded targets, design runs, exports
Center: Agent chat and run timeline
Right inspector: simulation preview, candidates, spectra, constraints
Bottom drawer: logs, raw JSON, generated commands
```

Recommended proportions:

- Left rail: 260-320 px.
- Center Agent panel: flexible, minimum 460 px.
- Right inspector: 380-520 px.
- Gaps: 10-14 px.
- Panel radius: 12-16 px.
- Card radius: 10-12 px.

Responsive behavior:

- On tablet, collapse left rail into a drawer and keep Agent + inspector tabs.
- On mobile, use a single-column run workflow with tabs: Agent, Preview, Candidates, Export.

## Components

### Top Bar

Dark elevated surface with subtle border. Include:

- Project name: "Opt-Sim Microstructure Agent".
- Current run state: Draft, Simulating, Needs approval, Exporting, Complete.
- Primary action: "Start design run" or "Approve export".

### Agent Timeline

Use message cards with tool-call blocks.

Agent message styles:

- Warm panel background.
- Small role label.
- Tool calls rendered as compact monospace cards.
- Approval-needed state uses accent border and a clear action row.

Timeline event types:

- Requirement parsed.
- Physics simulation started.
- Candidate search completed.
- Constraint warning.
- Human approval needed.
- Export completed.

### Candidate Cards

Each candidate should show:

- Candidate ID and rank.
- Design parameters with units.
- Target vs simulated color swatches.
- DeltaE.
- Sensitivity under process error.
- Constraint status.
- Select/compare action.

Selected candidate:

- Accent border.
- Slightly brighter surface.
- Checkmark or "Selected" text, not only color.

### Simulation Inspector

Show evidence first:

- Color swatches: target, simulated, process-error plus, process-error minus.
- Spectrum chart.
- Angle preview grid.
- Constraint checklist.
- Explanation of why the candidate was ranked.

Charts should use calm spectrum-inspired colors and thin grid lines. Avoid neon rainbow overload.

### Export Panel

Because high-resolution export can be expensive, show:

- Estimated dimensions.
- Estimated file size.
- Tile size.
- Output format.
- Required confirmation.
- Progress bar and tile counter.

## Motion

Use restrained motion:

- Agent events fade/slide in by 6-10 px.
- Tool progress pulses softly.
- Candidate selection transitions border/background over 120-180 ms.
- Avoid playful bouncing or excessive animation.

## Copy Tone

Use clear, collaborative engineering language:

- "I found 4 manufacturable candidate groups."
- "This candidate has lower DeltaE, but it is more sensitive to 0.5 um process error."
- "Please confirm before generating the 160k x 320k export."

Avoid vague AI claims like:

- "Magic design generated."
- "Perfect output guaranteed."

## Do

- Keep units visible everywhere.
- Show validation and uncertainty.
- Preserve selected candidate context across panels.
- Make long-running jobs feel transparent.
- Use dark surfaces with warm Claude-like accents.

## Don't

- Do not copy Claude logos, proprietary branding, or exact layouts.
- Do not make the UI a generic chatbot without simulation evidence.
- Do not hide physical constraints in advanced settings only.
- Do not rely on purple gradients or generic AI SaaS styling.
- Do not generate high-resolution export without explicit confirmation.
