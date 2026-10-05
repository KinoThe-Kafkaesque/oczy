---
name: Oczy Learning Workbench
description: A local evidence-reading interface with restrained actions and inspectable results.
colors:
  ground: "#0e1116"
  panel: "#161b22"
  line: "#303a49"
  text: "#d7dee7"
  muted: "#a1adbd"
  accent: "#a3b3ff"
  pass: "#77d5a0"
  fail: "#f18c9c"
  pending: "#e7ba69"
typography:
  headline:
    fontFamily: "'Iowan Old Style', Palatino, Georgia, serif"
    fontSize: "32px"
    fontWeight: 500
    lineHeight: 1.2
  title:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', sans-serif"
    fontSize: "18px"
    fontWeight: 600
    lineHeight: 1.55
  body:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.55
  code:
    fontFamily: "ui-monospace, Menlo, Consolas, monospace"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.7
rounded:
  control: "6px"
  code-panel: "8px"
spacing:
  detail: "4px"
  label: "8px"
  compact: "12px"
  regular: "16px"
  inset: "20px"
  group: "24px"
  wide: "28px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "#111628"
    rounded: "{rounded.control}"
    padding: "10px 16px"
  button-secondary:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    rounded: "{rounded.control}"
    padding: "10px 16px"
  input:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    rounded: "{rounded.control}"
    padding: "10px 12px"
  code-panel:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    typography: "{typography.code}"
    rounded: "{rounded.code-panel}"
    padding: "{spacing.inset}"
---

# Design System: Oczy Learning Workbench

## Overview

**Creative North Star: "The Research Record"**

This system applies only to `workbench/`. It records the built extension of the existing Oczy findings document: an ink background, lavender actions, serif research headings and flat, ruled evidence. The name describes the incumbent visual world; it does not represent a new identity approval.

The interface pairs reading with a small set of local controls. Dense results remain inspectable through clear type roles, explicit status words and adjacent limitations. Evidence receives the visual space; controls provide access to it. No imagery, decorative motion or card grid is required by this built world.

**Key Characteristics:**

- Dark ink ground with fine structural rules.
- Serif section headings, sans-serif reading text and monospace raw records.
- Lavender links and primary action; separate colors for outcome status.
- Full-width evidence with limitations kept alongside results.

The source of truth is `static/style.css`, `static/index.html` and `static/app.js`. The existing viewport captures and finish verdict corroborate the desktop hierarchy and corrected mobile capability reflow. This is a source-derived record, not a claim that every rendered state has been visually audited.

## Colors

The palette is a cool ink field with quiet blue-gray text, lavender interaction and distinct outcome colors. The frontmatter owns exact values.

### Primary

- **Lavender action (`accent`)** identifies links, disclosure controls, keyboard focus and the primary comparison action.

### Secondary

- **Positive mint (`pass`)** identifies explicitly correct outcomes and exact replay states.
- **Failure rose (`fail`)** identifies incorrect outcomes and changed replay states.
- **Pending amber (`pending`)** identifies study status, trial status and the local DEV scope marker. It is not a guarantee of success or an error by itself.

### Neutral

- **Ink ground (`ground`)** spans the document and all unboxed evidence.
- **Inset ink (`panel`)** distinguishes controls and raw-record panels.
- **Slate rule (`line`)** separates sections and evidence rows and outlines controls.
- **Reading silver (`text`)** carries headings, primary reading text and values.
- **Muted slate (`muted`)** carries explanatory prose, column labels, dates and secondary provenance.

**The Written Verdict Rule.** Color accompanies explicit words such as “Correct,” “Incorrect” or “failed”; it does not replace the verdict.

## Typography

**Display Font:** the existing Iowan Old Style / Palatino / Georgia serif stack.
**Body Font:** the native system sans-serif stack.
**Label/Mono Font:** UI monospace with Menlo and Consolas fallbacks for generated strings, transcripts, files and hashes.

The serif introduces research questions and sections. The sans-serif supports dense reading and familiar controls. Monospace preserves the distinction between raw records and interpretation.

### Hierarchy

- **Opening question:** serif, medium weight, balanced wrapping; the built introduction uses `clamp(38px,4vw,48px)` with a compact line height. This opening-specific setting is not a general display token.
- **Headline:** recurring serif section headings use the frontmatter headline role; they reduce to 29px at the narrow breakpoint.
- **Title:** semibold sans-serif for subsections and mobile capability names.
- **Body:** the default reading role; explanatory paragraphs stop at 72ch.
- **Label:** compact sans-serif for controls, evidence tables and supporting metadata. Fine provenance and mobile field labels step down to 12px; hints and disclosure notes use 13px.
- **Code:** raw panels use the code role; generated table strings inherit the table size while retaining monospace and preserved whitespace.

**The Raw Record Rule.** Preserve quotation, whitespace and wrapping in generated output. Do not style a cleaned-up interpretation as the actual output.

## Layout

The desktop document is centered with a maximum outer width of 1280px, 4vw side insets and generous ruled section separation. The header is a horizontal masthead; navigation sits to the right. The introduction places a wider question beside a narrower study-status area. These are sections of one continuous record, not independent tiles.

The trial form aligns labeled controls and actions along their lower edge. At 1050px and below it reduces to three control columns with actions on the next row. At 650px and below the header wraps, navigation receives a full row, side insets become 5vw, the introduction stacks, context spans the form, and each action occupies a full row. Tool transcripts and verified files move from adjacent columns to a vertical sequence.

Tables use collapsed borders, left-aligned headings, top-aligned cells and tabular numerals. Table wrappers allow horizontal scrolling and are keyboard-focusable after evidence loads. The repeated spacing vocabulary is in the frontmatter; local layout ratios and section-specific offsets remain implementation details.

**The Readable Evidence Rule.** At 650px and below, the prose capability table becomes full-width ruled entries. Each keeps its capability heading, then labeled “Current evidence” and “Limit” content. Desktop retains three columns. Preserve the row-header semantics; the repeated visual labels are hidden from assistive technology because the table already supplies the relationships.

## Elevation & Depth

The shipped interface has no shadows. Fine rules organize the ink ground, while the slightly lighter panel tone marks interactive controls and raw-record containers. Hover changes a control's fill; keyboard focus adds a separated lavender outline. Neither interaction changes apparent elevation.

**The Flat Record Rule.** Keep evidence on the continuous document ground. Reserve filled containers for controls and raw text panels, following the existing visual distinction.

## Shapes

Controls have modestly rounded corners; raw text panels use the slightly larger code-panel radius. Evidence rows, dividers and the outlined local-scope marker remain square. Borders are thin and quiet. There is no pill-button or rounded-card vocabulary in this workbench.

## Components

### Buttons

The primary action is a clear lavender fill with dark text and semibold weight. Saved inspection uses an inset-ink secondary button. Both share the control radius, a minimum height of 46px and the padding in the frontmatter. Hover lightens the fill; focus uses a 2px lavender outline offset by 4px. Disabled buttons use reduced opacity and a wait cursor. The sidecar carries these state rules because the frontmatter component schema does not support them.

### Inputs / Fields

Inputs and native selects share the panel fill, quiet rule border, control radius and a minimum height of 46px. Labels sit above fields with the label spacing. Text remains in the main reading color; placeholder text uses the muted color. The input caret and keyboard outline use lavender. Keep labels visible when controls reflow.

### Navigation

Underlined lavender section links sit beside the serif Oczy brand on desktop and wrap to their own row on mobile. Hover lightens links; focus remains explicit. The built navigation does not implement a current-section highlight. A skip link becomes visible when focused.

### Scope Marker

The local DEV marker is a small square outlined label with amber text. It communicates environment scope and is neither a filter nor an action.

### Evidence Record

Mechanism comparisons use flat ruled rows with the mechanism, actual quoted output and written exact score. Supporting distinctions appear as smaller muted text beneath the mechanism name. Capability rows pair each claim with a limitation and adopt the mobile reflow described in Layout. Outcome colors remain tied to written judgments.

### Raw Record Panels

Transcripts, verified files and reproduction commands share inset-ink panels with the code-panel radius and inset spacing. Text wraps while preserving whitespace; long tokens may wrap anywhere. Related transcript and file panels align side by side on desktop and stack on mobile. The background signals a literal record rather than a capability conclusion.

### Disclosures

Native details/summary controls use lavender summary text and familiar browser disclosure behavior. Supplemental score tables and provenance can be expanded without replacing the principal evidence. Expanded table regions have bounded vertical height and remain scrollable.

## Do's and Don'ts

### Do:

- **Do** retain explicit verdict text alongside outcome color.
- **Do** distinguish raw strings and records with monospace and preserved whitespace.
- **Do** keep each capability's limitation readable with its result at narrow widths.
- **Do** use the existing serif, sans-serif and monospace roles consistently.
- **Do** retain visible labels, keyboard outlines and access to scrollable evidence.

### Don't:

- **Don't** turn the continuous evidence record into a decorative card grid.
- **Don't** apply success styling to a capability solely because a historical failure replayed exactly.
- **Don't** visually collapse learned state, retrieved examples and externally supplied rules into one mechanism.
- **Don't** promote workbench-specific layouts or this document's tokens into repository-wide design policy.

Not canonized: overridden CSS declarations, one-off opening/section offsets, research result values and recovery wording. They describe local implementation or evidence state rather than a reusable visual rule. No craft-floor refusal was identified in the inspected source and supplied captures; this does not extend the finish review's coverage.
