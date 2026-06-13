# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository. It is the schema for this paper corpus and co-evolves with the user —
when a workflow stops working, propose an edit to this file and apply it on approval.

## What this is

SVVVL's **collection of LaTeX research papers.** Each paper is a self-contained `.tex`
document compiled to PDF, written in the SVVVL house style. The papers are
quantitative-finance theory (ergodicity economics, Kelly/Merton allocation, evolutionary
foundations of portfolio choice) and adjacent work applying the same variational /
least-action lens elsewhere (maximum-caliber / maximum-diffusion ideas in token generation;
optimal-control pacing in thoroughbred racing).

Papers here are part of SVVVL's research body; not all target external publication, so
motivate rigor by **internal correctness and corpus integrity**, not peer review.

> **Where paper ideas come from.** This repo holds the *published* papers. The reading and
> distilling that precede a paper happen in the research wiki, a separate Obsidian vault at
> `..\SVVVL_Second_Brain\`. Insight accumulates there as a synthesis, then **graduates**
> into a paper here. See [The bridge](#the-bridge--from-wiki-insight-to-paper) for the seam,
> and `..\SVVVL_Second_Brain\CLAUDE.md` for the wiki's own schema.

## Repository layout

```
SVVVL_Research/               # repo root
├── CLAUDE.md                 # this file — the paper-corpus schema
├── README.md                 # human-facing orientation
├── claude-answers.md         # Q&A scratchpad (written by /prepend; outside the corpus)
├── template/                 # canonical starting point for a new paper
├── research_papers/          # container for all papers (not itself a paper)
│   ├── goal_based_investing/                # paper: svvvl_ergodic_technical.tex
│   ├── nature_abhors_an_undiversified_bet/  # paper
│   ├── max_caliber_token_generation/        # paper-in-progress: plan only, no .tex yet (MaxDiff decoding)
│   └── optimal_pacing_in_thoroughbred_racing/  # methodology/ + stage0/ (two standalone LaTeX documents)
└── assets/                   # shared brand assets (e.g. SVVVL_Research_Wordmark.pdf)
```

Each paper lives **two levels deep** from the repo root, in its own directory under
`research_papers/`. The shape varies:

- **Single-`.tex` papers** — `goal_based_investing/` and `nature_abhors_an_undiversified_bet/`
  each hold one primary `.tex` source plus its build artifacts.
- **Plan-only** — `max_caliber_token_generation/` currently holds just a Markdown drafting
  plan (`maxdiff_decoding_paper_plan.md`); the `.tex` has not been scaffolded yet.
- **Multi-document** — `optimal_pacing_in_thoroughbred_racing/` holds two standalone LaTeX
  documents: a `methodology/` directory (`methodology.tex`, the full log-reparameterized
  variational formulation) and a `stage0/` directory whose `stage0.tex` states the Stage 0
  reduction (theory) and lays out its JAX/NumPyro inference implementation **as pseudocode**.
  No code, tests, or figures are checked in — it builds with LaTeX alone.

LaTeX build artifacts (`*.aux`, `*.log`, `*.fdb_latexmk`, `*.fls`, `*.out`,
`*.synctex.gz`, etc.) are git-ignored — `template/` and
`nature_abhors_an_undiversified_bet/` carry a `.gitignore` to that effect. Such artifacts
are currently checked in under several paper dirs (`nature_abhors_an_undiversified_bet/`,
`goal_based_investing/`, `optimal_pacing_in_thoroughbred_racing/`) — leave them or clean
them, but do not treat them as source.

## Writing style (prose & docs)

Applies to all prose — LaTeX, markdown, docstrings, comments, and chat answers. Distilled
from McCloskey, *Economical Writing*: clarity over brevity, since "paucity at the cost of
clarity is false economy."

- **Clarity is the writer's duty, not the reader's.** The reader is the boss; if a sentence
  is hard, fix the sentence.
- **Cut, then cut again.** Write, then rewrite. Kill boilerplate and throat-clearing —
  no "this paper/section will…" openings, no table-of-contents paragraphs.
- **Be concrete.** Prefer a worked example or a specific noun to an abstraction.
- **Strong verbs, active voice.** Replace weak verb + adverb ("ran quickly" → "sprinted") and
  nominalizations ("make a decision" → "decide").
- **One word per concept — no elegant variation.** Reuse the same term; don't reach for a
  synonym to relieve "monotony." (E.g., pick *ordering* and keep it; don't drift to
  *sequence*/*permutation*.)
- **Plain over pompous.** Anglo-Saxon over Latinate; never a big word where a small one
  serves; no jargon for its own sake.
- **No vague `this`/`that`/`it`** without a clear antecedent — name the thing.
- **Front-load and read aloud.** Put the point first; end a sentence on its strongest word;
  let your ear catch what your eye misses.

---

# Paper authoring (LaTeX)

Most papers are a self-contained `.tex` document in the SVVVL house style, with no code to
build or test beyond compiling the LaTeX. New papers should follow that house style
(below). Two existing papers depart from it, and that is fine — describe reality, don't
force-fit them:

- `optimal_pacing_in_thoroughbred_racing/` uses a **plain `article` preamble of its own**
  (no wordmark, `\makecover`, or SVVVL macros) and splits across **two `.tex` documents**
  (`methodology/methodology.tex` and `stage0/stage0.tex`) rather than one. Its `stage0.tex`
  describes a JAX/NumPyro inference pipeline as pseudocode, but no code is checked in — it
  builds with LaTeX alone.
- `max_caliber_token_generation/` is **not yet a `.tex`** — only a Markdown plan.

## Starting a new paper

Begin from `template/svvvl_paper_template.tex` rather than copying an existing paper. It
carries the full house-style preamble and boilerplate (logo lockup, `\makecover`, generic
math macros, theorem environments, the verbatim Disclosures section) with the prose
stripped to placeholders.

1. Copy the `template/` directory to a new directory under `research_papers/` named for
   the paper, and rename the `.tex` inside it.
2. Fill in the `\makecover` title/subtitle, the abstract, and the body sections; add any
   paper-specific notation macros in the marked "Custom commands" block.
3. Repoint the wordmark: a paper sits **two levels deep**, so it needs
   `../../assets/SVVVL_Research_Wordmark.pdf` (not the template's `../assets/`). See
   [Asset paths](#asset-paths-important-and-inconsistent-between-papers).

Edit content, not the preamble: the preamble is the shared house style and should stay
identical across papers.

## Building

Build with `latexmk` (MiKTeX toolchain). The repo pins exact MiKTeX binaries in
`.vscode/settings.json` because a second Perl (miniconda) on PATH can hijack the
toolchain. From a paper's directory:

```
latexmk -pdf -synctex=1 -interaction=nonstopmode -file-line-error <document>.tex
```

In VS Code (LaTeX Workshop), the recipe auto-builds on save and writes the PDF beside the
`.tex` (`outDir=%DIR%`).

## Asset paths (important, and inconsistent between papers)

pdflatex runs from the paper's own directory, so graphic paths are relative to that
directory and **must use forward slashes** (`/`) — even on Windows. A backslash in an
`\includegraphics`/path macro is parsed as a control sequence (e.g. `\assets` →
"Undefined control sequence") and silently mangles the filename.

Papers live two levels deep (`SVVVL_Research/research_papers/<paper>/`), so shared assets in
`SVVVL_Research/assets/` are reached with **`../../assets/`** (the template, one level deep,
correctly uses `../assets/` — don't copy that single-`..` into a paper).

- **Known bug:** both existing papers (`nature_abhors_an_undiversified_bet`,
  `goal_based_investing`) set `\WORDMARKFILE` to `../assets/…`, which at two-deep resolves
  to the nonexistent `research_papers/assets/` — they won't compile until repointed to
  `../../assets/SVVVL_Research_Wordmark.pdf`.

If a build fails on a missing graphic, check this path convention first.

## House style (shared LaTeX preamble conventions)

All papers share a deliberate visual style, captured canonically in
`template/svvvl_paper_template.tex` — preserve it when editing or creating a paper:

- `documentclass[11pt]{article}`, `letterpaper`, `geometry` margins `top/bottom=0.9in`,
  `left/right=0.85in`, `headheight=14pt`.
- Charter-family body font (`XCharter` / `charter`), `titlesec` section formatting,
  **section numbering disabled** (`\setcounter{secnumdepth}{0}`).
- `\setstretch{1.15}`, `\parskip=0.6em`, `\parindent=0pt`.
- Link color `linkcolor` = HTML `1F4E79`; `hyperref` with `colorlinks=true`.
- `fancyhdr` page style with a centered page-number footer and a clean `firstpage` style
  for the cover.
- A `\makecover{Title}{Subtitle}{Authors}{Date}` command and a `\logolockup` command
  drive the cover page.
- References are a hand-numbered `itemize` list using `\hypertarget{ref:N}` /
  `\hyperlink{ref:N}{N}` — there is no `.bib`/BibTeX. Cite with `[\hyperlink{ref:N}{N}]`
  and keep the numbering in sync with the reference list.
- Every paper ends with a **Disclosures** section (not investment advice). Keep it on new
  papers.

## Math notation (paper layer)

`nature_abhors_an_undiversified_bet.tex` defines custom macros reused throughout — prefer
them over inline expansions: `\lopt` (ℓ*), `\lkelly` (ℓ**), `\mue`/`\mub`/`\mus`
(excess/risk-free/risky drift), `\E`, `\Var`, `\Corr`, `\dd` (upright d). The template's
"Custom commands" block carries the generic operators (`\E`, `\Var`, `\Corr`, `\dd`); add
paper-specific notation there.

> **The wiki feeds this layer.** Wiki notes in `..\SVVVL_Second_Brain\` deliberately write
> math in plain, portable LaTeX with these paper macros **expanded** (Obsidian doesn't know
> SVVVL's `\lopt`, `\mue`, `\E`, …) and record the macro correspondence in prose. Porting a
> synthesis's math into a paper is therefore mechanical: swap each expanded expression back
> for its macro.

---

# The bridge — from wiki insight to paper

This is the seam between two repositories: how reading in the research wiki
(`..\SVVVL_Second_Brain\`) becomes a published paper here.

**Lifecycle.** A source enters `..\SVVVL_Second_Brain\raw\` → gets a `wiki/sources/` page →
seeds/updates `wiki/concepts` and `wiki/entities` → insight accumulates in a
`wiki/syntheses/` page. When that synthesis is **correct and complete enough for the corpus**
(not "would a referee accept it" — see *What this is*), it **graduates** into a LaTeX paper
under `research_papers/` here.

### Workflow: graduate a synthesis into a paper

1. **Confirm readiness with the user.** Graduation creates a new paper dir under
   `research_papers/` and shifts the work from markdown to LaTeX. Don't start unprompted.
2. **Scaffold the paper.** Follow [Starting a new paper](#starting-a-new-paper): copy
   `template/` to a new `kebab_or_snake_case` paper directory under `research_papers/`,
   rename the `.tex`.
3. **Port the prose.** Translate the synthesis's distilled markdown (in
   `..\SVVVL_Second_Brain\wiki\syntheses\`) into LaTeX body sections. The synthesis's section
   plan usually maps directly onto the paper's sections.
4. **Port the math.** Convert MathJax `$...$` into the paper's LaTeX, swapping expanded
   expressions for SVVVL macros (`\E`, `\mue`, `\lopt`, …) per the correspondences noted
   in the wiki. Define any new paper-specific macros in the "Custom commands" block.
5. **Build the reference list.** The synthesis and the `wiki/sources/` pages it cites
   carry full bibliographic data in their `## Citation` blocks. Turn each into a
   hand-numbered `\hypertarget{ref:N}` entry, and replace the wiki's `[[wikilinks]]` to those
   sources with `[\hyperlink{ref:N}{N}]`. Keep numbering in sync with the list.
6. **Keep the Disclosures** section verbatim.
7. **Record the graduation — in the wiki repo.** Set the synthesis's `**Status:**` to
   `graduated → paper-dir/`, link the paper dir from the synthesis, and append a `paper`
   entry to `..\SVVVL_Second_Brain\wiki\log.md`.

> The wiki is the system of record for *why* a paper says what it says. After graduation,
> the synthesis and source pages (in `..\SVVVL_Second_Brain\`) remain the audit trail behind
> the published claims — keep them rather than deleting them.

---

## The Q&A scratchpad (`claude-answers.md`)

`claude-answers.md` (repo root) is a plain Q&A log the `/prepend` skill writes — a scratchpad
for paper-side questions. Math is written so Obsidian MathJax renders it (`$...$` / `$$...$$`).
It is part of neither the paper corpus (no `\hyperlink` reference machinery) nor the wiki (no
`[[wikilinks]]`, no index/log bookkeeping).

**Rule: newest entry on top.** Every `/prepend` **prepends** its section to the top of the file
(reverse-chronological, most recent first), directly below the YAML frontmatter and the
`# Claude — Q&A scratchpad` title block, with a `---` rule separating it from the entry below.

---

## What lives where (quick reference)

| You want to...                              | Look in / write to            |
|---------------------------------------------|-------------------------------|
| Read or distill sources, look up an idea, see a synthesis | `..\SVVVL_Second_Brain\` (the research wiki) |
| Start / write a LaTeX paper                 | `template/` → new paper dir   |
| Find the canonical house style              | `template/svvvl_paper_template.tex` |
| Find an existing paper                      | `research_papers/<paper>/`    |
| Find shared brand assets (wordmark, …)      | `assets/`                     |
| Read/append the Q&A scratchpad (newest on top) | `claude-answers.md` (via `/prepend`) |
| Change how any of this works                | `CLAUDE.md` (this file)       |
