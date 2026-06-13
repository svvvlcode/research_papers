# SVVVL Research

SVVVL's **collection of LaTeX research papers** — self-contained `.tex` documents in the
house style, compiled to PDF. The papers are quantitative-finance theory — ergodicity
economics, Kelly/Merton allocation, evolutionary foundations of portfolio choice — and
adjacent work applying the same variational / least-action lens elsewhere (maximum-caliber /
maximum-diffusion ideas in token generation; optimal-control pacing in thoroughbred racing).

The reading and distilling that *precede* a paper happen in the research wiki, a separate
Obsidian vault at [`../SVVVL_Second_Brain/`](../SVVVL_Second_Brain/). This repo is where
mature insight gets published.

> **Working with Claude Code?** The authoritative, detailed spec for how this repository is
> organized and maintained lives in [`CLAUDE.md`](CLAUDE.md). This README is the human-facing
> orientation; `CLAUDE.md` is the schema the corpus co-evolves with.

## The papers

Most papers are a self-contained `.tex` document in the SVVVL house style, compiled to PDF;
one (optimal pacing) splits across two standalone `.tex` documents and uses its own plain
preamble.

- `template/` — canonical starting point for a new paper (`svvvl_paper_template.tex`).
- `research_papers/goal_based_investing/` — single `.tex` (`svvvl_ergodic_technical.tex`).
- `research_papers/nature_abhors_an_undiversified_bet/` — single `.tex`.
- `research_papers/max_caliber_token_generation/` — paper-in-progress: a Markdown drafting
  plan, no `.tex` yet.
- `research_papers/optimal_pacing_in_thoroughbred_racing/` — two standalone LaTeX documents:
  `methodology/` (the variational formulation, `methodology.tex`) and `stage0/` (`stage0.tex`,
  the Stage 0 reduction with its inference implementation written as pseudocode). Uses its own
  plain `article` preamble, not the house style.
- `assets/` — shared SVVVL brand assets (e.g. `SVVVL_Research_Wordmark.pdf`).

## Repository layout

```
SVVVL_Research/               # repo root
├── CLAUDE.md                 # Schema for the paper corpus (read this first if maintaining)
├── README.md                 # This file
├── claude-answers.md         # Q&A scratchpad (written by /prepend; outside the corpus)
├── template/                 # house-style paper template
├── research_papers/          # published / in-progress papers
│   ├── goal_based_investing/
│   ├── nature_abhors_an_undiversified_bet/
│   ├── max_caliber_token_generation/        # plan only, no .tex yet
│   └── optimal_pacing_in_thoroughbred_racing/  # methodology/ + stage0/ (two LaTeX docs)
└── assets/                   # shared brand assets
```

## The pipeline: from source to paper

Papers don't start here — they incubate in the research wiki and graduate across the
directory boundary:

```
source → ../SVVVL_Second_Brain/raw/  →  wiki/sources/  →  wiki/concepts + wiki/entities
                                      →  insight accumulates in wiki/syntheses/
                                      →  graduates into SVVVL_Research/research_papers/  (LaTeX)
```

A synthesis **graduates** into a paper when it is correct and complete enough for the
corpus. After graduation, the wiki's synthesis and source pages remain the audit trail behind
the paper's claims. See *The bridge* in [`CLAUDE.md`](CLAUDE.md) for the full workflow, and
[`../SVVVL_Second_Brain/CLAUDE.md`](../SVVVL_Second_Brain/CLAUDE.md) for the wiki's schema.

## Building a paper

Papers build with `latexmk` (MiKTeX toolchain). From a paper's directory:

```
latexmk -pdf -synctex=1 -interaction=nonstopmode -file-line-error <document>.tex
```

In VS Code, the LaTeX Workshop recipe auto-builds on save and writes the PDF beside the
`.tex`. Graphic paths are relative to the paper directory and **must use forward slashes**
(`/`) even on Windows — a backslash is parsed as a LaTeX control sequence. Papers sit two
levels deep, so shared assets in `assets/` are reached with `../../assets/`. If a build
fails on a missing graphic, check this first.

## Conventions at a glance

- **Paper math:** prefer the SVVVL macros (`\E`, `\Var`, `\Corr`, `\dd`, `\lopt`, `\mue`, …).
  The wiki writes the same math with these macros *expanded*, so porting into a paper is
  mechanical.
- **Paper citations:** hand-numbered `\hypertarget{ref:N}` / `[\hyperlink{ref:N}{N}]` —
  no `.bib`/BibTeX.
- Every paper ends with a verbatim **Disclosures** section (not investment advice).

---

*SVVVL Research. Papers here are part of SVVVL's internal research body; rigor is motivated
by internal correctness and corpus integrity, not external peer review.*
