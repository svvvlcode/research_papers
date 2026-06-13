---
description: Prepend my previous response to the top of claude-answers.md (math-ready, dated heading)
---

Prepend my **immediately preceding assistant response** in this conversation to `claude-answers.md` in the repository root. Rules:

- Prepend it as a NEW section at the **top** of the file (newest entries first), directly below the YAML frontmatter and the `# Claude — Q&A scratchpad` title block, above all existing entries.
- Place a horizontal rule `---` after the new section to separate it from the entry below it.
- Start the section with a heading: `## [YYYY-MM-DD] <short descriptive title>` using today's date. If the user supplied text in `$ARGUMENTS`, use that as the title; otherwise infer a concise one from the response.
- **Preserve all math verbatim**: keep `$...$` (inline) and `$$...$$` (block) delimiters exactly as written so Obsidian MathJax renders them. Convert any stray Unicode math (e.g. `τ`, `√`, `≤`) into LaTeX.
- Copy the substantive content of the last response **faithfully** — do not summarize, shorten, or re-derive. Drop only conversational filler (e.g. a trailing "want me to file this?" offer).
- If the last response referenced a source, include a short `**Source:**` line near the top of the section when one is evident.
- Use the Edit tool to prepend (read the file first if needed); do not overwrite existing entries.
- Do NOT touch `wiki/index.md` or `wiki/log.md` — this is a plain scratchpad, not the wiki.
- After writing, confirm in one line what was added (heading + approx length).

Optional title from the user: $ARGUMENTS
