---
name: typeless-link
description: Creates a plain Markdown link to any resource in this deep learning codebase — notebook, chapter, or source file — without specifying a link type. Use when you need a quick clickable reference to a notebook chapter, module, or concept.
argument-hint: [chapter-number or topic or filename]
---

Create a plain Markdown link to $ARGUMENTS in this deep learning project.

Steps:
1. Search for matching files using Glob with patterns like `**/*$ARGUMENTS*` or `notebooks/ch$ARGUMENTS.ipynb`
2. For notebooks, prefer paths under `notebooks/` (e.g., `notebooks/ch03.ipynb`)
3. For Python source files, use the relative path (e.g., `common/layers.py`)
4. Return the link in plain Markdown: `[label](path)` — no type prefix, no role syntax
5. If multiple matches exist, list all relevant links with concise labels
6. Use descriptive labels based on chapter topic or filename, not the raw path
