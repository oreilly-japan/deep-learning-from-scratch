---
name: summarize-notebook
description: Summarizes a Jupyter notebook's purpose, key concepts, and results. Use when asked to review or understand a notebook.
argument-hint: [path/to/notebook.ipynb]
disable-model-invocation: true
---

Summarize the notebook at $ARGUMENTS.

Read the notebook and produce:
1. **Purpose** (1 sentence): what this notebook demonstrates
2. **Key concepts** (bullet list): ML topics covered
3. **Notable results**: any outputs, accuracy numbers, or conclusions shown
4. **Next steps**: what to read or try next based on this content
