---
name: explain-layer
description: Explains a neural network layer in simple terms with a diagram. Use when someone asks how a layer works (e.g. Affine, ReLU, Softmax, Convolution).
argument-hint: [layer-name]
---

Explain the $ARGUMENTS neural network layer from this codebase.

Steps:
1. Find the layer's source file using Glob or Grep
2. Read the forward() and backward() implementation
3. Explain with:
   - One everyday analogy (what it "does" in plain English)
   - ASCII diagram of data flow
   - Key equation in plain text (no LaTeX)
   - One common mistake beginners make
