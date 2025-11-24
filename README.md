# RNA-Design-LM
LLM-RNA-Design is a research codebase for designing RNA sequences with autoregressive language models. Instead of solving each RNA inverse-folding instance from scratch with combinatorial search, we train a conditional LM to map target secondary structures (dot–bracket strings) directly to RNA sequences, combining supervised learning on solver-generated structure–sequence pairs with reinforcement learning that optimizes thermodynamic folding metrics such as Boltzmann probability, ensemble defect, and MFE uniqueness. Constrained decoding enforces base-pairing rules during generation so that all sampled sequences are structurally valid by construction, enabling fast, amortized RNA design at scale.

## 1. Install

## 2. Constrained Decoding
This script runs batched inference with an RNA LM on a JSONL test set of target structures, optionally using C++-accelerated constrained decoding (via prefix_allowed_tokens_fn) to enforce Watson–Crick–wobble pairing. It also supports resuming from an existing output file so you don’t waste samples on IDs that are already complete.

### What it does
Reads a JSONL test file with at least:
id: unique identifier for each target
target_structure: dot–bracket string

Loads a decoder-only LM from Hugging Face (SL or SL+RL flavor).

Builds prompts of the form “structure as chat message” → generate nucleotides.

Optionally enforces base-pair constraints during decoding:

Unpaired / opening ( → any of {A,C,G,U}

Closing ) → only nucleotides compatible with its partner (CG, GC, AU, UA, GU, UG)

Writes a JSONL output file with:

id

target_structure

designed_sequence (A/C/G/U string)

time (average seconds per sample in that batch)
