# RNA-Design-LM
LLM-RNA-Design is a research codebase for designing RNA sequences with autoregressive language models. Instead of solving each RNA inverse-folding instance from scratch with combinatorial search, we train a conditional LM to map target secondary structures (dot–bracket strings) directly to RNA sequences, combining supervised learning on solver-generated structure–sequence pairs with reinforcement learning that optimizes thermodynamic folding metrics such as Boltzmann probability, ensemble defect, and MFE uniqueness. Constrained decoding enforces base-pairing rules during generation so that all sampled sequences are structurally valid by construction, enabling fast, amortized RNA design at scale.

## 1. Install

## 2. Constrained Decoding
This script runs batched inference with an RNA LM on a JSONL test set of target structures, optionally using C++-accelerated constrained decoding (via prefix_allowed_tokens_fn) to enforce Watson–Crick–wobble pairing. It also supports resuming from an existing output file so you don’t waste samples on IDs that are already complete.

### What it does
Reads a JSONL test file with at least: id (unique identifier for each target), target_structure (dot–bracket string). Loads a decoder-only LM from Hugging Face (SL or SL+RL flavor). Builds prompts of the form “structure as chat message” → generate nucleotides. Optionally enforces base-pair constraints during decoding:
Unpaired / opening ( → any of {A,C,G,U} Closing ) → only nucleotides compatible with its partner (CG, GC, AU, UA, GU, UG). Finally writes a JSONL output file with:id, target_structure, designed_sequence (A/C/G/U string) and time (average seconds per sample in that batch)


### Key arguments

#### I/O & model selection

--test_path:
Path to test JSONL (default: ../test/eterna100.jsonl).
Each line should be a JSON dict with id and target_structure.

--output_path:
Where to write generated designs (JSONL).
If empty, a default is derived from test_path, e.g. ../test/eterna100.jsonl → ../eterna100_decoding_results.jsonl

--model_flavor: {sl, slrl} Which trained model flavor to use: sl = supervised-only model, slrl = SL+RL model (default)

--sl_model_path:
Default HF path for the SL model
(default: Milanmg/LLM-RNA-Design-2025/model/SL)

--slrl_model_path:
Default HF path for the SL+RL model
(default: Milanmg/LLM-RNA-Design-2025/model/SL+RL)


#### Sampling / decoding

--n_repeats: Number of samples to generate per id (default: 1000).

--batch_size: Number of structures per generation batch (default: 1024).

--do_sample: If set, use sampling; otherwise defaults to greedy-like decoding.

--temp: Sampling temperature (default: 2.0).

--top_p: Nucleus-sampling top_p (default: 1.0 = no truncation).

--max_decode_tokens: Maximum number of new tokens to generate (default: 512).

#### Constrained decoding & ID subset

--constrained_decode: If set, enables structure-aware constrained decoding that enforces base-pair rules.

--constrained_id: 
Optional list of integer IDs.
If provided, only those IDs are decoded (others are skipped).

#### Resume behavior

--resume_remaining / --no-resume_remaining:

Default: --resume_remaining (True)\
True: read output_path, count existing samples per id, and only generate the remaining repeats.\
False: ignore any existing output and start from scratch (file opened in w mode).\
The script logs how many IDs are fully done, partially done, and not started, then processes the remaining “tasks” in batches.
