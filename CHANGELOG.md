# Changelog

## 1.0.0 — 2026-11-05

First release: a dedicated runtime that reads Strata's default model file (ISTA-DASLab GSQ-RCO IQ3_S GGUF of Qwen3.8-Flash-Next) unchanged and runs it faster than Strata on the same M5 Max 128GB, with the same weights and FP16 compute.

Same-weights comparison against Strata (three alternating rounds, medians, tok/s): prose decode 110.5 vs 84.0, new code 126.3 vs 123.1, code edits 173.9 vs 148.2, decode after 10.7K 93.0 vs 79.7, after 47K 84.3 vs 77.9, prefill 10.7K 1519 vs 1217, 47K 1520 vs 1231. qwental's top-1 matches Strata's at every one of 600 positions of Strata's own outputs.

### GGUF profile
- `--gguf DIR` / `QW_GGUF`: loads the model's GGUF directly (raw ggml blocks), selectable per part (`QW_GGUF_PARTS`).
- Block decoders for IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, type 42, Q4_K, Q5_K, Q6_K, Q8_0, Q5_1 and BF16, bit-exact against the gguf package (`tools/dqtest.py`).
- Handles the GGUF's tiled DeltaNet V heads, F32 norms, precomputed A, split indexer and a CPU-only 28.8 GB PLE table.

### Decode
- Table-driven IQ MoE kernels with per-expert deduplicated verify; DeltaNet in three dispatches with deferred state writes; shared decode/verify QSA kernels with histogram top-k and a block-list core; fused hyper-connection kernels.
- DFlash2 drafter reading Q4_0 natively on the matrix units (3.3 -> 1.9 ms per round), Strata's selq4 pack, suffix (prompt-lookup) drafts, GPU-side acceptance with one CPU sync per round.
- Verify is bit-identical to single-token decode, so speculative output equals greedy; a whole-model trace (`QW_TRACE`) checks it.

### Prefill
- Token-split sparse attention core (1.7x), strip-tiled IQ MoE GEMM, GGUF DeltaNet projections on qmm_ggm, fused hyper-connection kernels.

### Server
- Anthropic Messages API (streaming, tool use, count_tokens) and OpenAI chat completions; `--version`.

### Known limits
- One model, one machine (M5 Max), one request at a time. The MLX model directory is still needed for the tokenizer.
