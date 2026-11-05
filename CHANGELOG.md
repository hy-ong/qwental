# Changelog

## 1.1.0 — 2026-11-06

INT8 prefill behind flags, faster long prompts, and a same-weights comparison against oMLX. Without the new flags the output is bit-identical to 1.0.0 on both the GGUF and the MLX profile.

Same-weights results (two alternating rounds, mean): against Strata, decode +3–17%, prefill 10.7K / 47K 1473 / 1461 → 1728 / 1747 tok/s by default and 2750 / 2541 with `--int8` (2671 at 47K with `--int8-qk`); against oMLX 0.7.4.dev1 on the MLX weights, decode +33–72%, prefill 1884 / 2072 → 2456 / 2526 with `--int8`.

### INT8 prefill (opt-in)
- `--int8` / `QW_INT8=1`: routed experts run INT8 × INT8 (IQ blocks decoded straight to int8 with per-128-K rescaling on GGUF, per-slice requantization on MLX); on GGUF, dense DeltaNet, attention and shared-expert projections also use a per-row INT8 copy made at load time (+3 GB). Decode is unchanged.
- `--int8-qk`: also runs the attention indexer scores and the sparse-attention Q·Kᵀ in INT8 (attention per layer −9% at 4K, −25% at 45K).
- `--int8-hyper` (GGUF, experimental): also runs the hyper-connection mix_down in INT8 (about 3% faster).
- `QW_INT8` also takes a category list: `lin,attn,shared,moe,qk,hyper`.
- Quality on 300 scored positions after 8K–32K prompts (code, English, Chinese), vs the same build without INT8: `--int8` top-1 97.9%, KL 0.006–0.023, perplexity −1.4% to +0.7%; `--int8-qk` 98.0%, 0.005–0.018, −1.2% to +1.3%; `--int8-hyper` 96.7%, up to +7.6% perplexity on English. For scale, the MLX 4/8-bit weights vs the IQ3_S GGUF: 93.3%, KL 0.038–0.141.
- Kernels: each activation is quantized once, the GGUF INT8 copy is stored [K][N] with one K run per tile (dense INT8 GEMM about 89 → 110 TOPS), IQ2_S grids are read from constant memory, and expert blocks are decoded two units per thread.

### Other changes
- `--fast-head` (GGUF): Q6_K lm_head on the matrix units with FP16-rounded weights; verify of 8 tokens 3.5 → 1.0 ms. Outputs can differ from the default at near-ties.
- Long prompts: the next segment's n-gram rows are prefetched while the GPU runs, and one RoPE table is shared per segment (44K cold prompt 54.2 → 30.9 s, bit-identical).
- Verify reads the PLE key/value projection weights once per round; the DFlash2 draft head uses the Q6_K matrix-unit kernel; the hyper-connection input kernel runs one threadgroup per token in GGUF prefill (all bit-identical).
- Speculative decoding skips the DFlash2 draft while copying a long prompt-lookup match (code/edit 2–4% faster). `QW_SPEC_STATS` prints acceptance statistics.
- CPU dequantization for dense GGUF formats (`src/ggml_dq.c`, checked with `tools/dqtest.py`).
- `QW_PPL_PF` and `tools/logit_cmp.py` measure how the prefill path affects later logits.

### Known limits
- On the MLX profile, speculative output can still differ from greedy at near-ties (as in 1.0.0). The GGUF profile is exact.

## 1.0.0 — 2026-11-05

First release: a dedicated runtime that reads Strata's default model file (ISTA-DASLab GSQ-RCO IQ3_S GGUF of Qwen3.8-Flash-Next) unchanged and runs it faster than Strata on the same M5 Max 128GB, with the same weights and FP16 compute.

Same-weights comparison against Strata (three alternating rounds, medians, tok/s): prose decode 110.5 vs 84.0, new code 126.3 vs 123.1, code edits 173.9 vs 148.2, decode after 10.7K 93.0 vs 79.7, after 47K 84.3 vs 77.9, prefill 10.7K 1519 vs 1217, 47K 1520 vs 1231. Qwental's top-1 matches Strata's at every one of 600 positions of Strata's own outputs.

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
