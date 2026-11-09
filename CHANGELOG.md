# Changelog

## 1.4.0 — 2026-11-09

A long prompt no longer holds up requests that arrive after it, and the server reports its state at `GET /status`. Without flags a single request's output is bit-identical to 1.0.0 on both the GGUF and the MLX profile, and concurrent GGUF output is bit-identical to running each request alone, as in 1.3.0.

### Long prompts yield to new requests
- In 1.3.0 a long prompt that arrived while nothing else ran was prefilled in one pass, and requests arriving meanwhile waited for it (about 19 s for a 30K-token prompt). Prefill now runs in segments of 4096 tokens (the same segments as one pass, so bit-identical) and yields between segments when a request is waiting; the rest of that prompt continues in slices alongside the new request.
- A 30K-token prompt alone, a short request sent 2 s later: its first token after 0.38 s. Both requests' outputs are unchanged (GGUF and MLX).
- Sliced prefill now serves the request with the fewest remaining prompt tokens first.

### Server
- `GET /status`: version, profile, uptime, totals of requests and of prompt, reused and generated tokens, and the queue length; with concurrency also the current mode, slot memory against its cap, the active requests and the measured rates.
- `--help`.
- `make install [PREFIX=/usr/local]` installs the single binary (the Metal kernels are embedded, so it runs from any directory).
- Homebrew: `brew install hy-ong/tap/qwental` (tap [hy-ong/homebrew-tap](https://github.com/hy-ong/homebrew-tap); builds from source).

### Batched speculative decoding (experimental, off by default)
- `QW_PAR_VSPEC=1` (GGUF only) uses speculative decoding for 2–3 concurrent requests. Each conversation keeps its own DFlash2 drafter context; the drafts of several conversations are verified together (up to 8 rows), and every row's logits are bit-identical to running alone (`build/run` `QW_PARV` / `QW_PARV_DF`).
- Measured in alternating runs on a cool machine (total tok/s, 1.3.0 / 1.4.0 default / with `QW_PAR_VSPEC=1`): 2 requests 107.0 / 105.2 / 108.2; 3 requests 126.3 / 123.7 / 127.5; 4 requests 137.3 / 141.8 / 134.3.
- Why it doesn't win: with several conversations already sharing each weight read, a plain batched step costs about 6–9 ms per token, while each extra verified row costs about 3 ms (each token reads its own experts), about 70% of drafts are accepted, and each round adds a drafting and acceptance step.

### Tests and tools
- `build/run`: `QW_PARV=k` (batched verification with oracle drafts that contain deliberate mistakes) and `QW_PARV_DF=1` (with real drafts).
- `bench/concurrent.py`: an `edit` mode (code-edit prompts).

### Known limits
- Total throughput still stops growing past about 8 concurrent requests: each group of 8 reads the weights again. Running more than 8 rows at once would save little: from the measured cost per row (about 2.9 ms), 16 rows would take about 55 ms against 64 ms in two groups (about 14% less), not worth rewriting every 8-row kernel.
- With `QW_GGUF_PARTS` selecting only some parts, requests are served one at a time.
- MLX concurrent output can differ from run to run (batched rounding, as in 1.3.0).

## 1.3.0 — 2026-11-09

Concurrent requests: several agents (for example Claude Code subagents) are served at once instead of one after another. A lone request runs as before; without flags its output is bit-identical to 1.0.0 on both the GGUF and the MLX profile.

Same IQ3_S GGUF, 400-token requests sent together (`bench/concurrent.py`, total tok/s, 1.2.0 → 1.3.0): one at a time 110.9 → 110.1; 4 at once 110.3 → 146.7 (+33%); 8 at once 107.9 → 172.3 (+60%); 16 at once 109.0 → 170.9 (+57%). MLX weights, 8 at once: 81.8 → 135.8 (+66%).

### Concurrency (on by default)
- Up to 16 requests at once (`--parallel N` sets the bound, up to 64; `--parallel 1` serves one at a time). Each conversation has its own KV slot and DeltaNet state; the weight-shared parts (embedding, projections, MoE, lm_head) run for all conversations in one pass, the attention core and DeltaNet recurrence per conversation.
- The scheduler works from measured speeds: a lone request uses speculative decoding; with more, it compares time-sliced speculation against batched decoding for the current count and uses the faster one. Batches over 8 are split into groups by the measured step time of each group size.
- GGUF: each conversation's output is bit-identical to running it alone (per-step logits for 1–16 conversations and 4K contexts; staggered, streaming, disconnect and load tests). MLX: batched decoding rounds slightly differently, the same kind of difference MLX's default speculative decoding already has; with `--exact-spec` it is bit-identical.
- Long prompts no longer pause the others: while other conversations are generating, a new prompt is prefilled in slices of about 500 ms (`QW_PAR_CHUNK_MS`), bit-identical to one pass. A 30K-token prompt next to running streams: longest pause 16.8 → 0.58 s; the prompt itself finishes in 32.5 s instead of 21.5 s.
- Prefixes are shared across slots: a request copies a longer prefix (system prompt, tool list) from another conversation's slot, even a busy one (17K tokens in about 0.1 s instead of an 8 s prefill). Batched prefill also writes system-prompt snapshots to the SSD cache.
- Memory guard: slot memory (KV is about 33 KB per token) is capped at physical memory minus the weight files minus 16 GB (`QW_SLOT_MEM_GB` sets it); the least recently used idle slots are freed first, and one more each second while the system's available memory is under 10%.
- Startup takes about 1.5–2 s longer (kernel warmup and step timing for batched decoding).

### Tests and tools
- `build/run`: `QW_PAR` (per-step logits of several prompts decoded together vs alone), `QW_PAR_TR` (per layer), `QW_PF_CHUNK` (sliced vs single prefill), `QW_MTP_CHAIN` (chained MTP draft acceptance, measured only) and `QW_VB_BENCH` (verify-round cost per kernel).
- A 15-minute run of 6 simulated agents (shared 17K-token system prompt, multi-turn, streaming, disconnects, temperature 0 and 0.7): 591 requests without errors, RSS steady around 27 GB. Claude Code with three parallel subagents against the server.

### Known limits
- Total throughput stops growing past about 8 concurrent requests: each group of 8 reads the weights again.
- With `QW_GGUF_PARTS` selecting only some parts, requests are served one at a time.
- A long prompt that arrives while nothing else runs is prefilled in one pass; requests arriving meanwhile wait for it.

## 1.2.0 — 2026-11-07

Runs from a single GGUF, and the last known 1.0 issue has an opt-in fix. Without the new flags the output is bit-identical to 1.0.0 on both the GGUF and the MLX profile.

Against 1.1.0 on the same IQ3_S GGUF (`bench/vs_server.py`, two alternating rounds, mean tok/s): prefill 10.7K / 48K with `--int8-qk` 2580 / 2700 → 2701 / 2801 (+4–5%); default and `--int8` within ±3%; decode unchanged. Profiling this round found prefill GEMMs near the matrix-unit limits (INT8 about 110 of 121 TOPS) and decode reading at or near memory bandwidth, so further bit-identical gains are small.

- No MLX model directory needed with `--gguf`: the tokenizer comes from the GGUF metadata (token ids identical to `tokenizer.json` on code, English, Chinese, emoji, special tokens and invalid UTF-8), the three PLE constant tensors from the GGUF's `ple.*` keys, and `dflash2/` is looked up next to the GGUF directory. With an MLX directory present, behaviour is unchanged.
- `--exact-spec` / `QW_EXACT_SPEC=1` (MLX weights): speculative decoding output equals greedy (safe-math MLX decode/verify kernels, one chunk sum shared by qmv/qmvb, verify uses decode's fused norm-gate/attention gate). MLX outputs change slightly (q300 PPL 1.337 → 1.340); decode 1–8% slower. GGUF is already exact.
- With an INT8 flag, startup also compiles the INT8 prefill kernels (2304-token and 64-token warmup prefills): first 12.7K-token request TTFT 5.4 → 4.3 s on GGUF `--int8`, at 0.9–1.8 s more startup (about 5 s on MLX). `QW_NO_I8_WARM=1` skips it.
- Prefill DeltaNet: the chunked (WY) prep and scan kernels give each SIMD group its own tile instead of one 8-group matmul; recurrence 3.39 → 2.50 ms per layer, 4096-token `--int8` prefill about 3% faster. Bit-identical; `QW_GDN_CHUNK=0` restores the old kernels.
- Prefill DeltaNet input step (conv + normalize) runs 16 consecutive tokens per threadgroup and reads each input once: 0.61 → 0.34 ms per layer. Bit-identical (explicit fma chain); `QW_GDN_TB=0` restores the old kernel.
- Verify of 5–8 tokens scores the attention indexer with one reduce-scatter per block instead of 32 `simd_sum`s (bit-identical, checked against the old kernel at startup; `QW_SCORE3=1` forces the old one).

### Known limits
- With `QW_GGUF_PARTS` selecting only some parts, the remaining weights still come from the MLX directory.

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
