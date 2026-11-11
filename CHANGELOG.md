# Changelog

## 1.7.3 — 2026-11-11

Fixes prefix-cache misses in long conversations while subagents run, and shows machine load on the dashboard. Engine output unchanged: default and INT8 outputs match the 1.7.2 references bit for bit, and cached / concurrent output still matches running alone (cache fuzz 0/40 mismatches).

### Prefix cache
- Each conversation keeps a limited number of state snapshots. Switching between batched and solo speculative decoding saved snapshots that cannot be resumed by another request, and these evicted the aligned checkpoint at the prompt end; a 166K-token Claude Code conversation then re-read its whole prompt (≈4 min) on the next turn.
- Snapshots that cannot be resumed across requests are now evicted first, the longest resumable one is always kept, and each slot keeps at least 4 snapshots (was 3).
- Repro (`bench/snap_evict.py`: a 49K-token conversation generating while 7 subagent requests run, then its next turn): 1.7.2 reused **0** tokens and re-read 50.8K in 18.5 s; 1.7.3 reuses **49,216** and answers in 0.7 s. Same result with the snapshot limit pinned to 3, so the eviction order alone fixes it.
- 1.7.1 SSD cache files (old format) are ignored since 1.7.2 and can be deleted.

### `/status`
- New `sys` object: `cpu_pct`, `gpu_pct`, `ram_used_gb` / `ram_total_gb` (Activity Monitor "Memory Used" and physical memory), `proc_gb` (server footprint), `disk_free_gb` / `disk_total_gb` (volume of `--cache-dir`) and `cache_gb` (size of the cache dir, recomputed at most every 30 s). GB = 10⁹ bytes; fields are `null` when unavailable.

### Dashboard
- A cache hit card (this run / all time) replaces "tokens left".
- Active rows read "Reading prompt · 110K / 167K read · about 50s left", with a cached-tokens column per conversation.
- Output shows 0 while nothing is generating.
- A system strip shows CPU, GPU, RAM used / total (GiB, like Activity Monitor) with Qwental's own footprint, and free disk on the cache volume with the cache size; hidden when the server does not report `sys`.

## 1.7.2 — 2026-11-11

Fixes answers that went off track in long Claude Code sessions, and a simpler dashboard. On the original IQ3_S GGUF, default, INT8 and speculative outputs are bit-identical to 1.7.1 (and so to 1.0.0); concurrent output is bit-identical to running alone.

### Reused state is now bit-identical to computing from scratch
- A request that resumed from cached state at a position that was not a multiple of 64 (the DeltaNet chunk size) — a prompt end, the previous turn's finish, a snapshot left after idle pruning — or from state produced by decoding, got last-position logits that differ from one-shot prefill by KL 0.01–0.43. Greedy then picked a different first token at near-ties, and long sessions drifted into "answering a different question".
- Each snapshot now records whether it is exact (reachable by aligned prefill from 0). Other requests resume only from exact snapshots at multiples of 64; the same request continuing its own generation still resumes where it stopped. Prefill also saves a checkpoint at the prompt end rounded down to 64, so a follow-up turn recomputes at most 63 tokens plus the previous answer (stress test: 313,876 → 312,178 tokens prefilled, 864 → 854 s).
- The SSD cache format is now QWC2, keyed by model and INT8 fingerprint; older files are ignored.
- A request running alone with speculative decoding now stops thinking exactly at the thinking cap (it emitted the rest of the accepted round, so the wrap-up came later than in batched decoding).
- `bench/cache_fuzz.py` drives a Claude-Code-like load (growing main conversation, subagents sharing prefixes, cancellations, memory pressure) and compares every answer with running it alone without reuse: before 8/40 and 11/23 (SC117 + `--int8`) answers differed, after 0/40 and 0/23. `QW_VERIFY_REUSE=N` makes the server recompute 1 in N reused prompts when idle and log `[verify] OK / MISMATCH`.

### Dashboard
- The main view is four numbers (output tok/s, conversations generating / reading, prompt tokens left to read with an ETA, memory as OK / Tight / Low), one 5-minute chart, a one-line list per conversation and one cost line. Totals, measured rates, cost details and usage by period are under "Details".

## 1.7.1 — 2026-11-10

Fixes the memory leak that made the server die every 15–20 minutes under Claude Code with many subagents. Engine output unchanged.

- The scheduler thread had no autorelease pool, so every command buffer and encoder lived until the thread ended, and GPU memory bound to them was not returned after `qw_buf_free`. The process footprint grew ~10–15 GB per minute (to 230 GB) until macOS killed it with SIGKILL; freeing or shrinking slots did not lower it. Each scheduler round now runs in its own pool, and buffer creation / release and command-buffer creation are wrapped too.
- Same load (8 clients, slot budget 8 GB, `--int8`), footprint after 255 s: 71 GB and climbing → 41 GB and flat; it now drops when slots are freed. Not specific to `--int8`; slot churn (subagents, shared-prefix copies, KV shrinking) made it worse.

## 1.7.0 — 2026-11-10

Subagents launched together no longer prefill the same prefix again and again, and a server that dies leaves a reason in the log. Engine output unchanged: on the original IQ3_S GGUF, default, INT8 and speculative outputs are bit-identical to 1.6.1 (and so to 1.0.0); concurrent output is bit-identical to running alone.

### Shared prefixes among concurrent prompts
- When prompts in sliced prefill share at least 2,048 tokens (`QW_COAL_MIN`, 0 disables), the shared part is prefilled once: a leader stops at the shared boundary and saves a snapshot there, followers copy it and prefill only their own suffix. Outputs are bit-identical to prefilling each alone.
- 8 prompts sent together sharing a 20.5K-token prefix: 104.2 s → 37.7 s; 7 × 20,544 tokens copied instead of recomputed.
- Admission control subtracts the prefix shared with prompts in flight, so followers are not refused with 529.
- `/status`: `coalesced_tokens`, `coalesced_requests`, `active[].follows`.

### Diagnostics
- On SIGSEGV / SIGBUS / SIGILL / SIGFPE / SIGABRT / SIGTRAP / SIGTERM / SIGINT / SIGHUP / SIGXCPU / SIGXFSZ and on exit, the server writes a last line to its log with the time, signal details, its memory footprint and the system memory level; fatal signals add a stack trace and still produce the OS crash report. The footprint and memory level are also logged when they change notably and at every memory action. A SIGKILL (e.g. from the OS under memory pressure) cannot be caught.

### Dashboard
- The output rate no longer spikes when a request finishes: finished output now moves from the active list to the totals atomically.

### Experimental, off by default
- `QW_PAR_MIX=1`: run a prefill segment and the batched decode of other conversations at the same time on two GPU queues (bit-identical results). On the M5 Max the GPU barely overlaps them (decode made 2.2 steps/s during a prefill, prefill slowed 15%), so the default alternation stays.

## 1.6.1 — 2026-11-10

Dashboard wording and layout only.

- The system memory line reads "Free memory 31% · swap 5.1 GB" (was "System memory level 31", which wrapped and did not say what the number is); when memory is low it adds "· low" and turns red, with the explanation in the tooltip.
- Chinese card sub-lines no longer break in the middle of a word.

## 1.6.0 — 2026-11-10

Faster under Claude Code with many subagents, from an analysis of a day of real use. Engine output unchanged: on the original IQ3_S GGUF, default, INT8 and speculative outputs are bit-identical to 1.5.4 (and so to 1.0.0); concurrent output is bit-identical to running alone.

### Less recomputation
- A prompt cancelled during sliced prefill keeps its progress: a snapshot is saved at the segment boundary and the retry resumes there with bit-identical output. In 1.5.1, 13 long prompts were cancelled and retried 10 times each from zero (5.8M tokens recomputed).
- Admission control: a request whose prompt processing would wait more than 90 s behind others gets 529 with `retry-after` instead of timing out after minutes (`QW_MAX_WAIT_S`); a prompt alone is always accepted, and a prompt refused for long enough is accepted.
- More persist points: where the tool list diverges from other recent requests, and after the leading system-reminder blocks of the first user message (sibling subagents of one kind share up to there).

### Memory
- KV capacity grows by a quarter (at least 4,096 tokens) instead of doubling; idle slots shrink to what they hold after 60 s and keep their two longest snapshots. Same scenario (60K-token main conversation + 3 subagents): slot memory 17.8 → 13.0 GB; a 33K-token conversation ≈ 2.8 → 1.6 GB.
- Slots of 4,096 tokens or more are written to the SSD cache before eviction (was 16,384).
- The slot budget leaves 24 GB outside (was 16 GB): with 16 GB the machine filled 79 of 80 GB of swap. Under system memory pressure (`QW_MEM_LEVEL_MIN`, default 20%, was 10%) idle slots are shrunk, then evicted.

### Scheduling
- Prefill segments target 200 ms and decoding conversations get at least as long between segments (was 500 ms vs 200 ms, so decode got at most 29% while anyone prefilled).

### Observability
- `/status`: `mem_level`, `swap_used_gb`, `rejected`, `queue_reason` (`limit` / `busy`), `live_output_tokens`, `prefill_backlog_tokens`, `rate.prefill`, `active[].prefill_left`; `active[].generated` now updates live.
- Log lines add thinking tokens, queue time and prefill compute time.
- Dashboard: output tok/s is computed from tokens as they are generated (was from completed requests only, which showed 0 and then 1,000+ spikes); the queue card states the real reason; system memory level and swap are shown and turn red when low; prefill progress per conversation.

## 1.5.4 — 2026-11-10

Usage that survives restarts, and API pricing on the dashboard. Engine output unchanged.

- Totals are kept across restarts in `<cache-dir>/stats.json` (`--stats-file PATH` to move it, `""` to disable): requests, prompt / cached / output tokens, cancellations, errors and uptime, plus one bucket per local day (kept 400 days). Saved atomically every 30 s and on SIGTERM / SIGINT; a corrupt file is renamed to `.bad`.
- `GET /status` adds a `lifetime` object; `/status?days=N` includes the last N daily buckets.
- Dashboard: "API cost" priced at OpenRouter `qwen/qwen3.8-flash` rates ($0.15 input, $0.016 cache read, $0.47 output per 1M tokens; editable and remembered), for this run and all time with the cost per hour; totals switch between this run and all time; usage by day, week (Monday first) or month with a cost chart and the current period highlighted.

## 1.5.3 — 2026-11-10

Dashboard layout only.

- Measured-rate labels no longer wrap (`batch ×10` and up), card hints and sub-lines wrap evenly instead of leaving one character on a line or being cut off with "…".

## 1.5.2 — 2026-11-10

Dashboard only; engine and server behavior unchanged.

- `/dashboard` is localized in English and Traditional Chinese: English by default, Chinese when the browser prefers it, an EN / 中文 toggle that is remembered, and `?lang=en|zh-TW`. Every label is in one language; numbers and times follow the locale.
- The totals panel shows the sums directly: total (prompt + output) and GPU computed (prefill + output).

## 1.5.1 — 2026-11-10

Claude Code works fully, also with subagents. Engine output is unchanged: on the original IQ3_S GGUF, default, INT8 and speculative outputs are bit-identical to 1.5.0 (and so to 1.0.0); concurrent output is still bit-identical to running each request alone.

### Requests as Claude Code sends them
Checked against 62 captured Claude Code 2.1.322 requests: the rendered prompt matches the official chat template byte for byte.
- `role: "system"` messages inside `messages` (Claude Code's environment, language setting, date and deferred-tool list) were dropped; leading ones now join the system prompt and later ones become user turns, as in Strata.
- Thinking effort follows the template: `high` / `xhigh` / `max` → xhigh (was medium), `budget_tokens` maps to an effort level, `none` / `off` disable thinking. Requests that do not ask for thinking skip it (`--anthropic-thinking on_request`, the default; `always` keeps the old behavior).
- The volatile `x-anthropic-billing-header` stamp is pinned, so the system prefix stays cacheable.
- Special-token literals in content (`<|im_end|>`, `</think>`, `<tool_call>` … in files or tool output) are tokenized as plain text instead of control tokens.
- Output parser rewritten after Strata's: `</parameter>` inside a value no longer truncates it, several calls in one `<tool_call>`, calls written inside thinking or without `<tool_call>` are recovered, mangled tool names are mapped to declared tools, no call is dropped silently, and a call cut off by `max_tokens` is returned as text instead of a partial `tool_use`.
- Completed empty assistant turns are skipped; `defer_loading` tools are left out of the tool list and `tool_reference` expands them; tool order is kept stable; images and binary documents get a clear note.

### Server
- Writes never block the engine: each connection has its own output buffer and a 30 s send timeout, so a stopped client (Ctrl-Z) no longer freezes every conversation.
- Keepalive `ping` every 10 s while a request is queued, prefilling or emitting a tool call.
- Cancellation: a closed connection is detected within 0.5 s, its request stops and its slot is released.
- Engine failures are reported (SSE `error` / HTTP 500 / 529) instead of an empty `end_turn`; more than 64 queued requests get 529.
- `usage` reports `cache_read_input_tokens` / `cache_creation_input_tokens` as Strata does.
- Thinking leaves room for the answer: with `max_tokens` ≥ 1024 thinking is wrapped up before the last quarter; a turn that stops inside thinking is closed and continued.
- `stop_sequences` match only answer text and are escaped in `message_delta`; `count_tokens` takes the tokenizer lock; request size limits (413 / 431 / 408); correct status texts and error types; `/v1/v1/…` alias; the requested model name is echoed.
- Scheduler: with 15 subagents active, 1.5.0 kept time-slicing speculative decoding and every request got 0.2–0.3 tok/s. More than 3 conversations now always decode together, a time slice lasts at least 250 ms, the time-sliced rate is measured on the wall clock including switches, and a mode is kept at least 5 s.
- Slots: a subagent sharing the system and tools prefix no longer overwrites the main conversation's slot, and memory pressure evicts by tokens held and idle time instead of pure LRU (long conversations are written to the SSD cache first). Main conversation of 60K tokens with 3 subagents: next turn 33.4 s → 0.6 s.
- `GET /dashboard` (and `/`): a live web page on `/status`; `/status` lists each active conversation with id, prefill progress, tok/s, age and model name, plus cancelled and error counts.

### Tests
- `build/chattest` (parser, effort mapping, special-token escaping), `tools/chat_template_diff.py` (rendering against the official template), `tests/serve_e2e.sh` (18 server checks), `bench/slot_keep.py` (main conversation kept while subagents run).

## 1.5.0 — 2026-11-09

Runs the abliterated SC117 IQ3_S GGUF ([`SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF`](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)) at full speed, concurrency included. On the original IQ3_S GGUF nothing changes: default, INT8 and speculative outputs are bit-identical to 1.4.0 (and so to 1.0.0).

### SC117
- SC117 is the ISTA-DASLab GSQ-RCO quantization with the abliterated write-to-residual projections transplanted in; 96 of them are Q8_0 (`ssm_out`, `attn_output`, `ffn_down_shexp`). The fused DeltaNet output projection did not accept Q8_0, so a lone request fell back to the generic path and concurrent decoding failed at warmup. `dn_proj` now has Q8_0 decoders (2-byte aligned loads) for the output projection.
- Against Strata on the same file, two alternating rounds (tok/s, Strata → Qwental): decode prose 92.9 → 100.9, new code 131.0 → 138.5, code edits 170.8 → 198.8; after 10.7K / 47K 90.7 / 85.1 → 98.2 / 87.9; prefill 10.7K 1330 → 1629, 47K 1421 → 1613.
- Concurrent output is bit-identical to running each request alone (1, 4, 8 and 16 at once); speculative output equals greedy. 8 at once: 162.8 tok/s total.
- Greedy text against Strata on SC117: identical on the code-edit prompt; on the other two it diverges at a near-tie or where Qwental picks Strata's own top-1 token.
- `--int8` on SC117 (alternating runs, same session): prefill 10.7K / 47K 1949 / 1860 → 2802 / 2815 tok/s, decode unchanged; against the same build without INT8, top-1 matches 99.0% (100 positions after 4K) and 99.3% (300 after 34K), mean KL 0.0024 / 0.0219.
- The attention output projection stays on the generic matvec for Q8_0: a fused Q8_0 variant measured 12.44 → 12.43 ms per token and slower prefill.

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
- Homebrew: `brew tap hy-ong/qwental https://github.com/hy-ong/qwental` then `brew install hy-ong/qwental/qwental` (formula in `Formula/`; builds from source).

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
