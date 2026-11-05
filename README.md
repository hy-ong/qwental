# Qwental

**English** | [繁體中文](README.zh-TW.md)

A Metal inference engine for **Qwen3.8-Flash-Next** on an **Apple M5 Max (128 GB)**, tuned toward the hardware limits. It is written in C23 and MSL (Metal 4 tensor API) and has no framework dependencies.

## Performance

Qwental runs Strata's default model file, the ISTA-DASLab GSQ-RCO **IQ3_S GGUF**, unchanged: same weights, FP16 compute. Measured against Strata on the same machine (`bench/vs_server.py`, two alternating rounds, mean tokens/s):

| | Strata | Qwental | Qwental `--int8` |
|---|---|---|---|
| Decode: prose | 97.2 | **112.5** (+16%) | |
| Decode: new code | 142.3 | **153.8** (+8%) | |
| Decode: code edits | 173.3 | **203.5** (+17%) | |
| Decode after a 10.7K prompt | 93.9 | **101.3** (+8%) | |
| Decode after a 47K prompt | 85.9 | **88.3** (+3%) | |
| Prefill 10.7K | 1473 | 1728 (+17%) | **2750** (+87%) |
| Prefill 47K | 1461 | 1747 (+20%) | 2541 (+74%); `--int8-qk` **2671** (+83%) |

`--int8` does not change decode. Without INT8 flags the output is the same model: over 600 positions of Strata's own output, Qwental's top-1 token matches Strata's every time (`bench/vs_strata_*.py`), and speculative decoding output is identical to greedy output.

Against oMLX 0.7.4 on the same MLX weights (mixed 4/8-bit, Qwental without `--gguf`):

| | oMLX | Qwental | Qwental `--int8` / `--int8-qk` |
|---|---|---|---|
| Decode: prose / new code / code edits | 53.4 / 78.4 / 136.4 | **91.8 / 131.9 / 181.5** | |
| Decode after 10.7K / 47K | 53.8 / 55.3 | **83.0 / 81.1** | |
| Prefill 10.7K | 1884 | 1950 | 2456 / **2480** |
| Prefill 47K | 2072 | 2015 | 2526 / **2611** |

### What INT8 costs

The INT8 flags change prefill numerics. Measured on 300 scored positions after an 8K–32K prompt, for code, English and Chinese text (`QW_PPL_PF`, `tools/logit_cmp.py`), against the same build without INT8 (GGUF; on the MLX weights `--int8` keeps top-1 at 96.3%, KL 0.008–0.039):

| | top-1 same | mean KL | perplexity |
|---|---|---|---|
| `--int8` | 97.9% | 0.006–0.023 | −1.4% to +0.7% |
| `--int8-qk` | 98.0% | 0.005–0.018 | −1.2% to +1.3% |
| `--int8-hyper` | 96.7% | 0.006–0.066 | up to +7.6% (English text) |
| For scale: MLX 4/8-bit vs the IQ3_S GGUF | 93.3% | 0.038–0.141 | −5.0% to +7.1% |

`--int8` and `--int8-qk` stay well inside the gap between two quantizations of the same model; `--int8-hyper` does not, and is experimental.

## Quick start

Requirements: macOS 27, Xcode 27, M5 Max with 128 GB.

```sh
make build/qwental-serve
mkdir -p models/iq3s-gguf
ln -s /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000?-of-00002.gguf models/iq3s-gguf/
./build/qwental-serve --gguf models/iq3s-gguf --port 8080
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=dummy claude --model qwen3.8-flash-next
```

Model files in `models/`:
- the IQ3_S GGUF (two shards of `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`)
- `Qwen3.8-Flash-Next-mixed-4-8bit/` from [`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit) (tokenizer and config)
- `dflash2/` with the DFlash2 drafter pack (`…q4_0-selq4.pack`)

## Server

- Anthropic Messages API: `/v1/messages` (streaming, thinking, tool use) and `/v1/messages/count_tokens`
- OpenAI Chat Completions: `/v1/chat/completions`
- Prefix cache with multiple KV slots and SSD-backed snapshots
- Use Claude Code or any Anthropic/OpenAI client as the front end
- `--int8`: INT8 prefill for expert matrices (and dense matrices with `--gguf`), about 25–60% faster on long prompts. Decode is unchanged; outputs differ slightly from the default (see "What INT8 costs"). With `--gguf` it uses 3 GB more memory.
- `--int8-qk` / `--int8-hyper`: on top of `--int8`, also run the attention Q·Kᵀ (`qk`) or the hyper-connection mix_down (`hyper`, `--gguf` only) in INT8. Each implies `--int8`. `--int8-qk` helps mostly on long prompts at about the same quality as `--int8`; `--int8-hyper` is experimental (it hurt English text in testing).
- `--fast-head` (with `--gguf`): runs the Q6_K lm_head on the matrix units with FP16-rounded weights, as llama.cpp does. Faster speculative verification; outputs can differ from the default at near-ties.

## Features

- Loads GGUF quantized weights directly (IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, Q4_K, Q5_K, Q6_K, Q8_0, Q5_1, BF16), bit-exact with the reference dequantization
- Speculative decoding with a DFlash2 block drafter and prompt-lookup drafts
- Batched prefill with sparse attention for long contexts

## Limits

- One model, one machine (M5 Max), one request at a time
- The MLX model directory is still required for the tokenizer

## License

MIT, see `LICENSE`. Third-party sources are listed in `THIRD_PARTY_NOTICES.md`. Model weights are not included.
