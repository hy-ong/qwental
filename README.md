# Qwental

**English** | [繁體中文](README.zh-TW.md)

A Metal inference engine for **Qwen3.8-Flash-Next** on an **Apple M5 Max (128 GB)**, tuned toward the hardware limits. It is written in C23 and MSL (Metal 4 tensor API) and has no framework dependencies.

## Performance

Qwental runs Strata's default model file, the ISTA-DASLab GSQ-RCO **IQ3_S GGUF**, unchanged: same weights, FP16 compute. Measured against Strata on the same machine (`bench/vs_iq3s.sh 3`, medians of three alternating rounds, tokens/s):

| | Strata | Qwental |
|---|---|---|
| Decode: prose | 84.0 | **110.5** (+31.5%) |
| Decode: new code | 123.1 | **126.3** (+2.6%) |
| Decode: code edits | 148.2 | **173.9** (+17.3%) |
| Decode after a 10.7K prompt | 79.7 | **93.0** (+16.7%) |
| Decode after a 47K prompt | 77.9 | **84.3** (+8.2%) |
| Prefill, 10.7K / 47K | 1217 / 1231 | **1519 / 1520** (+24%) |

The output is the same model: over 600 positions of Strata's own output, Qwental's top-1 token matches Strata's every time (`bench/vs_strata_*.py`). Speculative decoding output is identical to greedy output.

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

## Features

- Loads GGUF quantized weights directly (IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, Q4_K, Q5_K, Q6_K, Q8_0, Q5_1, BF16), bit-exact with the reference dequantization
- Speculative decoding with a DFlash2 block drafter and prompt-lookup drafts
- Batched prefill with sparse attention for long contexts

## Limits

- One model, one machine (M5 Max), one request at a time
- The MLX model directory is still required for the tokenizer

## License

MIT, see `LICENSE`. Third-party sources are listed in `THIRD_PARTY_NOTICES.md`. Model weights are not included.
