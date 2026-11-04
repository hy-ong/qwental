# qwental

[English](README.md) | **繁體中文**

專為 **Apple M5 Max（128 GB）** 上的 **Qwen3.8-Flash-Next** 打造、逼近硬體極限的 Metal 推論 engine。以 C23 與 MSL（Metal 4 tensor API）撰寫，不依賴任何框架。

## 效能

qwental 直接執行 Strata 預設的模型檔 ISTA-DASLab GSQ-RCO **IQ3_S GGUF**，不轉格式：權重相同、同樣以 FP16 運算。在同一台機器上與 Strata 對照（`bench/vs_iq3s.sh 3`，順序交錯三輪取中位數，tok/s）：

| | Strata | qwental |
|---|---|---|
| 解碼：英文文章 | 84.0 | **110.5**（+31.5%） |
| 解碼：寫新程式碼 | 123.1 | **126.3**（+2.6%） |
| 解碼：改程式碼 | 148.2 | **173.9**（+17.3%） |
| 10.7K prompt 之後的解碼 | 79.7 | **93.0**（+16.7%） |
| 47K prompt 之後的解碼 | 77.9 | **84.3**（+8.2%） |
| Prefill 10.7K / 47K | 1217 / 1231 | **1519 / 1520**（+24%） |

輸出是同一個模型：在 Strata 自己生成的 600 個位置上，qwental 的第一名全部與 Strata 相同（`bench/vs_strata_*.py`）。推測解碼的輸出與 greedy 完全一致。

## 快速上手

需要 macOS 27、Xcode 27、128 GB 的 M5 Max。

```sh
make build/qwental-serve
mkdir -p models/iq3s-gguf
ln -s /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000?-of-00002.gguf models/iq3s-gguf/
./build/qwental-serve --gguf models/iq3s-gguf --port 8080
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=dummy claude --model qwen3.8-flash-next
```

`models/` 裡需要的模型檔：
- IQ3_S GGUF（`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` 的兩個分片）
- `Qwen3.8-Flash-Next-mixed-4-8bit/`，來自 [`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit)（tokenizer 與設定）
- `dflash2/`，放 DFlash2 猜測模型的打包檔（`…q4_0-selq4.pack`）

## Server

- Anthropic Messages API：`/v1/messages`（串流、thinking、tool use）與 `/v1/messages/count_tokens`
- OpenAI Chat Completions：`/v1/chat/completions`
- Prefix cache：多組 KV slot，並把快照存到 SSD
- 前端可以用 Claude Code 或任何 Anthropic / OpenAI 相容的 client

## 功能

- 直接讀 GGUF 量化權重（IQ2_S、IQ3_XXS、IQ3_S、IQ4_XS、IQ4_NL、Q4_K、Q5_K、Q6_K、Q8_0、Q5_1、BF16），解碼結果與參考實作逐位元相同
- 推測解碼：DFlash2 區塊猜測模型，加上照抄前文的猜測
- 批次 prefill，長 context 用稀疏 attention

## 限制

- 只支援一個模型、一台機器（M5 Max），一次處理一個請求
- 目前仍需要 MLX 模型目錄裡的 tokenizer

## 授權

MIT，見 `LICENSE`。第三方程式碼列於 `THIRD_PARTY_NOTICES.md`。不含模型權重。
