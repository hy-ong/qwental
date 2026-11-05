# Qwental

[English](README.md) | **繁體中文**

專為 **Apple M5 Max（128 GB）** 上的 **Qwen3.8-Flash-Next** 打造、逼近硬體極限的 Metal 推論 engine。以 C23 與 MSL（Metal 4 tensor API）撰寫，不依賴任何框架。

## 效能

Qwental 直接執行 Strata 預設的模型檔 ISTA-DASLab GSQ-RCO **IQ3_S GGUF**，不轉格式：權重相同、同樣以 FP16 運算。在同一台機器上與 Strata 對照（`bench/vs_server.py`，順序交錯兩輪取平均，tok/s）：

| | Strata | Qwental | Qwental `--int8` |
|---|---|---|---|
| 解碼：英文文章 | 97.2 | **112.5**（+16%） | |
| 解碼：寫新程式碼 | 142.3 | **153.8**（+8%） | |
| 解碼：改程式碼 | 173.3 | **203.5**（+17%） | |
| 10.7K prompt 之後的解碼 | 93.9 | **101.3**（+8%） | |
| 47K prompt 之後的解碼 | 85.9 | **88.3**（+3%） | |
| Prefill 10.7K | 1473 | 1728（+17%） | **2750**（+87%） |
| Prefill 47K | 1461 | 1747（+20%） | 2541（+74%）；`--int8-qk` **2671**（+83%） |

`--int8` 不影響解碼。不加 INT8 旗標時輸出是同一個模型：在 Strata 自己生成的 600 個位置上，Qwental 的第一名全部與 Strata 相同（`bench/vs_strata_*.py`），推測解碼的輸出也與 greedy 完全一致。

與 oMLX 0.7.4 對照，用同一份 MLX 權重（mixed 4/8-bit，Qwental 不加 `--gguf`）：

| | oMLX | Qwental | Qwental `--int8`／`--int8-qk` |
|---|---|---|---|
| 解碼：英文文章／寫新程式碼／改程式碼 | 53.4／78.4／136.4 | **91.8／131.9／181.5** | |
| 10.7K／47K 之後的解碼 | 53.8／55.3 | **83.0／81.1** | |
| Prefill 10.7K | 1884 | 1950 | 2456／**2480** |
| Prefill 47K | 2072 | 2015 | 2526／**2611** |

### INT8 的代價

INT8 旗標會改變 prefill 的數值。以程式碼、英文、中文三種內容，在 8K–32K 的 prompt 之後打分 300 個位置（`QW_PPL_PF`、`tools/logit_cmp.py`），與同一版不開 INT8 比較（GGUF；MLX 權重下 `--int8` 第一名相同 96.3%、KL 0.008–0.039）：

| | 第一名相同 | 平均 KL | 困惑度 |
|---|---|---|---|
| `--int8` | 97.9% | 0.006–0.023 | −1.4% 到 +0.7% |
| `--int8-qk` | 98.0% | 0.005–0.018 | −1.2% 到 +1.3% |
| `--int8-hyper` | 96.7% | 0.006–0.066 | 最多 +7.6%（英文） |
| 對照：MLX 4/8-bit 與 IQ3_S GGUF 之間 | 93.3% | 0.038–0.141 | −5.0% 到 +7.1% |

`--int8` 與 `--int8-qk` 的差異遠小於同一模型兩種量化之間的差異；`--int8-hyper` 則不然，屬實驗性質。

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
- `--int8`：prefill 的 expert 矩陣（GGUF 再加上 dense 矩陣）改走 INT8，長 prompt 約快 25–60%。decode 不變；輸出與預設略有差異（見「INT8 的代價」）。搭配 `--gguf` 時多用 3 GB 記憶體。
- `--int8-qk`／`--int8-hyper`：在 `--int8` 之外，attention 的 Q·Kᵀ（`qk`）或 hyper-connection 的 mix_down（`hyper`，限 `--gguf`）也改走 INT8。兩者都隱含 `--int8`。`--int8-qk` 主要對長 prompt 有幫助，品質與 `--int8` 相當；`--int8-hyper` 屬實驗性質（測試中英文內容變差）。
- `--fast-head`（搭配 `--gguf`）：Q6_K 的 lm_head 改走矩陣單元，權重捨入成 FP16（同 llama.cpp）。推測解碼的驗證較快；機率幾乎相同時，輸出可能與預設不同。

## 功能

- 直接讀 GGUF 量化權重（IQ2_S、IQ3_XXS、IQ3_S、IQ4_XS、IQ4_NL、Q4_K、Q5_K、Q6_K、Q8_0、Q5_1、BF16），解碼結果與參考實作逐位元相同
- 推測解碼：DFlash2 區塊猜測模型，加上照抄前文的猜測
- 批次 prefill，長 context 用稀疏 attention

## 限制

- 只支援一個模型、一台機器（M5 Max），一次處理一個請求
- 目前仍需要 MLX 模型目錄裡的 tokenizer

## 授權

MIT，見 `LICENSE`。第三方程式碼列於 `THIRD_PARTY_NOTICES.md`。不含模型權重。
