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

另外也支援去審查版的 **SC117 IQ3_S**（[`SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF`](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)：同一份 GSQ-RCO 量化，其中 96 個投影改成 Q8_0），與 Strata 用同一個檔案對照（順序交錯兩輪取平均，tok/s）：

| | Strata | Qwental |
|---|---|---|
| 解碼：英文文章／寫新程式碼／改程式碼 | 92.9／131.0／170.8 | **100.9／138.5／198.8** |
| 10.7K／47K 之後的解碼 | 90.7／85.1 | **98.2／87.9** |
| Prefill 10.7K／47K | 1330／1421 | **1629／1613** |

`--int8` 在 SC117 上也能用：另一組交錯測試中，prefill 10.7K／47K 從 1949／1860 提高到 2802／2815 tok/s，解碼不變；與同一版不開 INT8 比較，第一名相同 99.0–99.3%，平均 KL 0.002–0.022（4K 與 34K 的 prompt）。

與 oMLX 0.7.4.dev1 對照，用同一份 MLX 權重（mixed 4/8-bit，Qwental 不加 `--gguf`）：

| | oMLX | Qwental | Qwental `--int8`／`--int8-qk` |
|---|---|---|---|
| 解碼：英文文章／寫新程式碼／改程式碼 | 53.4／78.4／136.4 | **91.8／131.9／181.5** | |
| 10.7K／47K 之後的解碼 | 53.8／55.3 | **83.0／81.1** | |
| Prefill 10.7K | 1884 | 1950 | 2456／**2480** |
| Prefill 47K | 2072 | 2015 | 2526／**2611** |

### 同時多個請求

多個 agent（例如 Claude Code 的 subagent）可以同時進行。同一個 IQ3_S GGUF，同時送出每個 400 token 的請求（`bench/concurrent.py`，總 tok/s）：

| 同時請求數 | 1.2.0 | 1.3.0 |
|---|---|---|
| 1 | 110.9 | 110.1 |
| 4 | 110.3 | **146.7**（+33%） |
| 8 | 107.9 | **172.3**（+60%） |
| 16 | 109.0 | **170.9**（+57%） |

MLX 權重同時 8 個：81.8 → 135.8。GGUF 的輸出與每個請求單獨跑逐位元相同。

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

用 Homebrew 安裝（從原始碼編譯），之後把下面的 `./build/qwental-serve` 換成 `qwental-serve`：

```sh
brew tap hy-ong/qwental https://github.com/hy-ong/qwental
brew install hy-ong/qwental/qwental
```

或在 repository 裡編譯：

```sh
make build/qwental-serve   # 選用：make install（把單一執行檔複製到 /usr/local/bin；用 PREFIX=… 改位置）
mkdir -p models/iq3s-gguf
ln -s /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000?-of-00002.gguf models/iq3s-gguf/
./build/qwental-serve --gguf models/iq3s-gguf --port 8080
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=dummy claude --model qwen3.8-flash-next
```

`models/` 裡需要的模型檔：
- IQ3_S GGUF（`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` 的兩個分片）；`--gguf` 時 tokenizer 直接從 GGUF 的 metadata 讀（token id 與 `tokenizer.json` 完全相同）；SC117 的 IQ3_S（兩個分片）用法相同
- `dflash2/`，放 DFlash2 猜測模型的打包檔（`…q4_0-selq4.pack`），與 GGUF 目錄同層（有 MLX 目錄時與 MLX 目錄同層）
- 選用：`Qwen3.8-Flash-Next-mixed-4-8bit/`，來自 [`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit)，只有跑 MLX 權重（不加 `--gguf`）時才需要。有這個目錄時用它的 tokenizer。

## Server

- Anthropic Messages API：`/v1/messages`（串流、thinking、tool use）與 `/v1/messages/count_tokens`
- OpenAI Chat Completions：`/v1/chat/completions`
- Prefix cache：多組 KV slot，並把快照存到 SSD；同時進行的對話之間共用前綴
- 同時多個請求（預設開，最多 16 個；`--parallel N` 改上限，最多 64，`--parallel 1` 一次一個）：只有一個請求時走推測解碼，多個時一起 decode，依實測速度自動選。新的長 prompt 分段 prefill，正在生成的對話不會停住；之後才到的請求也不必等它跑完（即使它是單獨進來的）。slot 的記憶體有上限（實體記憶體 − 權重 − 24 GB，或 `QW_SLOT_MEM_GB`）；閒置的 slot 會縮到實際用量，系統可用記憶體偏低時（`QW_MEM_LEVEL_MIN`，預設 20%）先縮小閒置 slot 或寫到 SSD 快取。前面的請求會讓 prompt 處理等超過 90 秒時回 529 並附 `retry-after`（`QW_MAX_WAIT_S`，0 表示關閉）；被取消的 prompt 會保留已算的進度給重試接續。
- `GET /status`：版本、profile、執行時間、請求與 token 的累計、排隊數，同時多個請求時另有目前的模式、slot 記憶體、實測速度與每個進行中的對話；`--help` 列出所有選項
- `GET /dashboard`（也可以用 `/`）：根據 `/status` 即時更新的網頁，顯示進行中的對話、模式、tok/s、快取命中率、slot 記憶體、近 5 分鐘的走勢圖、API 計價（OpenRouter `qwen/qwen3.8-flash` 價格，可自訂）以及依日／週／月的用量；英文或繁體中文
- `--stats-file PATH`：累計數字與每日用量跨重啟保存的位置（預設 `<cache-dir>/stats.json`，`""` 表示不保存）；`GET /status` 以 `lifetime` 回報，加 `?days=N` 附每日資料
- 實驗性質、預設關閉：同時 2–3 個請求時的批次推測解碼（`QW_PAR_VSPEC=1`，限 GGUF）
- 前端可以用 Claude Code 或任何 Anthropic / OpenAI 相容的 client。完整支援 Claude Code：`messages` 裡的 system 訊息、思考強度（`high` → xhigh）、subagent（任何 model 名稱都由同一個模型服務）、平行的 tool call、自動壓縮（"prompt is too long" 錯誤）、按 Esc 取消、長 prefill 期間的 keepalive ping、可重試的錯誤（529／SSE error），以及 `usage` 裡的快取欄位。檔案內容或工具輸出裡的特殊 token 字面維持一般文字。
- `--anthropic-thinking on_request|always`：沒有要求 thinking 的 Anthropic 請求不思考（`on_request`，預設，同 Strata）或照樣思考（`always`）
- `--int8`：prefill 的 expert 矩陣（GGUF 再加上 dense 矩陣）改走 INT8，長 prompt 約快 25–60%。decode 不變；輸出與預設略有差異（見「INT8 的代價」）。搭配 `--gguf` 時多用 3 GB 記憶體。
- `--int8-qk`／`--int8-hyper`：在 `--int8` 之外，attention 的 Q·Kᵀ（`qk`）或 hyper-connection 的 mix_down（`hyper`，限 `--gguf`）也改走 INT8。兩者都隱含 `--int8`。`--int8-qk` 主要對長 prompt 有幫助，品質與 `--int8` 相當；`--int8-hyper` 屬實驗性質（測試中英文內容變差）。
- `--fast-head`（搭配 `--gguf`）：Q6_K 的 lm_head 改走矩陣單元，權重捨入成 FP16（同 llama.cpp）。推測解碼的驗證較快；機率幾乎相同時，輸出可能與預設不同。
- `--exact-spec`（MLX 權重）：推測解碼的輸出與 greedy 完全相同。MLX 輸出與預設略有差異；GGUF 不受影響（本來就相同）。

## 功能

- 直接讀 GGUF 量化權重（IQ2_S、IQ3_XXS、IQ3_S、IQ4_XS、IQ4_NL、Q4_K、Q5_K、Q6_K、Q8_0、Q5_1、BF16），解碼結果與參考實作逐位元相同
- 推測解碼：DFlash2 區塊猜測模型，加上照抄前文的猜測
- 批次 prefill，長 context 用稀疏 attention

## 限制

- 只支援一個模型、一台機器（M5 Max）
- 同時多個請求：GGUF 的輸出與單獨跑逐位元相同；MLX 權重一起 decode 時捨入與單獨跑略有不同（同 MLX 預設的推測解碼），加 `--exact-spec` 就逐位元相同。只有一個請求時不受影響
- 同時超過約 8 個請求後，總速度不再增加

## 授權

MIT，見 `LICENSE`。第三方程式碼列於 `THIRD_PARTY_NOTICES.md`。不含模型權重。
