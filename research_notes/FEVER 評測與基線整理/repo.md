# 本地 repo 的 FEVER 評測與基線

## 現在評測什麼、怎麼計分？

### Takeaway
本地預設是從 FEVER shared-task dev 固定抽 900 筆（每類 300）評估 verifier，通常用 title-match + BM25 檢索、最多 5 頁及每頁 20 句。這是專案內部的 900 筆開發集結果，不等於官方完整 FEVER test leaderboard。

### Cited Findings
- 固定 dev 900、每類 300、seed 0、top 5 頁與每頁最多 20 句、temperature 0，見 [prompted baseline](../../verifier/docs/prompted-baseline-results.md#L1-L7)。原始 FEVER train/dev 與維基語料另外下載，不在 git，見 [data](../../verifier/docs/data.md#L8-L20)。
- 評測有 `bm25`、`titles`、`oracle` 三種模式；`oracle` 把 gold 頁放進上下文，`titles` 是標題匹配加 BM25 的真實檢索條件，見 [evaluator CLI](../../verifier/evaluation/eval_verifier.py#L8-L16)、[GRPO doc](../../verifier/docs/grpo-results.md#L53-L56)。
- `titles` 的 page recall@5 = 0.875，純 BM25 = 0.569，這是 1,000 筆 non-NEI 的**頁面**檢索統計，不是證據句 recall，見 [prompted baseline](../../verifier/docs/prompted-baseline-results.md#L9-L23)。
- 報告的 accuracy = verdict 正確率；FEVER score = verdict 正確且找到任一完整 gold 證據集合；non-NEI evidence F1 = 引用句 ID 對所有 gold ID 聯集算 F1；另有格式率、每類 recall、binary correctness Brier/ECE、輸出 tokens，見 [scorer](../../verifier/model.py#L400-L429)、[F1 implementation](../../core/reward.py#L82-L97)、[offline summary](../../verifier/evaluation/score_predictions.py#L78-L105)。注意此 evidence F1 定義是本地實作，不應直接當作官方證據 F1。
- 固定 5,895 個 oracle SFT 訓練樣本拆成 4,695／1,200，和 900 dev 零重疊；另有 5,500 個教師驗證修復樣本拆成 4,500 SFT／1,000 GRPO，SFT 部分三類各 1,500，dev overlap 0，見 [data](../../verifier/docs/data.md#L90-L92)、[4500 split manifest](../../verifier/data/train/sft/fever-oracle-disjoint-4500/split_manifest.json#L1-L21)。

### Inferences
- 比較 prompt、SFT、GRPO 時須鎖同一批 claim IDs、檢索、模型、解碼及 token cap；`oracle` 對 `titles`、不同 cap、不同資料量或不同 prompt 的行列不能歸因給 RL。這是根據 [plan](../../verifier/docs/plan.md#L42-L44) 的對照原則推得。

### Gaps
- 目前 repo 未見獨立且未參與 prompt 開發的最終 holdout 評測；[TODO](../../verifier/TODO.md#L17-L17) 明確提醒 900 dev 上調 prompt 後須換新 holdout。外部官方 FEVER leaderboard 成績須另查官方來源。

## 本地已做的 baseline 與結果是什麼？

### Takeaway
本地最重要的對照是無訓練 2B、同模型 direct SFT、CoT SFT 與從 SFT 起點做的 CoT GRPO；不能單拿最高分和不同設定的行比較。

### Cited Findings
- 無訓練 Qwen3.5-2B 在 titles 上 accuracy 45.9%、FEVER 34.0%、non-NEI evidence F1 42.3%；較強 OpenRouter `qwen/qwen3.8-flash` 在同模式是 72.0%、64.0%、71.7%，但模型和成本不同，見 [prompted baseline](../../verifier/docs/prompted-baseline-results.md#L25-L45)。
- 同 900 dev、titles、cap1024 下：base direct 47.6% accuracy／34.6% FEVER；base long CoT 62.8%／52.8%；5,895-row SFT direct 74.6%／66.3%；同 SFT long CoT 71.4%／62.4%，見 [six-condition table](../../verifier/docs/prompted-baseline-results.md#L74-L89)。
- 4,500-row disjoint JSON SFT direct = accuracy 76.67%、FEVER 67.56%、non-NEI evidence F1 64.93%；對應 1,000-row JSON CoT GRPO = 74.67%、67.33%、74.73%。兩者 prompt 類型不同（direct vs `cot_native`），所以「GRPO 提高 F1」不等於排除了 prompt confound，見 [SFT metrics](../../verifier/evaluation/results/sft-direct-from-base-qwen3p5-2b-disjoint-4500-samples/direct-cap1024/run1/metrics.json#L1-L15)、[GRPO metrics](../../verifier/evaluation/results/grpo-cot-from-sft4500-qwen3p5-2b-no-calibration-kl0-1000-samples/cot-native-cap1024/run1/metrics.json#L1-L15)。
- 5,895-row SFT 起點的 1,200-row CoT GRPO（seen arm）報 74.1% accuracy、67.1% FEVER、72.2% non-NEI F1；4,695-row 起點（unseen arm）報 76.2%、67.2%、73.7%，見 [TODO](../../verifier/TODO.md#L11-L15)。其中 unseen arm 是**第二階段 1,200 訓練 claim 未被前一階段 SFT 看過**，並不是 FEVER 測試分布外泛化；兩臂起始 SFT 資料量也不同。
- Tag-format CoT GRPO 的 gold-in-context 子集 n=517 有 accuracy 90.1%、FEVER 81.4%，但全 900 只有 accuracy 67.56%、FEVER 58.0%、NEI recall 33.67%；不可把子集數字當整體結果，見 [tag GRPO metrics](../../verifier/evaluation/results/grpo-cot-from-sft4500tagged-qwen3p5-2b-no-calibration-kl0-1000-samples/cot-native-tag-cap1024/run1/metrics.json#L1-L15)、[stratum](../../verifier/evaluation/results/grpo-cot-from-sft4500tagged-qwen3p5-2b-no-calibration-kl0-1000-samples/cot-native-tag-cap1024/run1/metrics.json#L50-L55)。
- 歷史 direct-output GRPO 的兩個 seed 有約 2.5 percentage point accuracy 與 25 point REFUTED recall 差，且 token-probability ECE 比 SFT 差，見 [GRPO results](../../verifier/docs/grpo-results.md#L58-L76)。

### Inferences
- 合理主表應至少分兩條：A. 同模型同 prompt/cap 的 base→SFT；B. 同起點同 prompt/cap、1,200（或 1,000）第二階段 claims 的 SFT continuation→GRPO，並附上 oracle/titles、每類 recall、FEVER、evidence F1、成本與 seeds。這由 [plan](../../verifier/docs/plan.md#L42-L44) 與現存結果配置推得。

### Gaps
- `verifier/docs/grpo-results.md` 仍稱 native-thinking CoT GRPO 結果 pending（第 3–5 行），和較新的 TODO/metrics 矛盾；應以具體 metrics 與 manifest 為準。尚需核對每組的輸入 context、起點 adapter、輸出 cap、實際總訓練步數，並跑獨立 holdout、多 seed 以支持提升結論。

## 資料「不乾淨」具體是哪一種問題？

### Takeaway
必須分開三件事：原 FEVER 標籤、檢索上下文是否含足夠證據、以及 SoCLaaS 教師 CoT 是否忠實。拒收教師輸出不等於原 FEVER 標籤錯；目前的 5,500 樣本是結構驗證後的訓練集，不能稱為人工確認的語義乾淨集。

### Cited Findings
- `qwen3.8:27b` 教師回傳 5,895 筆；validation-v2 接受 5,177、拒絕 718。抽查三類拒絕理由各 10 筆，30 筆的 claim 文本與標籤都吻合官方 FEVER train，見 [teacher audit](../../verifier/docs/cot-teacher-rejection-audit.md#L1-L7)。
- 教師拒絕有 citation 格式不符、`gold_justified=false` 自相矛盾、教師 relation 錯、以及 shown evidence 不足或與原標籤衝突等不同原因；抽查報告特別指出不要從 rejection code 推出原 FEVER label 錯誤，見 [citation audit](../../verifier/docs/cot-teacher-rejection-audit.md#L24-L24)、[gold audit](../../verifier/docs/cot-teacher-rejection-audit.md#L26-L41)、[relation audit](../../verifier/docs/cot-teacher-rejection-audit.md#L43-L58)。
- 5,177 accepted 中再抽 30，發現 2 個明顯的 prompt/label 或 explanation 衝突和 3 個有爭議推論；報告明說這不是全體錯誤率估計，見 [accepted audit](../../verifier/docs/cot-teacher-rejection-audit.md#L67-L79)。
- 後來 5,500 = 5,177 原通過 + 258 個 citation deterministic 修復 + 65 個 NEI 布林旗標修復；還有 395 筆未收。這是驗證與修復構成，不是 5,500 筆全經人工審定，見 [teacher audit](../../verifier/docs/cot-teacher-rejection-audit.md#L83-L96)、[summary](../../verifier/data/train/sft/fever-oracle-cot-soclaas/summary.json#L1-L29)。
- 900 dev 的 title-match 上下文在 600 個 non-NEI 中只對 517 個含完整 gold 證據，因此 83 個是檢索／上下文問題，不能全歸為 verifier 推理錯，見 [prompted baseline](../../verifier/docs/prompted-baseline-results.md#L27-L40)。

### Inferences
- 不建議現在將所有 5,500 筆重標或重訓當作「清資料」；先做分層 fail-case audit（缺證據、標籤／語境衝突、teacher CoT 錯、格式錯）並保留原標籤與 provenance。需要乾淨對照時，在同 ID 上評測不同模型，另建經人工核對的小型 holdout；由 [TODO fail-case plan](../../verifier/TODO.md#L17-L17) 與 [teacher audit decisions](../../verifier/docs/cot-teacher-rejection-audit.md#L60-L65) 推得。

### Gaps
- 沒有對所有 5,500 教師理由做語義人工驗證，亦沒有可據以估計整體 teacher error rate 的隨機足量樣本。原 FEVER label 的全面重標問題也未調查；現有 30 筆只能排除本地 claim/label join 錯誤的抽樣疑慮。
