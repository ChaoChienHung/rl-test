# FEVER 主流方法與基線

## 哪些方法可當 FEVER 的代表性 baseline？

### Takeaway
主流 FEVER 1.0 是「文件檢索 → 證據句選取 → 三分類驗證」的 pipeline；方法演進主要在改進檢索、跨句證據整合與判斷模型。下列數字都須連同 split 與檢索設定閱讀。

### Cited Findings
- **官方原始 baseline**：2018 shared task 最終 blind-test leaderboard 的 FEVER baseline：label accuracy **48.92%**、evidence F1 **18.66%**、FEVER score **27.71%**。原始 NAACL 論文摘要另報「正確 label 加證據」31.87% 和忽略證據 50.91%，來自較早的論文設定；勿混為最終 shared-task test 數字。— [官方 leaderboard](https://fever.ai/2018/task.html); [原始論文](https://aclanthology.org/N18-1074/)
- **2018 經典競賽系統**：UNC-NLP（ESIM 類 verifier）、UCL MRG（四階段檢索、NLI、聚合）、Athene；在 KGAT 論文彙整的 FEVER 1.0 表格，dev/test 的 (label accuracy, FEVER score) 分別為 UNC **(69.72,66.49)/(68.21,64.21)**、UCL **(69.66,65.41)/(67.62,62.52)**、Athene **(68.49,64.74)/(65.46,61.58)**，單位 %。UCL 自己的論文將其流程列成 document retrieval、sentence retrieval、NLI、aggregation。— [KGAT 原論文表 2](https://aclanthology.org/2020.acl-main.655.pdf); [UCL 原論文](https://aclanthology.org/W18-5515/)
- **最終官方 leaderboard 與後續論文引述有版本差異**：官方 2018 final leaderboard（加入約 1,200 筆社群標註）列 UNC 63.98%、UCL 62.34%、Athene 61.32% FEVER score，而早期/provisional test 常引 UNC 64.21%、UCL 62.52%、Athene 61.58%。比較時要說清楚 leaderboard 版本。— [官方 2018 leaderboard 與註記](https://fever.ai/2018/task.html); [KGAT 原論文表 2](https://aclanthology.org/2020.acl-main.655.pdf)
- **BERT sentence selector + BERT verifier**：Soleimani 等的兩個 BERT 模型分別選句及分類；原論文報證據句 recall **87.1%**、FEVER score 約 **69.7%**。KGAT 論文同一比較表引 BERT-large dev/test FEVER **72.42%/69.66%**、BERT-base **71.38%/68.50%**；其文件檢索採用 MediaWiki API/實體片語，不能當成固定 BM25 的單一模型效果。— [BERT 原論文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7148011/); [KGAT 原論文表 2 與方法](https://aclanthology.org/2020.acl-main.655.pdf)
- **多證據整合圖模型**：GEAR 在檢索到的句子上建 fully connected evidence graph，再用 BERT 表徵及圖上資訊傳遞，blind-test FEVER **67.10%**；KGAT 用 kernel graph attention，原論文表 2 的 RoBERTa-large 版本 dev/test label accuracy **78.29%/74.07%**、FEVER **76.11%/70.38%**。該論文另報 verifiable dev claim 中 **14.7%** 需多句證據，因此這一類方法解的是真實但非所有樣本都有的瓶頸。— [GEAR 原論文](https://aclanthology.org/P19-1085.pdf); [KGAT 原論文表 2 與分析](https://aclanthology.org/2020.acl-main.655.pdf)
- **T5 listwise 證據推理**：LisT5 在同一個輸入裡整合候選句，並以加雜訊的 evidence-set 訓練；原論文報 dev/test label accuracy **81.26%/79.35%**、FEVER **77.75%/75.87%**。文件檢索沿用既有方案並加入 BM25，句子選取用 T5，所以不是「BM25 單獨達 75.87%」。— [LisT5 原論文摘要、表 1 與方法](https://aclanthology.org/2021.acl-short.51.pdf)
- **現代 dense/hybrid 檢索方向**：RAV 用 bi-encoder 初選、cross-encoder 重排，讓驗證訊號也改善證據選取；接 KGAT 的版本在原論文表 1 報 dev/test FEVER **78.53%/76.45%**，但這是 2024 年研究方法，不宜稱作「傳統 baseline」。這組數字也同時更換了 retrieval 與訓練方式。— [RAV 原論文與表 1](https://aclanthology.org/2024.findings-acl.551.pdf)
- **不檢索的 LM 對照**：Lee 等用 BERT 的參數知識做 closed-book fact checking，原論文表 1：zero-shot BERT-as-KB **49% label accuracy**，fine-tuned BERT encoder **57% label accuracy**；論文明說未提交證據，所以**沒有可比的 FEVER score**。它適合作為 claim-only / closed-book 對照，而非 evidence-grounded pipeline 的同等分數。— [原論文表 1 與評估說明](https://aclanthology.org/2020.fever-1.5.pdf)

### Inferences
- 對課堂專案，最有解釋力的對照是固定同一批 claims、Wikipedia 快照、檢索結果與評估器，比「BM25 + 零樣本 2B verifier → SFT → GRPO」的增益；若要對照文獻分數，需另跑完整 FEVER 1.0 官方 split 與官方 scorer。這由各原論文檢索與資料設定差異推得。— [FEVER 官方 task](https://fever.ai/2018/task.html); [KGAT 方法](https://aclanthology.org/2020.acl-main.655.pdf)
- 若想增加一個合理的外部強 baseline，BERT cross-encoder 或 T5 listwise 比額外加 controller agent 更貼近 FEVER 已建立的研究路線；這是依上述原論文方法與問題設定做的研究設計建議。— [BERT](https://pmc.ncbi.nlm.nih.gov/articles/PMC7148011/); [LisT5](https://aclanthology.org/2021.acl-short.51.pdf)

### Gaps
- 本筆記沒有找到可直接與現有 repo **900-claim、特定 BM25/title 檢索與 Qwen 2B 設定**等條件對齊的已發表分數；論文 blind-test 分數只能當歷史脈絡，不能當直接勝負比較。
- 大型 instruction-tuned LLM 的 zero/few-shot FEVER 結果高度依模型、prompt、是否給 gold 或 retrieved evidence 而變；此處未找到足以作為一個穩定「主流標準分數」的單一官方基線，因此不列精確數值。

## 比較 FEVER 系統時，哪些條件會讓數字不可直接相比？

### Takeaway
必須區分完整的 evidence-grounded FEVER score、單純三分類 accuracy、oracle/gold-evidence 上限，以及不同資料切分或檢索設定。

### Cited Findings
- 官方任務要求對 SUPPORTED/REFUTED 提交完整的一組 Wikipedia 證據句；NEI 不需證據。FEVER score 需同時答對 label 並交出至少一組完整 gold evidence，label accuracy 則只看類別。— [官方 task definition](https://fever.ai/2018/task.html); [GEAR 原論文 metrics](https://aclanthology.org/P19-1085.pdf)
- KGAT 把 gold evidence 已提供的 GFEVER 與正常檢索後的 FEVER 分開；它也明示 document retrieval 使用實體片語加 MediaWiki API，sentence retrieval 有 ESIM、BERT 兩種。這些是不同檢索條件，不能把 downstream verifier 分數單獨對比。— [KGAT 原論文方法與指標](https://aclanthology.org/2020.acl-main.655.pdf)
- FEVER 2018 shared task 的 test 是 blind leaderboard，LisT5 原論文也是盲測；與挑出的 900 筆 dev subset 在樣本數與分布上不同。— [官方 leaderboard](https://fever.ai/2018/task.html); [LisT5 原論文](https://aclanthology.org/2021.acl-short.51.pdf)

### Inferences
- 報告表格宜至少設 `split / n / retrieval / evidence來源 / model / label accuracy / FEVER score` 欄；若是 gold evidence 或無 evidence 的 closed-book 設定，要另表，不應用同一欄混排。— [官方 scoring task](https://fever.ai/2018/task.html); [LM closed-book 原論文](https://aclanthology.org/2020.fever-1.5.pdf)

### Gaps
- 若需對照目前 repo 的確切 scoring 實作、claim IDs 與檢索設定，應讀 repo evaluator/result metadata；外部論文無法代替這一步。
