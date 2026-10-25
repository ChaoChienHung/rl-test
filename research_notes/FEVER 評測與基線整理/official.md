# FEVER 官方任務、評測與基線

## 任務與資料切分

### Takeaway
FEVER 2018 是以 Wikipedia 句子為證據的三分類事實查核；輸入 claim，系統必須自己找句子並判斷 `SUPPORTS`、`REFUTES`、`NOT ENOUGH INFO`。2018 shared task 使用 145,449 筆 train、19,998 筆 dev 與 19,998 筆 blind test；原論文另將 shared dev 拆成各 9,999 筆的 paper dev/test。

### Cited Findings
- 官網稱全資料共有 185,445 筆 claim，據 Wikipedia 句子改寫產生；支持及反駁類有人標註必要證據句。— [FEVER dataset](https://fever.ai/dataset/fever.html)
- 官方資料是 JSONL；train/dev 欄位為 `id`, `label`, `claim`, `evidence`；一筆可有多個可替代的 evidence set，每個 set 可含多句，句子以 Wikipedia page name 與 sentence ID 指定；NEI 的證據位置是 null。Blind test 不附 label/evidence。— [FEVER dataset](https://fever.ai/dataset/fever.html)
- Shared task 論文表 1：train 的 S/R/NEI 為 80,035 / 29,775 / 35,639，合計 145,449；dev 和 test 各為 6,666 / 6,666 / 6,666，合計各 19,998。資料依產生 claim 的 Wikipedia page 做切分。— [Shared task paper, §2.1](https://aclanthology.org/W18-5501.pdf)
- 官網指出 paper dev 與 paper test 合起來才是 shared task dev 的 19,998 筆；原論文表 1 的 paper dev/test 各類 3,333，合計各 9,999。— [FEVER dataset](https://fever.ai/dataset/fever.html); [FEVER paper, Table 1](https://aclanthology.org/N18-1074.pdf)
- 原論文稱 16.82% 的需證據 claim 需合併多於一句；12.15% 的證據跨頁。— [FEVER paper, introduction](https://aclanthology.org/N18-1074.pdf)

### Inferences
- Repo 若只使用 900 筆 frozen dev IDs 或較小訓練集，屬同資料集上的本地子集實驗，不能直接把百分比當作官方 blind-test 排名。
- 多句與跨頁案例確實存在；單句 verifier 或只返回一條 evidence 的設定有結構性限制。

### Gaps
- 本筆記未核對 repo 當前的 900 個 ID 是否都在官方 shared dev 中，需另查本地 manifest。

## 官方評測指標與解讀

### Takeaway
主指標 FEVER score 是「標籤正確，且非 NEI 時找出至少一組完整 gold evidence」的比例；只有標籤正確的 accuracy 不等於 FEVER score。

### Cited Findings
- 2018 任務定義要求從 Wikipedia 取句子證據，再作三分類；一個 claim 可需要多句合看。— [Official 2018 task](https://fever.ai/2018/task.html)
- Shared task 論文 §2.2：對 SUPPORTED/REFUTED，只有正確標籤加至少一個完整 evidence set 才得主分；NEI 判對無須 evidence。標籤正確但證據缺失或不完整者，FEVER score 為 0。— [Shared task paper](https://aclanthology.org/W18-5501.pdf)
- 只計提交的前五句 predicted evidence；超出的證據被丟棄。官方 scorer 回傳 strict FEVER score、label accuracy、evidence precision、recall、F1；evidence recall 也要求至少找齊一組完整證據。— [Official 2018 task](https://fever.ai/2018/task.html); [Official scorer README](https://github.com/sheffieldnlp/fever-scorer)
- Scorer 的示例展示 label accuracy 可以是 1.0，但缺少完整 evidence set 時 strict score 只有 0.5；多報錯句會降低 evidence precision。— [Official scorer README](https://github.com/sheffieldnlp/fever-scorer)
- 由於 dev/test 三類平衡，永遠預測 NEI 的 label accuracy 與 FEVER score 均為 33.33%；隨機猜標籤若無正確證據，FEVER score 通常更低。— [Shared task paper, §2.2](https://aclanthology.org/W18-5501.pdf)
- 原論文承認 gold evidence 標註未必完整，因此有時模型提出有效但未標註的證據，也會在自動證據評分上被扣分；2018 官網後來註明 final leaderboard 已納入約 1,200 筆社群補充標註。— [FEVER paper, §5.2](https://aclanthology.org/N18-1074.pdf); [Official 2018 leaderboard](https://fever.ai/2018/task.html)

### Inferences
- 報告應並列 label accuracy、FEVER score、evidence recall/F1，並標明用的是官方 scorer 還是 repo 自定義指標；尤其 confidence/ECE 不是原始 FEVER 官方主指標。
- 若要比官方 leaderboard，必須匹配 blind test、Wikipedia 版本、輸出前五句與官方 scorer；不同條件下只宜作背景參照。

### Gaps
- 未核查本 repo 的 `FEVER score` 實作是否與官方 scorer 完全一致；要作外部數值比較應另對照程式與樣本。

## 官方基線與歷史排行榜

### Takeaway
FEVER 官方基線本身就是三段式 pipeline：文件檢索、句子選取、NLI/verifier；2018 多數參賽隊伍亦沿用這三段式。這個 benchmark 原本沒有要求 agent/controller。

### Cited Findings
- 原論文官方 baseline：DrQA 的 unigram/bigram TF-IDF 取文件，TF-IDF 相似度排序句子，最後用 MLP 或 decomposable attention (DA) 作文字蘊含分類；NEI 因沒有標註證據，使用鄰近頁面句子或隨機 Wikipedia 句子模擬訓練。— [FEVER paper, §4](https://aclanthology.org/N18-1074.pdf)
- 原論文 dev 上最佳完整 pipeline (`DA / NEARESTP`，top 5 文件、top 5 句) 的 label accuracy 為 52.09%，要求證據完整的分數為 32.57%；paper test 分別為 50.91% 與 31.87%。— [FEVER paper, Tables 4 and §5.7](https://aclanthology.org/N18-1074.pdf)
- 2018 shared-task 論文的 blind test 表 2：官方 FEVER Baseline label accuracy 48.84%、FEVER score 27.45%；冠軍 UNC-NLP 為 68.21%、64.21%。論文說 23 隊參賽、19 隊超越既有 baseline，多數隊伍採用文件選取 → 句子選取 → NLI 的三階段設計。— [Shared task paper, Table 2 and §3](https://aclanthology.org/W18-5501.pdf)
- 2018 官網修訂後 final leaderboard 記載 FEVER Baseline accuracy 0.4892、FEVER score 0.2771；UNC-NLP 為 0.6798、0.6398，並說結果納入約 1,200 筆補充 evidence 標註。這些數值與 shared-task 論文原始 blind-test 表格不同，需註明版本。— [Official 2018 final leaderboard](https://fever.ai/2018/task.html)
- 原論文的 retrieval oracle：在 dev、top 5 文件中，55.30% 非 NEI claim 的完整 gold evidence 頁面被包含；再取 top 5 句後，DrQA 句選取能找齊完整證據的比例為 44.22%。— [FEVER paper, §§5.3–5.4](https://aclanthology.org/N18-1074.pdf)

### Inferences
- 用「BM25/TF-IDF → evidence selection → verifier」當經典 baseline 有充分先例；額外 controller 應作為待驗證新方法，不能說是 FEVER 的標準需求。
- 31.87%（原論文 paper test）、27.45%（2018 blind test 論文）、27.71%（官網修訂後 final leaderboard）屬不同切分或評分版本，不能混列成同一 run 的表現。

### Gaps
- 本筆記只涵蓋 FEVER 官方原論文與 2018 shared task；當代 LLM／dense-retrieval 系統的代表成績應另由各自原始論文核對，避免把 2018 排名稱為現今 SOTA。
