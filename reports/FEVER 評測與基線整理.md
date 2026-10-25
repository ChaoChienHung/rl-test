# FEVER 評測與基線整理

**可以直接用 FEVER 的人工標註。**官方 train/dev 已給每筆 claim 的三分類標籤，以及 `SUPPORTS`／`REFUTES` 所需的 Wikipedia 句子位置；沒有必要先請另一個模型重新發明標籤。需要另外製作的，是「把原始標註轉成目前 Qwen verifier 的輸入與輸出格式」及「讓模型學習自然語言推理過程」這兩種衍生資料。後者的教師 CoT 不是人工 gold，不能因為格式檢查通過就視為語義正確。評測時，FEVER score 同時要求答對類別與找齊至少一組完整證據；本 repo 的 900 筆 dev 子集成績應與同批 claim、同檢索和同 prompt 的本地對照比較，不宜直接當作官方 blind-test 排名。([官方資料頁](https://fever.ai/dataset/fever.html)，[官方 scorer](https://github.com/sheffieldnlp/fever-scorer))

## 原始 FEVER 已有人工標籤和證據，仍須轉成模型可用的配對

FEVER 原論文先從 **2017 年 6 月 Wikipedia** 的熱門頁面開頭抽句，請人工寫出單一事實 claim，並以改寫、否定、替換實體或關係、一般化及特化產生變體；另一批標註者在不知道原始來源句的情況下查 Wikipedia，給出 `SUPPORTS`、`REFUTES` 或 `NOT ENOUGH INFO`，並對前兩類選出足以支持判斷的證據句。資料因此不是機器自動生成、再拿模型自動標註的集合。原論文另報，在抽樣的 227 筆作者複核中，91.2% 的標註整體正確；多數問題是證據不足而非 label 錯誤。([FEVER 原論文，§3](https://aclanthology.org/N18-1074.pdf))

官方 JSONL 每行有 `id`、`claim`、`label`、`evidence`。`evidence` 是**多組可替代的完整證據集合**，集合內可有多句，每句以 Wikipedia 頁名與句子編號指向語料；NEI 的證據位置為 `null`。例如某筆 SUPPORTED claim 可同時有一組單句證據和另一組跨頁雙句證據。它**沒有**預先寫好的 verifier prompt、`E1` 編號、JSON completion、confidence 或 CoT。要做本 repo 的 SFT，仍須根據官方頁名／句號讀出句子、組成候選上下文、映射引用 ID 並決定在上下文缺失時如何處理標籤；這是資料轉換工作，不是重新標註原始 FEVER。([官方資料格式與例子](https://fever.ai/dataset/fever.html))

最重要的陷阱是 **FEVER 的 label 是對整個 Wikipedia 證據庫的標註，不保證我們取出的 top-k 候選句足夠**。若 gold 是 SUPPORTED，但檢索漏掉關鍵句，把那筆直接當「給定這些句子仍應輸出 SUPPORTED」的 SFT 樣本，會教模型在證據不足時猜答案。NEI 則沒有正面標註的「不足以判斷的句子」；原論文 baseline 因而從相近頁面或隨機頁面抽句，合成 NEI 的驗證器輸入。原論文還指出，16.82% 的需驗證 claim 必須組合多句，12.15% 涉及跨頁；不能假設一筆 claim 永遠只需一條證據。([FEVER 原論文，§3–5](https://aclanthology.org/N18-1074.pdf))

## 官方基線以檢索、選句、驗證三段完成任務

FEVER 2018 shared task 的訓練集有 **145,449** 筆，標註 dev 和 blind test 各 **19,998** 筆；dev/test 三類各 6,666。完整系統給定 claim，須自己找 Wikipedia 句子再判斷。官方主指標 **FEVER score** 對 SUPPORTED／REFUTED 要求正確 label 且預測的前五條證據包含任一**完整** gold evidence set；NEI 判對即可。只看三分類的 label accuracy 會高估未找對證據的系統。官方 scorer 另外輸出證據 precision、recall、F1；本 repo 的 non-NEI evidence F1 是對 gold 句 ID 聯集算的本地指標，不能與官方 evidence F1 同名直接比較。([2018 shared task 論文](https://aclanthology.org/W18-5501.pdf)，[官方 scorer](https://github.com/sheffieldnlp/fever-scorer))

原始官方 baseline 是 **DrQA TF-IDF 選文件 → TF-IDF 選句 → MLP 或 decomposable attention 判三類**，不是 agent。原論文的 paper test 對最佳完整 pipeline 報 **50.91% label accuracy／31.87% 證據嚴格分數**；後來 shared-task blind test 的**最終、補充標註後**官方榜單把 FEVER baseline 列為 **48.92%／27.71%**，UNC-NLP 為 **67.98%／63.98%**。這些數字分屬不同切分／評分版本，不能當成同一場測試的升降。([FEVER 原論文，§5](https://aclanthology.org/N18-1074.pdf)，[官方 2018 最終榜單](https://fever.ai/2018/task.html))

後續有代表性的 evidence-grounded 系統仍沿著相同問題改進：**GEAR** 在多句證據上做圖式整合，原論文報 test FEVER **67.10%**；**KGAT** 以細粒度圖注意力整合證據，報 test FEVER **70.38%**；**LisT5** 用 T5 做 listwise 選句與判斷，報 blind-test FEVER **75.87%**。這些是各論文自己的檢索器、模型、訓練資料和評測設定，不是換掉本 repo 的 verifier 就應達到的數字，也不應稱為 2026 年的最新 SOTA。([GEAR 原論文](https://aclanthology.org/P19-1085.pdf)，[KGAT 原論文](https://aclanthology.org/2020.acl-main.655.pdf)，[LisT5 原論文](https://aclanthology.org/2021.acl-short.51/))

下表把具代表性的整體系統放在一起作**方法地圖**。數字均為百分比；「—」表示此處未列可核對的同 run 數字。**paper test、官方 blind test 與本地 900-dev 不能橫向排高低**；各研究還使用不同檢索器、證據候選與評分版本。

| 系統與方法 | 測試切分／設定 | Label accuracy | FEVER score | 原始來源 |
|---|---|---:|---:|---|
| 原論文 DrQA TF-IDF 檢索／選句 + decomposable attention | 原論文 paper test，9,999 筆 | 50.91 | 31.87 | [FEVER 原論文，§5.7](https://aclanthology.org/N18-1074.pdf) |
| 2018 FEVER baseline，同一經典三段式 | 官方最終 blind test，19,998 筆，含補充證據標註 | 48.92 | 27.71 | [官方 2018 榜單](https://fever.ai/2018/task.html) |
| UNC-NLP，2018 競賽檢索／選句／驗證系統 | 官方最終 blind test，19,998 筆，含補充證據標註 | 67.98 | 63.98 | [官方 2018 榜單](https://fever.ai/2018/task.html) |
| BERT-large 選句 + BERT-large 驗證 | FEVER blind test，論文 2019/2020 報告版本 | 71.86 | 69.66 | [Soleimani 等原論文](https://arxiv.org/abs/1910.02655) |
| GEAR，BERT + 多證據圖整合 | FEVER blind test，論文報告版本 | 71.60 | 67.10 | [GEAR 原論文](https://aclanthology.org/P19-1085.pdf) |
| KGAT，RoBERTa-large + kernel graph attention | FEVER blind test，論文報告版本 | 74.07 | 70.38 | [KGAT 原論文，表 2](https://aclanthology.org/2020.acl-main.655.pdf) |
| LisT5，T5 listwise 選句與判斷 | FEVER blind test，論文報告版本 | 79.35 | 75.87 | [LisT5 原論文，表 1](https://aclanthology.org/2021.acl-short.51.pdf) |
| 本地 Qwen3.5-2B base，title + BM25，direct cap1024 | 本 repo frozen shared dev 子集，900 筆 | 47.6 | 34.6 | [本地 matched-cap 紀錄](/Users/ludwigchao/Desktop/Ludwig/assignment/rl-test/verifier/docs/prompted-baseline-results.md) |
| 本地 Qwen3.5-2B，5,895 筆 direct SFT，direct cap1024 | 同一 frozen shared dev 子集，900 筆 | 74.6 | 66.3 | [本地 matched-cap 紀錄](/Users/ludwigchao/Desktop/Ludwig/assignment/rl-test/verifier/docs/prompted-baseline-results.md) |

## 本 repo 應先做同條件對照，再談外部排名

本 repo 以 frozen **900 筆 shared-task dev claim**、每類 300 筆評估；真實檢索條件是 title match 加 BM25，與把 gold 頁塞進上下文的 `oracle` 必須分開。先前 1,000 筆 non-NEI 的 top-5 **頁面** recall，title 加 BM25 為 **87.5%**，純 BM25 為 **56.9%**；頁面召回不是完整證據句召回。無訓練 Qwen3.5-2B 在早期 title 條件是 **45.9% accuracy／34.0% FEVER**；統一 cap1024 的後續 run 中，base direct 為 **47.6%／34.6%**，base long CoT 為 **62.8%／52.8%**，5,895 筆 direct SFT 的 direct prompt 為 **74.6%／66.3%**。這些是本地 dev 數字，prompt 和版本差異應跟著記錄。([本地 baseline 紀錄](/Users/ludwigchao/Desktop/Ludwig/assignment/rl-test/verifier/docs/prompted-baseline-results.md))

目前 **4,500 筆 disjoint direct SFT** 的 900-dev title run 為 **76.67% accuracy／67.56% FEVER**；從其出發、在另 1,000 筆訓練 claim 上做的 **CoT GRPO** 為 **74.67%／67.33%**，但其推論 prompt 是 `cot_native`，而 SFT 行是 `direct`。因此現有兩行不能單獨歸因為「GRPO 讓性能升降」；至少要固定起點、claim IDs、retrieval、prompt、token cap 和 scorer，對照同一第二階段集合上的 SFT continuation 與 GRPO，並留未調 prompt 的 holdout。([SFT metrics](/Users/ludwigchao/Desktop/Ludwig/assignment/rl-test/verifier/evaluation/results/sft-direct-from-base-qwen3p5-2b-disjoint-4500-samples/direct-cap1024/run1/metrics.json)，[GRPO metrics](/Users/ludwigchao/Desktop/Ludwig/assignment/rl-test/verifier/evaluation/results/grpo-cot-from-sft4500-qwen3p5-2b-no-calibration-kl0-1000-samples/cot-native-cap1024/run1/metrics.json)，[本地評分程式](/Users/ludwigchao/Desktop/Ludwig/assignment/rl-test/verifier/model.py))

## 清資料先分清三種錯誤，無須推倒官方標註

目前看到的「不乾淨」至少有三層。**原始 FEVER** 有少量人工標註瑕疵及不完整 gold evidence；**本地檢索／prompt** 可能漏證據、放進與原標籤衝突的其他句子；**SoCLaaS 教師 CoT** 又可能引錯 E ID、錯判關係或寫出與自己旗標不一致的理由。第三層拒收不能推論第一層標籤錯。本地審核中，`qwen3.8:27b` 教師對 5,895 筆回覆的 validation-v2 接受 5,177、拒收 718；從三種拒收原因各抽 10 筆，30 筆 claim 文本與 label 均與官方 train 一致。後來的 **5,500 筆**是 5,177 原通過、258 citation 修復、65 NEI 旗標修復的組合，並非 5,500 筆都經人逐一確認語義。([本地教師資料審核](/Users/ludwigchao/Desktop/Ludwig/assignment/rl-test/verifier/docs/cot-teacher-rejection-audit.md)，[FEVER 原論文品質檢查](https://aclanthology.org/N18-1074.pdf))

因此**現在先直接使用官方 label 與 gold evidence 做 direct SFT 和評測**，保留每筆來源 ID；對 CoT、引文和上下文做有針對性的篩選。優先審核模型 fail case：先查 gold 完整證據是否真的進入輸入，再分別記錄檢索漏召回、上下文與 label 衝突、verifier 判錯、引用不全、教師理由有誤。對已抽查有疑慮的樣本標記或排除衍生 SFT，而不是悄悄改官方 gold。這是從原始任務與目前本地失敗類型推得的低成本處理順序；若要宣稱資料清洗提升泛化，還需在相同 ID／訓練量與獨立 holdout 上重跑對照。([官方資料格式](https://fever.ai/dataset/fever.html)，[本地審核](/Users/ludwigchao/Desktop/Ludwig/assignment/rl-test/verifier/docs/cot-teacher-rejection-audit.md))

## Conclusion

你們已經有人工 gold，可以把研究重點放回 **檢索是否找齊、verifier 是否善用證據，以及 RL 對同條件基線的增益**。額外的教師 CoT 只是一種可檢驗的訓練策略；它的拒收率既不是 FEVER 的錯標率，也不是先重做整份資料的理由。
