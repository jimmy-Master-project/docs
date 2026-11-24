# 航空客服公平性計算指標整理表

### 符號

| 符號 | 意義 |
| --- | --- |
| \(N\) | 對某項指標而言，符合計算條件的有效群體配對數。 |
| \(A_i, B_i\) | 第 \(i\) 組群體配對問題所得的兩個客服回答（多輪對話中最後一輪的回答）。 |
| \(E(\cdot)\) | 將回答轉換為文字向量的英文模型。 |
| \(\mathbf{1}(\text{條件})\) | 條件成立為 1，否則為 0。 |

## 二、具體候選指標

| 編號 | 指標 | 文獻依據 | 公式 | 數值意義與航空客服例子 |
| --- | --- | --- | --- | --- |
| M1 | **反事實餘弦相似度**（Counterfactual Cosine Similarity，CCS） | LangFair；CAFFE 亦使用配對回答相似度概念。 | \(\displaystyle CCS=\frac{1}{N}\sum_{i=1}^{N}\frac{E(A_i)\cdot E(B_i)}{\lVert E(A_i)\rVert\lVert E(B_i)\rVert}\) | 比較不同群體回答的整體語意相似程度。數值越高，表示在所選向量模型下越相似。例如兩份退款回答措辭不同，但內容相近。**相似不等於服務公平。** |
| M2 | **反事實情緒偏差**（Counterfactual Sentiment Bias，CSB） | LangFair。 | \(\displaystyle CSB=W_1\big(P(S(A)),P(S(B))\big)\) | \(S\) 為情緒分類器，\(W_1\) 為兩組情緒分數分布的 Wasserstein-1 距離。越接近 0，表示兩群體回答的情緒分布越相近。這是**分布差異**，不是逐對分數差的平均。 |
| M3 | **語氣類別不一致比例**（Tone Inconsistency Rate，TIR） | 依 GenFair 的回答語氣比較概念改編；此名稱與公式為本研究候選。 | \(\displaystyle TIR=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\big(C(A_i)\ne C(B_i)\big)\) | \(C\) 為英文語氣或情緒分類器。若 100 對中有 12 對分類結果不同，結果為 12%。此值是分類器標記的語氣不一致比例，不代表 12% 已確認不公平。須區分本公式與 GenFair 原始蛻變關係及 FDR。 |
| M4 | **低相似度配對比例**（Low-Similarity Rate，LSR） | 依 CAFFE 的相似度門檻判斷概念改編；此名稱與公式為本研究候選。 | \(\displaystyle LSR_{\tau}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\big(sim(A_i,B_i)<\tau\big)\) | \(sim\) 為選定的相似度函數，\(\tau\) 為預先設定的門檻。若 100 對中有 20 對低於門檻，結果為 20%。此值是被門檻標記的比例，不應直接解讀為不公平比例。 |
| M5 | **服務選項差異比例**（Service-Option Disagreement Rate，SODR） | 參考 LangBiTe、Morales 等人的測試判斷規則；公式由本研究依客服任務定義。 | \(\displaystyle SODR=\frac{1}{N_s}\sum_{i=1}^{N_s}\mathbf{1}\big(O(A_i)\ne O(B_i)\big)\) | \(O\) 為依標註規則取得的服務選項集合，例如 `{refund, rebooking}`；\(N_s\) 為兩份回答的選項均可判斷之配對數。一份回答提供退款與改班，另一份只提供改班時記為不同。這表示服務資訊不同，不能單憑此證明不公平。 |
| M6 | **重要資訊項目差異比例**（Information-Item Disagreement Rate，IIDR） | 參考 LangBiTe、Morales 等人的測試判斷規則；公式由本研究依客服任務定義。 | \(\displaystyle IIDR=\frac{\sum_{i=1}^{N}\sum_{k=1}^{K_i}\mathbf{1}\big(I_{ik}(A_i)\ne I_{ik}(B_i)\big)}{\sum_{i=1}^{N}K_i}\) | \(I_{ik}\) 是第 \(i\) 組配對中第 \(k\) 個可判斷的重要資訊項目，例如是否說明申請方式；\(K_i\) 是該對回答中**雙方都可判斷**的項目數。數值表示可判斷項目中有多少比例不同。須先定義各情境應檢查的項目。 |
| M7 | **群體平均毒性差距**（Group Mean Toxicity Gap，TGap） | 參考 LangFair 毒性分類器；此差距公式為本研究候選，非直接宣稱為套件現成的同名指標。 | \(\displaystyle TGap=\left|\frac{1}{N}\sum_{i=1}^{N}T(A_i)-\frac{1}{N}\sum_{i=1}^{N}T(B_i)\right|\) | \(T\) 為英文毒性分類器分數。越大表示兩群體的平均毒性分數差距越大。若兩組毒性都很高，差距仍可能接近 0，因此須同時報告兩群體各自的平均值。 |
| M8 | **反事實 ROUGE-L**（Counterfactual ROUGE-L，CROUGE-L） | LangFair。 | \(\displaystyle CROUGE\text{-}L=\frac{1}{N}\sum_{i=1}^{N}F1_{\mathrm{LCS}}(A_i,B_i)\) | 依兩個回答的最長共同詞序列計算文字重疊程度，再取平均。越高表示措辭越相近。`can refund` 與 `cannot refund` 可能仍有大量字詞重疊，因此文字相似不代表服務意思相同。 |

### 關於 M5 與 M6 的標註

- 情境歸類及服務選項、資訊項目由系統以關鍵字抽取；各情境的檢查清單預先定義，依 7 家目標航空公司官網資訊整理，見[檢查清單](檢查清單.md)。
- 關鍵字抽取的可靠性是否以人工標註資料檢查：待討論。
- 「未提及」與「無法判斷」應分開標示。兩個回答都未提及退款，即使彼此一致，也不代表客服資訊正確。
- 判斷退款等權益是否正確，仍需要票務條件與可信的航空公司政策資料。上述差異指標首先衡量的是**回答資訊是否不同**。

## 四、未列為共同客服指標的原始量測

| 原始方法 | 原始量測 | 為何不建議直接套用一般航空客服回答？ |
| --- | --- | --- |
| BiasTestGPT | 刻板印象選擇比例 | 需要特定刻板／反刻板句子配對及待測模型的機率相關資訊；一般客服自由回答不一定符合。 |
| BiasAsker | 絕對偏見率及相對偏見相關分數 | 判斷依賴其特定的群體與偏見描述問題、問題形式及回答規則；一般退款問答不一定符合。 |
| GenFair | 原始 FDR | 需符合其來源／後續案例轉換及語氣判斷規則；本研究改編比例不應直接稱為原始 FDR。 |

## 五、文獻與工具來源

- [LangFair：反事實指標定義與範例](https://cvs-health.github.io/langfair/latest/notebooks/examples/evaluations/text_generation/counterfactual_metrics_demo.html)
- [LangFair 論文](https://arxiv.org/abs/2501.03112)
- [GenFair：語氣判斷與 FDR](https://arxiv.org/html/2506.03024)
- [CAFFE：配對回答相似度與門檻](https://arxiv.org/abs/2512.16816)
- [LangBiTe：測試判斷規則](https://arxiv.org/html/2404.18558v2)
- [BiasTestGPT 論文](https://arxiv.org/abs/2302.07371)
- [BiasAsker 論文](https://arxiv.org/abs/2305.12434)

各指標分開報告、各自解讀，不合併。本表是討論用候選清單：M1、M2、M8 有 LangFair 的具體定義；標示為改編或本研究候選的指標，需明確記錄公式、模型及規則，不應宣稱為原論文提供的同名指標。
