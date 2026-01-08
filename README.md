# TikTok-project-analyze-inspect-demo

## Use Tiktok data to use analyze

ENV
- Lang: python
- IDE: colab

class
- Inspect_and_analyze_data
- hypothesis
- seaborn
- regresission
- machine learning

流程（EDA $\rightarrow$ 特徵工程 $\rightarrow$ NLP 處理 $\rightarrow$ 模型調優 $\rightarrow$ 評估）

# TikTok 影片分類：機器學習專案分析報告

## 📊 專案流程概覽
本專案透過結構化數據與文字特徵，建立了一個能自動辨別影片是否為「事實陳述 (Claim)」或「個人觀點 (Opinion)」的分類系統。

### 1. 核心開發流程
1. **數據探索 (EDA)**：使用 `Pandas` 檢查遺失值、重複值，並確認目標變數 `claim_status` 的比例。
2. **特徵工程 (Feature Engineering)**：
   - 計算 `text_length`（字幕長度）。
   - 對分類變數（如驗證狀態）進行 `One-hot Encoding`。
3. **文字處理 (NLP)**：
   - 利用 `CountVectorizer` 提取高頻 2-grams 與 3-grams 片語特徵。
4. **資料切分 (Data Splitting)**：
   - 將數據切分為 **Train (60%)** / **Val (20%)** / **Test (20%)** 三組，確保評估的公平性。
5. **模型調優 (Grid Search CV)**：
   - 同時測試 **Random Forest** 與 **XGBoost**。
   - 以 `Recall` 作為優化目標（確保不漏掉任何潛在違規影片）。
6. **最終測試與分析**：
   - 使用 **Confusion Matrix** 評估誤差。
   - 繪製 **Feature Importance** 找出最具影響力的關鍵特徵。

---

## 🛠 使用工具與模型
| 類別 | 使用工具 | 具體功能 |
| :--- | :--- | :--- |
| **資料架構** | `Pandas`, `NumPy` | 矩陣運算、資料清理、欄位合併 (concat)。 |
| **視覺化** | `Matplotlib`, `Seaborn` | 分佈直方圖、熱力圖、特徵重要性圖表。 |
| **NLP 提取** | `CountVectorizer` | 將雜亂文字轉為數值計數矩陣。 |
| **核心模型** | `RandomForestClassifier` | 透過多棵決策樹整合，提供穩定且精準的分類。 |
| **進階模型** | `XGBoost` | 利用梯度提升技術追求極致的預測效能。 |

---

## 📈 模型評估指標說明
在專案中，我們使用了 `classification_report` 來觀察三個關鍵指標：
* **Precision (精確率)**：預測為 Claim 的影片中，有多少真的是 Claim？
* **Recall (召回率)**：實際是 Claim 的影片中，我們成功抓到了多少？（專案重心）
* **F1-Score**：精確率與召回率的平衡分數。

---

## 🔍 關鍵發現
透過 **Feature Importance (特徵重要性)** 圖表，我們發現：
- **影片互動數據**（觀看、按讚、分享數）對分類有極高的貢獻度。
- **NLP 片語特徵** 能有效幫助模型捕捉到隱藏在文字中的「語氣」，從而區分正式報導與日常隨筆。
