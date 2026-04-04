# 🤖 AI 甲子園 - 測資生成與裁判訓練系統 (AI Discriminator Pipeline)

🌍 **Language Select / 言語選択**
* [繁體中文 (Traditional Chinese)](#繁體中文-traditional-chinese)
* [日本語 (Japanese)](#日本語-japanese)

---

# 繁體中文 (Traditional Chinese)

這是一個專為「日本地標衛星空照圖」設計的高階自動化測資生成與防禦型 AI 裁判訓練流水線。本專案不僅包含資料擴增，更整合了高階特徵分析、以及能自動剔除「一眼假」廢圖的強健訓練機制。

## 📂 核心模組詳細介紹 (Modules in Detail)

本專案將複雜的流程拆分為多個獨立模組，確保程式碼的可維護性：

### 1. 🚀 自動化流水線 (Automation Pipelines)
* **`auto_gen_test_data.py` (一鍵測資生成打包)**
  * **功能**：負責自動化「出考卷」的所有流程。
  * **亮點**：內建防呆導航與大掃除機制（自動處理 Windows 唯讀檔案刪除報錯）。它會依序呼叫核心生成器與展平工具，並自動將最終測資打包成 `fin_test.zip` 與 `test.zip`。
* **`auto_pipline.py` (一鍵裁判訓練)**
  * **功能**：負責 AI 裁判的自動化訓練準備。
  * **亮點**：執行嚴格的環境清理（徹底清除舊的假圖與隔離區），接著呼叫假圖採樣腳本，最後啟動裁判訓練，並在結尾輸出總耗時。

### 2. 🧠 核心處理引擎 (Core Engines)
* **`generate_test_data.py` (測資生成引擎)**
  * **功能**：讀取 `config.json` 設定，對原始圖片進行各種干擾與擴增。
  * **亮點**：支援多進程 (Multiprocessing) 加速。內建完美無黑邊旋轉、空間扭曲 (Shear)、色彩偏移、雜訊、模糊、馬賽克與遮擋等「十全大補」技能，高度自訂化。
* **`train_discriminator.py` (防禦型 AI 裁判訓練)**
  * **功能**：訓練一個圖片二元分類器 (真/假)。
  * **亮點**：引入了 **「廢圖自動隔離機制 (Bad-Data Quarantine)」**。在訓練過程中，若模型以 >90% 的信心判定某張圖為假圖，代表該圖破綻太大，系統會動態將其移至 `bad_data_quarantine` 資料夾，避免模型學到過於簡單的特徵。

### 3. 🔍 分析與資料處理工具 (Utilities)
* **`color_analyzer.py` (終極特徵分析報告)**
  * **功能**：量化比較官方真圖與您生成的假圖。
  * **亮點**：使用 OpenCV (Canny) 進行邊緣密度分析，並計算極暗 (dark_ratio) 與極亮 (highlight_ratio) 區域。程式最終會給出具體的戰術建議（例如：陰影缺乏，建議開啟 config 中的 shadows）。
* **`flatten_and_rename.py` (資料夾展平與重命名)**
  * **功能**：將帶有標籤的樹狀資料夾轉換為單一資料夾結構。
  * **亮點**：支援設定隨機種子 (Seed) 確保實驗可重現性。自動從 1 開始重新命名，並同步產出對應的 `mapping.csv`，方便後續對答案。

## 🛠️ 環境需求 (Prerequisites)

```bash
# 建議使用 Python 3.8 以上版本
pip install torch torchvision opencv-python pillow numpy tqdm
