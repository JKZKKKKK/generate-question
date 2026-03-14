# 🌍 衛星空照圖 AI 測資生成與資料擴充引擎
> **Satellite Image Data Augmentation & Test-bed Engine**

專為「衛星空照圖」與「無人機影像辨識」任務打造的終極資料擴充神器。
無論你是要為 AI 模型進行大規模的資料擴充（Data Augmentation）以解決稀有地標資料量不足的問題，還是要生成高難度的競賽測資（如：AI 甲子園），本引擎都能完美勝任。

---

## ✨ 核心特色 (Key Features)

* 📂 **無縫接軌 AI 訓練框架**：自動讀取並繼承原始輸入的「類別資料夾 (Class Folders)」架構，擴充後的資料可直接餵給 PyTorch `ImageFolder` 或 TensorFlow `ImageDataGenerator`。
* 🚀 **多核心超跑加速**：內建 `ProcessPoolExecutor` 全核心平行運算，搭配 `tqdm` 動態進度條，處理數千張高解析度影像只需幾分鐘。
* 🎛️ **30+ 招干擾模組 (1★~5★)**：涵蓋天氣變化（雲霧、雨絲）、感測器瑕疵（入塵、壞點）、光學干擾（色差、耀光）到極限破壞（空間錯切、馬賽克）。全數可透過 `config.json` 一鍵開關。
* 🎯 **智慧正負樣本裁切 (Negative Sampling)**：可自訂目標萃取機率，強迫模型學習區分「地標主體」與「無效背景」，大幅降低實戰誤判率。

---

## 📂 資料夾結構要求 (Directory Structure)

為了讓程式能自動替你的擴充圖片標記正確的類別，請將你的原始圖片依照「類別名稱」放在對應的子資料夾中。**即使每個地標只有 2~3 張圖也沒問題！**

**📥 輸入結構 (Input)：**
```text
dataset/
 ├── landmark_1/
 │    ├── img_01.jpg
 │    └── img_02.jpg
 └── landmark_2/
      ├── img_03.jpg
      └── img_04.jpg
