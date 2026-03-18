import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def sample_images(source_dir, dest_dir, sample_size=1000):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # 1. 確保來源資料夾存在
    if not source_path.exists():
        print(f"❌ 找不到來源資料夾: {source_dir}")
        return

    # 2. 建立目標資料夾 (如果不存在的話會自動建立)
    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"🔍 正在掃描 {source_dir} 中的所有圖片...")
    
    # 支援的圖片格式
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    
    # 找出所有圖片檔案 (包含所有子資料夾)
    all_images = [p for p in source_path.rglob("*") if p.suffix.lower() in valid_exts]
    total_images = len(all_images)
    
    print(f"✅ 共找到 {total_images} 張圖片。")

    if total_images == 0:
        print("❌ 沒有圖片可以抽樣，程式結束。")
        return

    # 3. 決定要抽幾張 (如果總圖片數少於 sample_size，就全抽)
    actual_sample_size = min(sample_size, total_images)
    print(f"🎲 準備隨機抽出 {actual_sample_size} 張圖片...")

    # 執行隨機抽樣
    sampled_images = random.sample(all_images, actual_sample_size)

    # 4. 開始複製檔案 (加上進度條)
    for img_path in tqdm(sampled_images, desc="📂 複製圖片中"):
        # 為了避免不同資料夾內有同名檔案被覆蓋，我們把「原資料夾名稱」加到檔名前面
        # 例如: landmark_01 / img_001.png -> landmark_01_img_001.png
        parent_name = img_path.parent.name
        new_filename = f"{parent_name}_{img_path.name}"
        
        dest_file_path = dest_path / new_filename
        
        # 複製檔案 (copy2 會保留原本的建立/修改時間等 Metadata)
        shutil.copy2(img_path, dest_file_path)

    print(f"\n🎉 抽樣完成！已經將 {actual_sample_size} 張圖片複製到: {dest_dir}")

if __name__ == "__main__":
    # ==========================================
    # 在這裡設定你的來源與目標路徑
    # ==========================================
    SOURCE_DIRECTORY = "./data/test"                 # 你用 config 生成的假測資資料夾
    DESTINATION_DIRECTORY = "./discriminator_dataset/fake" # 準備給 AI 裁判看的 fake 資料夾
    SAMPLE_COUNT = 420                               # 你想要隨機抽幾張
    
    sample_images(SOURCE_DIRECTORY, DESTINATION_DIRECTORY, SAMPLE_COUNT)
    
    #python generate_test_data.py --input_dir ./data/landmarks --output_dir ./augmented_fake --config config.json --workers 8