# ==============================================================================
# AUTH: Zk
# DATE: 2026-04-04
# VER:  1.0
# DESC: Randomly samples fake images and copies them with parent directory prefixes.
# ==============================================================================

import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def sample_images(source_dir, dest_dir, sample_size=1000):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # --- 1. CHK SRC (檢查來源) ---
    if not source_path.exists():
        print(f"❌ 找不到來源資料夾: {source_dir}")
        return

    # --- 2. INIT DEST (初始化目標) ---
    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"🔍 正在掃描 {source_dir} 中的所有圖片...")
    
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    
    # --- 3. GET ALL IMGS (遞歸取得所有圖片) ---
    all_images = [p for p in source_path.rglob("*") if p.suffix.lower() in valid_exts]
    total_images = len(all_images)
    
    print(f"✅ 共找到 {total_images} 張圖片。")

    if total_images == 0:
        print("❌ 沒有圖片可以抽樣，程式結束。")
        return

    # --- 4. CALC SIZ (計算實際抽樣數) ---
    actual_sample_size = min(sample_size, total_images)
    print(f"🎲 準備隨機抽出 {actual_sample_size} 張圖片...")

    sampled_images = random.sample(all_images, actual_sample_size)

    # --- 5. EXEC CPY (執行複製與重命名) ---
    for img_path in tqdm(sampled_images, desc="📂 複製圖片中"):
        # 為了避免不同資料夾內有同名檔案被覆蓋，把「原資料夾名稱」加到檔名前面
        # 例如: landmark_01 / img_001.png -> landmark_01_img_001.png
        parent_name = img_path.parent.name
        new_filename = f"{parent_name}_{img_path.name}"
        
        dest_file_path = dest_path / new_filename
        
        # 複製檔案 (copy2 會保留原本的 Metadata)
        shutil.copy2(img_path, dest_file_path)

    print(f"\n🎉 抽樣完成！已經將 {actual_sample_size} 張圖片複製到了 {dest_dir}。")

if __name__ == "__main__":
    # --- DFLT RUN (預設執行參數) ---
    sample_images("./discriminator_dataset/all_fakes", "./discriminator_dataset/fake", 1000)