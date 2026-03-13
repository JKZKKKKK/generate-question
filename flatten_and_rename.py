r"""
將目錄內所有子目錄的圖片拷貝到單一目錄，隨機打亂後從 1 開始依序命名

使用方法：
python flatten_and_rename.py --input_dir data/test_generated --output_dir data/flattened --seed 42
C:\Users\bluer\AppData\Local\Programs\Python\Python310\python.exe C:\Users\bluer\OneDrive\Desktop\AI甲子園\測資生成工具程式\flatten_and_rename.py --input_dir C:\Users\bluer\OneDrive\Desktop\AI甲子園\AI2026LabV3Src\data\test --output_dir C:\Users\bluer\OneDrive\Desktop\AI甲子園\AI2026LabV3Src\data\fin_test --seed 42

# 基本用法
python flatten_and_rename.py --input_dir <來源目錄> --output_dir <目標目錄>

# 指定隨機種子（可重現相同順序）
python flatten_and_rename.py --input_dir <來源目錄> --output_dir <目標目錄> --seed 42

參數說明：
--input_dir: 來源目錄，會遞歸收集所有子目錄的圖片
--output_dir: 目標目錄，所有圖片會拷貝到這裡並重新命名
--seed: 隨機種子（可選），確保每次打亂順序一致
"""

import argparse
import csv
import shutil
from pathlib import Path
import random

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(input_dir):
    """遞歸收集所有圖片檔案"""
    input_path = Path(input_dir)
    images = []
    
    for file_path in input_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTS:
            images.append(file_path)
    
    return images


def main():
    parser = argparse.ArgumentParser(description="將子目錄圖片拷貝到單一目錄並隨機重新命名")
    parser.add_argument("--input_dir", required=True, help="來源目錄")
    parser.add_argument("--output_dir", required=True, help="目標目錄")
    parser.add_argument("--seed", type=int, default=None, help="隨機種子（可選）")
    
    args = parser.parse_args()
    
    # 設定隨機種子
    if args.seed is not None:
        random.seed(args.seed)
    
    # 建立輸出目錄
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 收集所有圖片
    print(f"正在收集 {args.input_dir} 內的所有圖片...")
    images = collect_images(args.input_dir)
    print(f"找到 {len(images)} 張圖片")
    
    if len(images) == 0:
        print("沒有找到任何圖片")
        return
    
    # 隨機打亂
    random.shuffle(images)
    print("已隨機打亂順序")
    
    # 拷貝並重新命名，同步寫入對應紀錄
    record_path = output_path / "mapping.csv"

    print("開始拷貝並重新命名...")
    with record_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "new_name", "source_dir", "source_path"])

        for idx, src_path in enumerate(images, start=1):
            ext = src_path.suffix  # 保留原附檔名
            dst_name = f"{idx}{ext}"
            dst_path = output_path / dst_name

            shutil.copy2(src_path, dst_path)
            writer.writerow([idx, dst_name, str(src_path.parent), str(src_path)])
            print(src_path.parent.name, end=", " if idx < len(images) else "\n")
    
    print(f"完成！所有圖片已拷貝至 {args.output_dir}")
    print(f"對應紀錄已輸出至 {record_path}")


if __name__ == "__main__":
    main()
