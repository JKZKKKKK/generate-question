# ==============================================================================
# AUTH: Zk
# DATE: 2026-04-04
# VER:  1.0
# DESC: Color and advanced feature analyzer for image datasets.
# ==============================================================================

import os
import cv2
import numpy as np

def analyze_folder(folder_path):
    print(f"🔍 正在分析資料夾: {folder_path} ...")
    
    metrics = {
        'brightness': [], 'contrast': [],
        'R': [], 'G': [], 'B': [],
        'saturation': [], 'sharpness': [], 'noise_level': [],
        # --- ADV FEAT ---
        'edge_density': [],  # 邊緣/紋理密度 (%)
        'dark_ratio': [],    # 極暗區域比例 (%)
        'highlight_ratio': []# 極亮/過曝比例 (%)
    }
    
    valid_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not valid_images:
        print(f"⚠️ 找不到圖片！請確認路徑: {folder_path}")
        return None

    for filename in valid_images:
        filepath = os.path.join(folder_path, filename)
        img = cv2.imread(filepath)
        if img is None: continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 基礎亮度、對比、RGB
        metrics['brightness'].append(np.mean(img_gray))
        metrics['contrast'].append(np.std(img_gray))
        metrics['R'].append(np.mean(img_rgb[:,:,0]))
        metrics['G'].append(np.mean(img_rgb[:,:,1]))
        metrics['B'].append(np.mean(img_rgb[:,:,2]))
        
        # 飽和度、銳利度、雜訊
        metrics['saturation'].append(np.mean(img_hsv[:,:,1]))
        metrics['sharpness'].append(cv2.Laplacian(img_gray, cv2.CV_64F).var())
        metrics['noise_level'].append(np.std(img_gray))

        # --- NEW METRICS ---
        # 1. 邊緣密度 (使用 Canny 偵測，計算邊緣像素佔整體的百分比)
        edges = cv2.Canny(img_gray, 100, 200)
        metrics['edge_density'].append(np.mean(edges > 0) * 100)
        
        # 2. 極暗區域比例 (亮度低於 30 的像素百分比)
        metrics['dark_ratio'].append(np.mean(img_gray < 30) * 100)
        
        # 3. 極亮區域比例 (亮度高於 225 的像素百分比)
        metrics['highlight_ratio'].append(np.mean(img_gray > 225) * 100)

    return {k: np.mean(v) for k, v in metrics.items()}

def main():
    official_dir = "./discriminator_dataset/real"
    my_fake_dir = "./discriminator_dataset/fake"
    
    official_stats = analyze_folder(official_dir)
    my_stats = analyze_folder(my_fake_dir)
    
    if not official_stats or not my_stats:
        return

    print("\n" + "="*55)
    print("📊 【AI甲子園 - 終極特徵分析報告】")
    print("="*55)
    
    print(f"{'指標':<15} | {'官方真圖 (目標)':<15} | {'你的假圖 (現狀)':<15} | {'差距'}")
    print("-" * 65)
    
    for key in official_stats.keys():
        off_val = official_stats[key]
        my_val = my_stats[key]
        diff = my_val - off_val
        
        ratio = my_val / off_val if off_val != 0 else 1.0
        
        alert = ""
        if ratio > 1.25 or (diff > 5 and key in ['dark_ratio', 'highlight_ratio', 'edge_density']): 
            alert = " 🔴 (太高)"
        elif ratio < 0.75 or (diff < -5 and key in ['dark_ratio', 'highlight_ratio', 'edge_density']): 
            alert = " 🔵 (太低)"
            
        print(f"{key:<17} | {off_val:>14.2f} | {my_val:>14.2f} | {diff:>6.2f} {alert}")

    print("\n💡 【高階戰術修改建議 (針對 config.json)】")
    
    # 亮度與對比建議
    b_ratio = official_stats['brightness'] / my_stats['brightness']
    print(f"👉 亮度中心點建議 : {b_ratio:.2f} 倍。")
    c_ratio = official_stats['contrast'] / my_stats['contrast']
    print(f"👉 對比中心點建議 : {c_ratio:.2f} 倍。")
    
    # 銳利度建議
    sharp_ratio = official_stats['sharpness'] / my_stats['sharpness']
    if sharp_ratio < 0.8:
        print(f"👉 銳利度 (Sharpness) : 你的圖太銳利！請降低 edge_enhance 機率，或微調 blur。")
    elif sharp_ratio > 1.2:
        print(f"👉 銳利度 (Sharpness) : 你的圖太糊！請減小 blur 範圍，或提高 edge_enhance 機率。")

    # --- ADV TACTIC RECOM ---
    edge_diff = official_stats['edge_density'] - my_stats['edge_density']
    if edge_diff > 3.0:
        print(f"🛡️ [紋理缺乏] 官方圖有更多的建築/線條紋理。考慮減少 blur，或輕微開啟 `noise` (雜訊) 增加顆粒感。")
    elif edge_diff < -3.0:
        print(f"🛡️ [紋理過雜] 你的圖看起來太碎裂。考慮微調 blur 來柔化，或檢查是否過度壓縮。")

    if official_stats['dark_ratio'] > my_stats['dark_ratio'] + 2.0:
        print(f"🌘 [陰影缺乏] 官方圖有大面積黑影！建議在 config.json 開啟 `shadows` (局部陰影) 或 `vignette` (暗角)。")
    
    if official_stats['highlight_ratio'] > my_stats['highlight_ratio'] + 2.0:
        print(f"☀️ [過曝缺乏] 官方圖有嚴重反光或死白！建議在 config.json 開啟 `sun_glare` (太陽耀光)。")

if __name__ == "__main__":
    main()