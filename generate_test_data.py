# ==============================================================================
# AUTH: Zk
# DATE: 2026-04-04
# VER:  2.0
# DESC: Advanced test data generation script with multiprocessing & config control.
# ==============================================================================
r'''
專為「日本地標衛星空照圖」設計的測資生成腳本 (十全大補模組化版)。
特點：
1. 完全讀取外部 config.json 控制。
2. 完美無黑邊旋轉 + 空間扭曲 (Shear)。
3. 包含：縮放、旋轉、扭曲、翻轉、色彩、灰階、馬賽克、JPEG失真、模糊、雜訊、遮擋。
'''

import os
import io
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(root_dir):
    root = Path(root_dir)
    files = []
    for p in root.rglob("*"):
        if p.suffix.lower() in VALID_EXTS:
            files.append(p)
    return sorted(files)

def ensure_rgb(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

# ==========================================
# 擴充技能模組 (全數交由 Config 控制)
# ==========================================

def apply_scale(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    scale = rng.choice(cfg["scales"])
    if scale == 1.0: return img
        
    W, H = img.size
    new_W, new_H = int(W * scale), int(H * scale)
    scaled = img.resize((new_W, new_H), Image.Resampling.BICUBIC)

    start_x = (new_W - W) // 2
    start_y = (new_H - H) // 2
    return scaled.crop((start_x, start_y, start_x + W, start_y + H))


def safe_crop_rotate_shear(img, rng, out_size, cfg):
    if not cfg.get("enabled", False):
        return img.resize((out_size, out_size), Image.Resampling.BICUBIC)

    W, H = img.size
    min_side = min(W, H)
    
    # 取出 60% ~ 99% 的大畫布
    safe_size = int(rng.uniform(min_side * 0.6, min_side * 0.99))
    
    cx_min, cx_max = safe_size / 2, W - safe_size / 2
    cy_min, cy_max = safe_size / 2, H - safe_size / 2
    if cx_min > cx_max: cx_min = cx_max = W / 2
    if cy_min > cy_max: cy_min = cy_max = H / 2
    
    cx = rng.uniform(cx_min, cx_max)
    cy = rng.uniform(cy_min, cy_max)
    
    x, y = int(cx - safe_size / 2), int(cy - safe_size / 2)
    patch = img.crop((x, y, x + safe_size, y + safe_size))
    
    # 旋轉
    rot_range = cfg["rotate_angle_range"]
    angle = float(rng.uniform(rot_range[0], rot_range[1]))
    patch = patch.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    
    # 空間扭曲 (Shear)
    if cfg.get("shear_enabled", False) and rng.random() < cfg["shear_prob"]:
        sr = cfg["shear_range"]
        shear_x = float(rng.uniform(sr[0], sr[1]))
        shear_y = float(rng.uniform(sr[0], sr[1]))
        patch = patch.transform(
            patch.size, Image.Transform.AFFINE, 
            (1, shear_x, 0, shear_y, 1, 0), 
            resample=Image.Resampling.BICUBIC
        )

    # 去黑邊裁切 (有開啟Shear時安全係數需加大到1.6)
    factor = 1.6 if cfg.get("shear_enabled", False) else 1.415
    final_crop_size = int(safe_size / factor)
    offset = (safe_size - final_crop_size) // 2
    patch = patch.crop((offset, offset, offset + final_crop_size, offset + final_crop_size))
    
    return patch.resize((out_size, out_size), Image.Resampling.BICUBIC)


def apply_flip(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["h_prob"]:
        img = ImageOps.mirror(img)
    if rng.random() < cfg["v_prob"]:
        img = ImageOps.flip(img)
    return img


def apply_color_jitter(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    b_range = cfg["brightness"]
    c_range = cfg["contrast"]
    s_range = cfg["saturation"]
    
    img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(b_range[0], b_range[1])))
    img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(c_range[0], c_range[1])))
    img = ImageEnhance.Color(img).enhance(float(rng.uniform(s_range[0], s_range[1])))
    return img


def apply_grayscale(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        # 轉換成黑白後，要再轉回 RGB 以確保後續陣列處理不報錯
        img = ImageOps.grayscale(img).convert("RGB")
    return img


def apply_blur(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]: 
        radius = float(rng.uniform(cfg["radius"][0], cfg["radius"][1]))
        img = img.filter(ImageFilter.GaussianBlur(radius))
    return img

def apply_channel_shuffle(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        channels = list(img.split())
        rng.shuffle(channels)
        img = Image.merge("RGB", channels)
    return img

def apply_vignette(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        w, h = img.size
        # 產生徑向漸變遮罩
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        radius = np.sqrt(X**2 + Y**2)
        
        strength = float(rng.uniform(cfg["strength"][0], cfg["strength"][1]))
        mask = 1 - np.clip(radius * strength, 0, 1)
        mask = mask ** 1.5  # 讓漸層變得更自然柔和
        
        arr = np.asarray(img).astype(np.float32)
        arr = arr * mask[:, :, np.newaxis]
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img

def apply_solarize(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        threshold = int(rng.integers(cfg["threshold"][0], cfg["threshold"][1]))
        img = ImageOps.solarize(img, threshold=threshold)
    return img

def apply_posterize(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        bits = int(rng.integers(cfg["bits"][0], cfg["bits"][1]))
        # bits 不能低於 1，否則會報錯
        bits = max(1, bits)
        img = ImageOps.posterize(img, bits)
    return img


def apply_noise(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    arr = np.asarray(img).astype(np.float32)
    if rng.random() < cfg["prob"]: 
        sigma = float(rng.uniform(cfg["sigma"][0], cfg["sigma"][1]))
        noise = rng.normal(0, sigma, arr.shape)
        arr += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_pixelate(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        w, h = img.size
        factor = float(rng.uniform(cfg["factor"][0], cfg["factor"][1]))
        small_w, small_h = max(1, int(w * factor)), max(1, int(h * factor))
        img = img.resize((small_w, small_h), Image.Resampling.NEAREST)
        img = img.resize((w, h), Image.Resampling.NEAREST)
    return img


def apply_jpeg_compression(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        quality = int(rng.integers(cfg["quality"][0], cfg["quality"][1] + 1))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        img = Image.open(buffer).convert("RGB")
    return img


def apply_cutout(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        w, h = img.size
        area = w * h
        target_area = rng.uniform(cfg["area_ratio"][0], cfg["area_ratio"][1]) * area
        aspect_ratio = rng.uniform(cfg["aspect_ratio"][0], cfg["aspect_ratio"][1])
        
        h_c = int(round((target_area * aspect_ratio) ** 0.5))
        w_c = int(round((target_area / aspect_ratio) ** 0.5))
        
        if w_c < w and h_c < h:
            x1 = rng.integers(0, w - w_c)
            y1 = rng.integers(0, h - h_c)
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x1 + w_c, y1 + h_c], fill=(0, 0, 0))
    return img


def apply_clouds(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        w, h = img.size
        # 建立一個透明圖層
        cloud_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(cloud_layer)
        
        count = int(rng.integers(cfg["count"][0], cfg["count"][1] + 1))
        for _ in range(count):
            r = rng.integers(cfg["radius"][0], cfg["radius"][1])
            x = rng.integers(-r, w)
            y = rng.integers(-r, h)
            # 畫白色半透明的圓，alpha 值隨機 (100~200)
            alpha = int(rng.uniform(100, 200))
            draw.ellipse([x, y, x + r*2, y + r*2], fill=(255, 255, 255, alpha))
            
        # 大量高斯模糊讓圓形變成自然擴散的雲霧
        cloud_layer = cloud_layer.filter(ImageFilter.GaussianBlur(rng.uniform(15, 30)))
        # 將雲霧疊加到原圖上
        img = Image.alpha_composite(img.convert("RGBA"), cloud_layer).convert("RGB")
    return img


def apply_shadows(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        w, h = img.size
        shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shadow_layer)
        
        count = int(rng.integers(cfg["count"][0], cfg["count"][1] + 1))
        for _ in range(count):
            # 產生隨機大小的陰影區塊
            x0, y0 = rng.integers(-w//2, w), rng.integers(-h//2, h)
            x1 = x0 + rng.integers(w//2, w)
            y1 = y0 + rng.integers(h//2, h)
            alpha = int(rng.uniform(cfg["alpha"][0], cfg["alpha"][1]))
            draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, alpha))
            
        # 稍微模糊邊緣，模擬光線繞射的柔和陰影
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(rng.uniform(10, 25)))
        img = Image.alpha_composite(img.convert("RGBA"), shadow_layer).convert("RGB")
    return img


def apply_emboss(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        img = img.filter(ImageFilter.EMBOSS)
    return img


def apply_edge_enhance(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        # EDGE_ENHANCE_MORE 會產生非常強烈且不自然的銳化
        img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    return img

def apply_scanlines(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        w, h = img.size
        draw = ImageDraw.Draw(img)
        count = int(rng.integers(cfg["count"][0], cfg["count"][1] + 1))
        # 決定是橫線還是直線，以及是黑色還是白色
        is_horizontal = rng.random() < 0.5
        fill_color = (0, 0, 0) if rng.random() < 0.5 else (255, 255, 255)
        
        for _ in range(count):
            width = int(rng.integers(cfg["width"][0], cfg["width"][1] + 1))
            if is_horizontal:
                y = rng.integers(0, h - width)
                draw.rectangle([0, y, w, y + width], fill=fill_color)
            else:
                x = rng.integers(0, w - width)
                draw.rectangle([x, 0, x + width, h], fill=fill_color)
    return img

def apply_chromatic_aberration(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        # 分離 RGB 通道
        r, g, b = img.split()
        shift = int(rng.integers(cfg["shift"][0], cfg["shift"][1] + 1))
        
        # 使用 ImageChops 進行通道平移 (紅色往左上，藍色往右下)
        from PIL import ImageChops
        r_shifted = ImageChops.offset(r, -shift, -shift)
        b_shifted = ImageChops.offset(b, shift, shift)
        
        # 合併回去
        img = Image.merge("RGB", (r_shifted, g, b_shifted))
    return img

def apply_color_temp(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        w, h = img.size
        intensity = float(rng.uniform(cfg["intensity"][0], cfg["intensity"][1]))
        
        # 決定是變暖(橘黃)還是變冷(深藍)
        is_warm = rng.random() < 0.5
        color = (255, 150, 0) if is_warm else (0, 50, 255)
        
        # 建立純色圖層並與原圖進行 Alpha 混色
        overlay = Image.new("RGB", (w, h), color)
        img = Image.blend(img, overlay, intensity)
    return img

def apply_sun_glare(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        w, h = img.size
        # 建立一個白色的眩光遮罩
        glare_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(glare_layer)
        
        # 隨機決定光源中心點在四個角落的某處
        cx = rng.choice([0, w]) + rng.integers(-20, 20)
        cy = rng.choice([0, h]) + rng.integers(-20, 20)
        
        # 畫多個同心圓製造漸層強光
        max_radius = int(max(w, h) * 1.5)
        step = max_radius // 10
        for r in range(max_radius, 0, -step):
            alpha = int(255 * (1 - r/max_radius)**2) # 越靠近中心越白
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 255, 255, alpha))
            
        img = Image.alpha_composite(img.convert("RGBA"), glare_layer).convert("RGB")
    return img

from PIL import ImageChops # 如果上方沒 import 到，請補上這行

def apply_motion_blur(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        size = int(rng.integers(cfg["kernel_size"][0], cfg["kernel_size"][1]))
        angle = float(rng.uniform(0, 360))
        blended = img.copy()
        # 利用圖片平移疊加來模擬殘影
        for i in range(1, size):
            dx = int(i * np.cos(np.radians(angle)))
            dy = int(i * np.sin(np.radians(angle)))
            shifted = ImageChops.offset(img, dx, dy)
            blended = Image.blend(blended, shifted, alpha=0.5)
        return blended
    return img

def apply_moire_pattern(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        w, h = img.size
        arr = np.array(img).astype(np.float32)
        freq = float(rng.uniform(cfg["freq"][0], cfg["freq"][1]))
        intensity = float(rng.uniform(cfg["intensity"][0], cfg["intensity"][1]))
        angle = rng.uniform(0, np.pi)
        
        # 產生 2D 正弦波紋
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        wave = np.sin(freq * (xx * np.cos(angle) + yy * np.sin(angle)))
        
        # 將波紋乘上原圖亮度
        multiplier = 1.0 + wave * intensity
        arr = arr * multiplier[:, :, np.newaxis]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img

def apply_grid_dropout(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        w, h = img.size
        draw = ImageDraw.Draw(img)
        holes = int(rng.integers(cfg["holes"][0], cfg["holes"][1]))
        for _ in range(holes):
            hole_w = int(rng.integers(cfg["hole_size"][0], cfg["hole_size"][1]))
            hole_h = int(rng.integers(cfg["hole_size"][0], cfg["hole_size"][1]))
            x = rng.integers(0, max(1, w - hole_w))
            y = rng.integers(0, max(1, h - hole_h))
            draw.rectangle([x, y, x + hole_w, y + hole_h], fill=(0, 0, 0))
    return img

def apply_false_color(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        r, g, b = img.split()
        if rng.random() < 0.5:
            # 植被假色：紅綠通道對調
            img = Image.merge("RGB", (g, r, b))
        else:
            # 類熱成像：反相後提高對比
            img = ImageOps.invert(img)
            img = ImageEnhance.Color(img).enhance(1.5)
    return img

def apply_lens_dust(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        w, h = img.size
        dust_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dust_layer)
        
        count = int(rng.integers(cfg["count"][0], cfg["count"][1] + 1))
        for _ in range(count):
            r = rng.integers(cfg["size"][0], cfg["size"][1])
            x = rng.integers(0, w)
            y = rng.integers(0, h)
            # 畫出深灰色半透明圓形 (模擬灰塵)
            alpha = int(rng.uniform(50, 120))
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(30, 30, 30, alpha))
            
        # 大量高斯模糊讓灰塵看起來是在鏡頭玻璃上(失焦)
        dust_layer = dust_layer.filter(ImageFilter.GaussianBlur(rng.uniform(2.0, 5.0)))
        img = Image.alpha_composite(img.convert("RGBA"), dust_layer).convert("RGB")
    return img

def apply_gradient_light(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        w, h = img.size
        intensity = float(rng.uniform(cfg["intensity"][0], cfg["intensity"][1]))
        
        # 產生水平或垂直漸層 (1.0為原亮度，+/- intensity為明暗變化)
        is_horizontal = rng.random() < 0.5
        gradient = np.linspace(1.0 - intensity, 1.0 + intensity, w if is_horizontal else h)
        
        if rng.random() < 0.5:
            gradient = gradient[::-1] # 隨機反轉明暗方向
            
        # 擴充為 2D 矩陣
        if is_horizontal:
            mask = np.tile(gradient, (h, 1))
        else:
            mask = np.tile(gradient[:, np.newaxis], (1, w))
            
        arr = np.asarray(img).astype(np.float32)
        arr = arr * mask[:, :, np.newaxis]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img

def apply_rain_streaks(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        w, h = img.size
        rain_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rain_layer)
        
        drops = int(rng.integers(cfg["drops"][0], cfg["drops"][1] + 1))
        angle = rng.uniform(70, 110) # 模擬雨絲落下的角度
        
        for _ in range(drops):
            l = rng.integers(cfg["length"][0], cfg["length"][1])
            x = rng.integers(-20, w + 20)
            y = rng.integers(-20, h + 20)
            
            # 計算雨絲的終點
            dx = int(l * np.cos(np.radians(angle)))
            dy = int(l * np.sin(np.radians(angle)))
            
            alpha = int(rng.uniform(100, 200))
            draw.line([(x, y), (x+dx, y+dy)], fill=(200, 200, 200, alpha), width=rng.integers(1, 3))
            
        # 輕微動態模糊效果
        rain_layer = rain_layer.filter(ImageFilter.GaussianBlur(0.5))
        img = Image.alpha_composite(img.convert("RGBA"), rain_layer).convert("RGB")
    return img

def apply_salt_pepper(img, rng, cfg):
    if not cfg.get("enabled", False): return img
    if rng.random() < cfg["prob"]:
        arr = np.asarray(img).copy()
        amount = rng.uniform(cfg["amount"][0], cfg["amount"][1])
        
        # 產生均勻分佈的隨機矩陣
        rand_matrix = rng.random(arr.shape[:2])
        
        # Salt (純白)
        arr[rand_matrix < (amount / 2)] = [255, 255, 255]
        # Pepper (純黑)
        arr[(rand_matrix > (amount / 2)) & (rand_matrix < amount)] = [0, 0, 0]
        
        return Image.fromarray(arr)
    return img


# ==========================================
# 核心生成循環
# ==========================================

def generate_one_image(img_path, output_root, config, rng):
    img = Image.open(img_path)
    img = ensure_rgb(img)

    stem = img_path.stem
    out_dir = output_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_cfg = config["generation"]
    aug_cfg = config["augmentations"]
    
    # 🌟 新增：讀取針對物件裁切的機率 (預設 0.75)
    object_crop_prob = gen_cfg.get("object_crop_prob", 0.75)

    for i in range(gen_cfg["num_per_image"]):
        valid_patch = False
        attempts = 0
        
        while not valid_patch and attempts < gen_cfg["max_attempts"]:
            size = int(rng.integers(gen_cfg["min_size"], gen_cfg["max_size"] + 1))
            
            # 1. 尺度變換 (先對大圖做，這樣後面裁切的座標才會正確對應)
            scaled_img = apply_scale(img, rng, aug_cfg["scale"])
            
            # ==========================================
            # 🌟 2. 75% 萃取物件 vs 25% 隨機背景
            # ==========================================
            if rng.random() < object_crop_prob:
                # 🎯 [75% 機率] 萃取特定物件 (Positive Sample)
                # 🎯 [75% 機率] 萃取特定物件 (Positive Sample)
                W, H = scaled_img.size
                
                # --- 自動特徵尋找系統 (Edge Center of Mass) ---
                
                # 1. 將 PIL 圖片轉為 OpenCV 灰階矩陣
                img_cv = np.array(scaled_img)
                # 確保圖片是 RGB 格式再轉灰階
                if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
                    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = img_cv # 已經是灰階
                
                # 2. 使用 Canny 演算法抓取所有「輪廓與線條」
                edges = cv2.Canny(img_gray, 50, 150)
                
                # 3. 計算畫面上所有線條的「物理重心」
                M = cv2.moments(edges)
                
                # 防呆：確保畫面不是完全空白的
                if M["m00"] != 0:
                    obj_cx = int(M["m10"] / M["m00"]) # 線條密集的 X 軸中心
                    obj_cy = int(M["m01"] / M["m00"]) # 線條密集的 Y 軸中心
                else:
                    # 如果真的什麼線條都沒有，才無奈退回正中心
                    obj_cx = W // 2  
                    obj_cy = H // 2
                # ---------------------------------------------
                
                # 根據計算出的物件中心點，推算裁切框的左上角 (x, y)，並防止超出邊界
                x = int(max(0, min(obj_cx - size / 2, W - size)))
                y = int(max(0, min(obj_cy - size / 2, H - size)))
                
                # 直接裁切物件
                patch = scaled_img.crop((x, y, x + size, y + size))
                
                # (備註：如果你希望針對物件裁切時也能套用旋轉與空間扭曲，
                # 建議未來可以修改你的 safe_crop_rotate_shear 讓它支援傳入中心座標)
                
            else:
                # 🎲 [25% 機率] 純隨機背景裁切 (Negative Sample)
                # 呼叫你原本寫好的安全裁切函數，讓它在大圖上隨機找地方切
                patch = safe_crop_rotate_shear(scaled_img, rng, size, aug_cfg["crop_rotate_shear"])

            # ==========================================
            # 3. 依序套用剩下的所有干擾技能
            # ==========================================
            patch = apply_flip(patch, rng, aug_cfg["flip"])
            patch = apply_chromatic_aberration(patch, rng, aug_cfg["chromatic_aberration"])
            patch = apply_color_temp(patch, rng, aug_cfg["color_temp"])
            patch = apply_color_jitter(patch, rng, aug_cfg["color_jitter"])
            patch = apply_grayscale(patch, rng, aug_cfg["grayscale"])
            patch = apply_channel_shuffle(patch, rng, aug_cfg["channel_shuffle"])
            patch = apply_solarize(patch, rng, aug_cfg["solarize"])
            patch = apply_posterize(patch, rng, aug_cfg["posterize"])
            patch = apply_vignette(patch, rng, aug_cfg["vignette"])
            
            patch = apply_clouds(patch, rng, aug_cfg["clouds"])
            patch = apply_shadows(patch, rng, aug_cfg["shadows"])
            patch = apply_sun_glare(patch, rng, aug_cfg["sun_glare"])
            patch = apply_scanlines(patch, rng, aug_cfg["scanlines"])
            patch = apply_motion_blur(patch, rng, aug_cfg["motion_blur"])
            patch = apply_moire_pattern(patch, rng, aug_cfg["moire_pattern"])
            patch = apply_false_color(patch, rng, aug_cfg["false_color"])
            patch = apply_grid_dropout(patch, rng, aug_cfg["grid_dropout"])
            patch = apply_pixelate(patch, rng, aug_cfg["pixelate"])
            patch = apply_jpeg_compression(patch, rng, aug_cfg["jpeg_compression"])
            patch = apply_blur(patch, rng, aug_cfg["blur"])
            patch = apply_noise(patch, rng, aug_cfg["noise"])
            patch = apply_emboss(patch, rng, aug_cfg["emboss"])
            patch = apply_edge_enhance(patch, rng, aug_cfg["edge_enhance"])
            patch = apply_cutout(patch, rng, aug_cfg["cutout"])
            patch = apply_gradient_light(patch, rng, aug_cfg["gradient_light"])
            patch = apply_lens_dust(patch, rng, aug_cfg["lens_dust"])
            patch = apply_rain_streaks(patch, rng, aug_cfg["rain_streaks"])
            patch = apply_salt_pepper(patch, rng, aug_cfg["salt_pepper"])

            # 檢測變異度防呆
            arr = np.asarray(patch)
            if np.std(arr) > gen_cfg["std_threshold"]:  
                valid_patch = True
            else:
                attempts += 1 
        
        # 保底機制
        if not valid_patch:
            # 如果真的切不出好圖片，就拿原圖直接硬縮放交差
            patch = img.resize((size, size), Image.Resampling.BICUBIC)

        name = f"{stem}_{i:04d}_{size}.png"
        patch.save(out_dir / name)

# ==========================================
# 多核心任務包裝器 (確保平行運算時不會出錯)
# ==========================================
def process_single_image_task(args):
    img_path, output_dir, config, task_seed = args
    # 為每個 CPU 核心建立獨立的亂數種子，保證每次產生的圖片都一樣
    rng = np.random.default_rng(task_seed)
    
    try:
        generate_one_image(img_path, output_dir, config, rng)
        return True, img_path.name
    except Exception as e:
        return False, f"{img_path.name} 發生錯誤: {str(e)}"


# ==========================================
# 主程式 (多核心超跑版 + tqdm進度條)
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", type=str, default="config.json", help="設定檔路徑")
    # 🌟 新增：允許使用者指定要用幾顆 CPU 核心 (預設火力全開)
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="使用的 CPU 核心數")
    
    args = parser.parse_args()

    # 讀取 JSON 設定檔
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    total = len(images)

    if total == 0:
        print(f"❌ 在 {input_dir} 找不到任何圖片！")
        return

    print(f"📄 成功載入設定檔: {args.config}")
    print(f"🔥 大魔王干擾技能啟動！準備生成終極測資...")
    print(f"🚀 啟動多核心引擎：分配了 {args.workers} 個 CPU 核心同步運算。")
    print("-" * 50)

    # 從設定檔取得基礎隨機種子
    base_seed = config["generation"].get("seed", 42)

    # 1. 將所有圖片打包成「任務清單」，並分配獨立的 seed
    tasks = []
    for i, img_path in enumerate(images):
        tasks.append((img_path, output_dir, config, base_seed + i))

    # 2. 啟動多核心進程池 (ProcessPoolExecutor)
    success_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # 將任務交給 CPU，並綁定 tqdm 動態進度條
        futures = {executor.submit(process_single_image_task, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=total, desc="🔥 產圖進度", unit="張原圖"):
            success, msg = future.result()
            if success:
                success_count += 1
            else:
                # 如果某張圖報錯，用 tqdm.write 顯示警告才不會把進度條弄亂
                tqdm.write(f"⚠️ 警告: {msg}")

    print("-" * 50)
    print(f"🎉 任務完成！成功處理了 {success_count} / {total} 張原始圖片。")
    print("Finished generating satellite test images.")

if __name__ == "__main__":
    main()