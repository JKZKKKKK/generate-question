# ==============================================================================
# AUTH: Zk
# DATE: 2026-04-04
# VER:  1.0
# DESC: Robust discriminator training script with bad-data quarantine feature.
# ==============================================================================

import os
import shutil
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

# --- 1. RND NOISE OBJ ---
class AddRandomNoise(object):
    def __init__(self, noise_level=0.02):
        self.noise_level = noise_level

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.noise_level

    def __repr__(self):
        return self.__class__.__name__ + f'(noise_level={self.noise_level})'

# --- 2. CUST DS (W/ PATH) ---
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return (original_tuple[0], original_tuple[1], path)

def train_robust_discriminator():
    data_dir = './discriminator_dataset'
    
    # 建立異常隔離區，用來放那些「一眼假」的廢圖
    bad_data_dir = './bad_data_quarantine'
    os.makedirs(bad_data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), # 翻轉可以保留，不會破壞畫質
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 目前使用的設備: {device}")

    model = models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion_individual = nn.CrossEntropyLoss(reduction='none')
    
    # --- FIX 3: LR ADJ ---
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    print("⚖️ 開始執行「防禦型」AI 裁判訓練，並啟動『自動剔除明顯假圖』機制...")
    
    # 門檻設定：如果模型有超過 90% 的信心認定它是假圖，就代表假得太明顯了，直接踢掉！
    CONFIDENCE_THRESHOLD = 0.90

    for epoch in range(5):
        # --- CORE FIX 1: DYN DS RELOAD ---
        dataset = ImageFolderWithPaths(data_dir, transform=transform)
        
        # 動態尋找 'fake' 資料夾對應的標籤數字 (通常是 0，但讓程式自己抓最保險)
        if 'fake' in dataset.class_to_idx:
            fake_label_idx = dataset.class_to_idx['fake']
        else:
            fake_label_idx = 0 

        dataloader = DataLoader(
            dataset, 
            batch_size=64, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True
        )

        start_time = time.time()
        model.train() 
        correct = 0
        total = 0
        kicked_count = 0 
        
        for images, labels, paths in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            
            # --- CORE FIX 2: SOFTMAX PROB ---
            probabilities = torch.softmax(outputs, dim=1)
            
            # 計算 Loss (為了反向傳播訓練模型)
            individual_losses = criterion_individual(outputs, labels)
            
            # 🔍 檢查是否有「一眼假」的爛圖
            for i in range(len(labels)):
                # 只針對標籤為 fake 的圖片進行檢查
                if labels[i].item() == fake_label_idx:
                    prob_is_fake = probabilities[i][fake_label_idx].item()
                    
                    # 如果模型超級有自信它就是假圖 (>90%)
                    if prob_is_fake > CONFIDENCE_THRESHOLD:  
                        bad_img_path = paths[i]
                        # 確保檔案還在，才進行搬移
                        if os.path.exists(bad_img_path):
                            print(f"🚨 抓到明顯廢圖 (假圖特徵太明顯，被識破機率: {prob_is_fake*100:.1f}%)，踢入隔離區: {os.path.basename(bad_img_path)}")
                            shutil.move(bad_img_path, os.path.join(bad_data_dir, os.path.basename(bad_img_path)))
                            kicked_count += 1

            # 模型依然需要平均 Loss 來更新權重學習
            loss = individual_losses.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        accuracy = 100 * correct / total
        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/5] - 耗時: {epoch_time:.1f} 秒 | 裁判準確率: {accuracy:.2f}% | 🗑️ 本回合剔除: {kicked_count} 張明顯假圖')

if __name__ == '__main__':
    train_robust_discriminator()