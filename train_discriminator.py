import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import time

# 🌟 修復核心：定義一個正式的類別來處理隨機噪點，這樣 Windows 才能識別
class AddRandomNoise(object):
    def __init__(self, noise_level=0.02):
        self.noise_level = noise_level

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.noise_level

    def __repr__(self):
        return self.__class__.__name__ + f'(noise_level={self.noise_level})'

def train_robust_discriminator():
    data_dir = './discriminator_dataset'

    # 使用我們剛剛定義的 AddRandomNoise，取代原來的 lambda
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 3.0)),
        transforms.ToTensor(),
        AddRandomNoise(0.02), # <--- 這裡改用正式物件
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # 注意：如果你的電腦依然報錯，可以暫時把 num_workers 設為 0 (雖然會慢一點點但最保險)
    dataloader = DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 目前使用的運算設備: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001)

    print("⚖️ 開始執行「防禦型」AI 裁判訓練...")
    
    for epoch in range(5):
        start_time = time.time()
        model.train() 
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        accuracy = 100 * correct / total
        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/5] - 耗時: {epoch_time:.1f} 秒 | 裁判準確率: {accuracy:.2f}%')

if __name__ == '__main__':
    train_robust_discriminator()