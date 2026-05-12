import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

class RPSDataset(Dataset):
    def __init__(self, folder_path):
        self.images = []
        self.labels = []
        label_map = {'rock': 0, 'paper': 1, 'scissors': 2}
        
        for category, label_idx in label_map.items():
            category_path = os.path.join(folder_path, category)
            if not os.path.exists(category_path):
                subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
                if subdirs:
                    category_path = os.path.join(folder_path, subdirs[0], category)
            if not os.path.exists(category_path):
                continue
                
            print(f"📂 正在載入 {category} 的圖片...")
            for filename in os.listdir(category_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(category_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        # 沿用邊緣特徵，以對抗背景干擾
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                        edges = cv2.Canny(blurred, 30, 150)
                        resized = cv2.resize(edges, (64, 64))
                        
                        # PyTorch CNN 需要的格式為 (Channels, Height, Width)
                        feature = np.expand_dims(resized, axis=0) / 255.0
                        self.images.append(feature)
                        self.labels.append(label_idx)
                        
        self.images = torch.tensor(np.array(self.images), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class SimpleCNN(nn.Module):
    """輕量級 CNN 網路架構"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一層卷積
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二層卷積
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 全連接層
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16) # 攤平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'dataset', 'train')
    test_dir = os.path.join(base_dir, 'dataset', 'test')
    demo_dir = os.path.join(base_dir, 'demo')

    print("=== [CNN] 開始準備資料集 ===")
    train_dataset = RPSDataset(train_dir)
    test_dataset = RPSDataset(test_dir)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    if len(train_dataset) == 0:
        print("❌ 錯誤：找不到任何圖片！")
        return

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n=== 開始訓練 CNN (約需 1~2 分鐘) ===")
    epochs = 15
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    print("\n=== 評估 CNN 模型 ===")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"🎯 CNN 模型準確率: {accuracy * 100:.2f}%\n")
    print(classification_report(all_labels, all_preds, target_names=['Rock', 'Paper', 'Scissors']))

    # 匯出 ONNX
    os.makedirs(demo_dir, exist_ok=True)
    onnx_path = os.path.join(demo_dir, 'rps_cnn.onnx')
    
    dummy_input = torch.randn(1, 1, 64, 64)
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=['input'], output_names=['output'])
    print(f"✅ 模型已成功匯出為 ONNX 格式: {onnx_path}")
    print("您現在可以使用 demo/carema_cnn.py 來進行極速預測了！")

if __name__ == "__main__":
    main()
