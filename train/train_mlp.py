import os
import cv2
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def extract_features(img_path):
    """讀取圖片並使用 Canny 邊緣檢測提取輪廓特徵"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)
    resized = cv2.resize(edges, (64, 64))
    return resized.flatten() / 255.0

def load_dataset(folder_path):
    images = []
    labels = []
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
                features = extract_features(img_path)
                if features is not None:
                    images.append(features)
                    labels.append(label_idx)
    return np.array(images), np.array(labels)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'dataset', 'train')
    test_dir = os.path.join(base_dir, 'dataset', 'test')
    demo_dir = os.path.join(base_dir, 'demo')

    print("=== [MLP] 開始讀取訓練集圖片並進行特徵萃取 ===")
    X_train, y_train = load_dataset(train_dir)
    print("=== [MLP] 開始讀取測試集圖片 ===")
    X_test, y_test = load_dataset(test_dir)

    print(f"\n📊 訓練樣本數: {len(X_train)}, 測試樣本數: {len(X_test)}")

    if len(X_train) == 0:
        print("❌ 錯誤：找不到任何圖片！")
        return

    print("\n=== 開始訓練 MLP 神經網路模型 (可能需要幾分鐘) ===")
    # 建立多層感知機 (神經網路)，設定隱藏層與最高迭代次數
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    clf.fit(X_train, y_train)

    print("\n=== 評估 MLP 模型 ===")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 MLP 模型準確率: {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=['Rock', 'Paper', 'Scissors']))

    os.makedirs(demo_dir, exist_ok=True)
    model_path = os.path.join(demo_dir, 'rps_mlp_model.pkl')
    joblib.dump(clf, model_path)
    print(f"✅ 模型已成功儲存至: {model_path}")

if __name__ == "__main__":
    main()
