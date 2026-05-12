import os
import cv2
import numpy as np
import joblib
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'dataset', 'train')
    test_dir = os.path.join(base_dir, 'dataset', 'test')
    demo_dir = os.path.join(base_dir, 'demo')
    
    # 確保有模型檔
    os.makedirs(demo_dir, exist_ok=True)
    task_path = os.path.join(demo_dir, 'hand_landmarker.task')
    if not os.path.exists(task_path):
        print("📥 正在下載 MediaPipe 骨架模型檔...")
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        urllib.request.urlretrieve(url, task_path)

    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    def extract_landmarks(img_path):
        img = cv2.imread(img_path)
        if img is None: return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        detection_result = detector.detect(mp_image)
        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]
            features = []
            for landmark in landmarks:
                features.extend([landmark.x, landmark.y, landmark.z])
            return np.array(features)
        return None

    def load_dataset(folder_path):
        images, labels = [], []
        label_map = {'rock': 0, 'paper': 1, 'scissors': 2}
        for category, label_idx in label_map.items():
            category_path = os.path.join(folder_path, category)
            if not os.path.exists(category_path):
                subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
                if subdirs: category_path = os.path.join(folder_path, subdirs[0], category)
            if not os.path.exists(category_path): continue
                
            print(f"📂 正在使用 MediaPipe 掃描 {category} 的圖片...")
            for filename in os.listdir(category_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(category_path, filename)
                    features = extract_landmarks(img_path)
                    if features is not None:
                        images.append(features)
                        labels.append(label_idx)
        return np.array(images), np.array(labels)

    print("=== [MediaPipe] 開始抓取訓練集骨架特徵 ===")
    X_train, y_train = load_dataset(train_dir)
    print("=== [MediaPipe] 開始抓取測試集骨架特徵 ===")
    X_test, y_test = load_dataset(test_dir)

    print(f"\n📊 成功抓取骨架樣本數 - 訓練集: {len(X_train)}, 測試集: {len(X_test)}")
    if len(X_train) == 0:
        print("❌ 錯誤：無法在圖片中偵測到任何手部骨架！")
        return

    print("\n=== 開始訓練 SVM (基於骨架 63 維特徵) ===")
    clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    clf.fit(X_train, y_train)

    print("\n=== 評估 MediaPipe 模型 ===")
    if len(X_test) > 0:
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"🎯 MediaPipe 模型準確率: {accuracy * 100:.2f}%\n")
        print(classification_report(y_test, y_pred, target_names=['Rock', 'Paper', 'Scissors']))
    else:
        print("❌ 測試集中沒有偵測到任何手勢！")

    model_path = os.path.join(demo_dir, 'rps_mp_model.pkl')
    joblib.dump(clf, model_path)
    print(f"✅ MediaPipe 模型已成功儲存至: {model_path}")

if __name__ == "__main__":
    main()
