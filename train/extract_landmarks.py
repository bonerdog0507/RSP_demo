"""
extract_landmarks.py
使用 MediaPipe Tasks API (Hand Landmarker) 從圖片中提取 21 個手部關鍵點座標,
並進行平移正規化 (Translation Normalization) 和
比例正規化 (Scale Normalization) 以提升分類效果。
"""
import os
import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 初始化 Hand Landmarker (Tasks API)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'hand_landmarker.task')

if not os.path.exists(model_path):
    print(f"[ERROR] hand_landmarker.task not found at {model_path}")
    print("   Please download it first.")
    exit(1)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)


def extract_landmarks(image):
    """從圖片中提取並正規化手部關鍵點座標"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = landmarker.detect(mp_image)

    if not detection_result.hand_landmarks:
        return None

    hand_landmarks = detection_result.hand_landmarks[0]
    landmark_list = []
    for lm in hand_landmarks:
        landmark_list.extend([lm.x, lm.y, lm.z])

    # === 1. Translation Normalization (平移正規化) ===
    # 以手腕 (landmark 0) 為原點，消除手在畫面中位置的影響
    base_x, base_y, base_z = landmark_list[0], landmark_list[1], landmark_list[2]
    translated = []
    for i in range(0, len(landmark_list), 3):
        translated.append(landmark_list[i] - base_x)
        translated.append(landmark_list[i + 1] - base_y)
        translated.append(landmark_list[i + 2] - base_z)

    # === 2. Scale Normalization (比例正規化) ===
    # 除以所有關鍵點到原點的最大距離，消除手大小/距離的影響
    max_dist = 0
    for i in range(0, len(translated), 3):
        dist = np.sqrt(translated[i] ** 2 + translated[i + 1] ** 2 + translated[i + 2] ** 2)
        if dist > max_dist:
            max_dist = dist

    if max_dist == 0:
        max_dist = 1  # 避免除以零

    normalized = [x / max_dist for x in translated]

    return normalized


def process_dataset(folder_path):
    """處理整個資料集資料夾，回傳特徵矩陣與標籤"""
    data = []
    labels = []
    label_map = {'rock': 0, 'paper': 1, 'scissors': 2}

    for category, label_idx in label_map.items():
        category_path = os.path.join(folder_path, category)

        # 處理解壓後可能多包一層資料夾的情況
        if not os.path.exists(category_path):
            subdirs = [d for d in os.listdir(folder_path)
                       if os.path.isdir(os.path.join(folder_path, d))]
            if subdirs:
                category_path = os.path.join(folder_path, subdirs[0], category)

        if not os.path.exists(category_path):
            print(f"[WARN] Folder not found for '{category}' -> {category_path}")
            continue

        print(f"[INFO] Processing {category}...")
        count = 0
        skipped = 0
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(category_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    lms = extract_landmarks(img)
                    if lms:
                        data.append(lms)
                        labels.append(label_idx)
                        count += 1
                    else:
                        skipped += 1
        print(f"   -> Extracted {count} samples, Skipped {skipped} (no hand detected)")

    return np.array(data), np.array(labels)


def main():
    train_dir = os.path.join(base_dir, 'dataset', 'train')
    test_dir = os.path.join(base_dir, 'dataset', 'test')

    print("=" * 50)
    print("  MediaPipe Hand Landmark Extraction")
    print("=" * 50)

    print("\n=== Step 1: Extracting Training Landmarks ===")
    X_train, y_train = process_dataset(train_dir)

    print("\n=== Step 2: Extracting Testing Landmarks ===")
    X_test, y_test = process_dataset(test_dir)

    print(f"\n[Summary]")
    print(f"   Training samples : {len(X_train)}")
    print(f"   Testing  samples : {len(X_test)}")
    if len(X_train) > 0:
        print(f"   Feature dimension: {X_train.shape[1]} (21 landmarks × 3 coords)")

    if len(X_train) == 0:
        print("[ERROR] No landmarks extracted. Check your dataset folder structure!")
        return

    # 儲存提取的 landmark 資料
    save_path = os.path.join(base_dir, 'train', 'landmark_data.joblib')
    joblib.dump({
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test
    }, save_path)
    print(f"\n[DONE] All landmark data saved to: {save_path}")
    print("   -> Next step: Run train_advanced.py to train models.")


if __name__ == "__main__":
    main()
