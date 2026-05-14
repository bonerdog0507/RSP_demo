"""
carema_landmark.py
使用 MediaPipe Tasks API + 訓練好的 Landmark 模型進行即時手勢辨識。
支援: Rock, Paper, Scissors, Error (無法辨識)
按 'q' 退出, 按 's' 截圖。
"""
import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
import os
import time

# 標籤名稱與顏色
LABELS = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}
LABEL_COLORS = {
    'Rock': (0, 0, 255),       # 紅色
    'Paper': (0, 255, 0),      # 綠色
    'Scissors': (255, 165, 0), # 橘色
    'Error': (128, 128, 128)   # 灰色
}


def normalize_landmarks(hand_landmarks):
    """將 hand landmarks 轉換為正規化特徵向量 (Tasks API format)"""
    landmark_list = []
    for lm in hand_landmarks:
        landmark_list.extend([lm.x, lm.y, lm.z])

    # Translation Normalization
    base_x, base_y, base_z = landmark_list[0], landmark_list[1], landmark_list[2]
    translated = []
    for i in range(0, len(landmark_list), 3):
        translated.append(landmark_list[i] - base_x)
        translated.append(landmark_list[i + 1] - base_y)
        translated.append(landmark_list[i + 2] - base_z)

    # Scale Normalization
    max_dist = 0
    for i in range(0, len(translated), 3):
        dist = np.sqrt(translated[i] ** 2 + translated[i + 1] ** 2 + translated[i + 2] ** 2)
        if dist > max_dist:
            max_dist = dist

    if max_dist == 0:
        max_dist = 1

    normalized = [x / max_dist for x in translated]
    return normalized


def draw_landmarks_on_image(image, detection_result):
    """在影像上繪製手部骨架"""
    if not detection_result.hand_landmarks:
        return image

    hand_landmarks_proto = solutions.hands.HandLandmark
    mp_drawing = solutions.drawing_utils
    mp_drawing_styles = solutions.drawing_styles
    hand_connections = solutions.hands.HAND_CONNECTIONS

    for hand_landmarks in detection_result.hand_landmarks:
        # 轉換為 NormalizedLandmarkList (proto format) 以便繪製
        from mediapipe.framework.formats import landmark_pb2
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        for lm in hand_landmarks:
            landmark = hand_landmarks_proto.landmark.add()
            landmark.x = lm.x
            landmark.y = lm.y
            landmark.z = lm.z

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    return image


def main():
    # 載入分類模型
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'best_landmark_model.pkl')
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        print("   Please run extract_landmarks.py -> train_advanced.py first!")
        return

    print("[INFO] Loading classifier model...")
    clf = joblib.load(model_path)
    print("[OK] Classifier model loaded!")

    # 初始化 MediaPipe Hand Landmarker (Tasks API)
    base_dir = os.path.dirname(script_dir)
    task_path = os.path.join(base_dir, 'hand_landmarker.task')
    if not os.path.exists(task_path):
        print(f"[ERROR] hand_landmarker.task not found at {task_path}")
        return

    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    print("[OK] MediaPipe Hand Landmarker initialized!")

    # 開啟攝影機
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera!")
        return

    print("\n[Camera] Started!")
    print("   Press 'q' to quit")
    print("   Press 's' to screenshot")

    screenshot_dir = os.path.join(script_dir, 'screenshots')
    os.makedirs(screenshot_dir, exist_ok=True)
    gesture_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 鏡像翻轉
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = landmarker.detect(mp_image)

        gesture = "Error"
        confidence = 0.0

        if detection_result.hand_landmarks:
            # 繪製手部骨架
            frame = draw_landmarks_on_image(frame, detection_result)

            # 提取並正規化特徵
            hand_landmarks = detection_result.hand_landmarks[0]
            features = normalize_landmarks(hand_landmarks)
            features_array = np.array(features).reshape(1, -1)

            # 預測
            prediction = clf.predict(features_array)[0]
            gesture = LABELS.get(prediction, "Error")

            # 如果模型支援 probability, 取得信心度
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(features_array)[0]
                confidence = max(proba)

        # 繪製結果
        color = LABEL_COLORS.get(gesture, (128, 128, 128))

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 文字
        cv2.putText(frame, f"Gesture: {gesture}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        if confidence > 0:
            cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("RSP Demo - MediaPipe Landmark", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            gesture_count += 1
            filename = os.path.join(screenshot_dir, f"gesture_{gesture_count}_{gesture}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[Screenshot] Saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Camera closed.")


if __name__ == "__main__":
    main()
