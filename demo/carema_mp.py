import cv2
import numpy as np
import joblib
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main():
    model_path = 'rps_mp_model.pkl'
    task_path = 'hand_landmarker.task'
    
    if not os.path.exists(model_path) or not os.path.exists(task_path):
        print(f"❌ 錯誤：找不到模型檔案或 task 檔。")
        print("請先執行 python train/train_mediapipe.py 訓練模型。")
        return

    print("⏳ 載入 MediaPipe 預測模型中...")
    clf = joblib.load(model_path)
    
    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    print("✅ 模型載入成功！")

    labels = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}
    cap = cv2.VideoCapture(0)

    print("🚀 您現在可以把手放在畫面的任何地方！")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        # MediaPipe 需要 RGB 格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # 進行手部偵測
        detection_result = detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]
            
            # 畫出 21 個節點
            h, w, c = frame.shape
            features = []
            for landmark in landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                features.extend([landmark.x, landmark.y, landmark.z])
            
            # 連線 (簡單畫幾個主要的線段)
            connections = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), 
                          (5,9), (9,10), (10,11), (11,12), (9,13), (13,14), (14,15), 
                          (15,16), (13,17), (0,17), (17,18), (18,19), (19,20)]
            for p1, p2 in connections:
                x1, y1 = int(landmarks[p1].x * w), int(landmarks[p1].y * h)
                x2, y2 = int(landmarks[p2].x * w), int(landmarks[p2].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
            features_arr = np.array([features])
            pred_idx = clf.predict(features_arr)[0]
            prediction = labels.get(pred_idx, "Unknown")
            
            text_x = int(landmarks[8].x * w)
            text_y = int(landmarks[8].y * h) - 20
            cv2.putText(frame, f"MediaPipe: {prediction}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("MediaPipe Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
