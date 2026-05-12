import cv2
import numpy as np
import os

def main():
    model_path = 'rps_cnn.onnx'
    
    if not os.path.exists(model_path):
        print(f"❌ 錯誤：找不到模型檔案 '{model_path}'")
        print("請先執行 python train/train_cnn.py 訓練並產生 ONNX 模型。")
        return

    print("⏳ 載入 CNN (ONNX) 模型中...")
    # 使用 OpenCV 內建的 DNN 模組讀取 ONNX，免裝 PyTorch！
    net = cv2.dnn.readNetFromONNX(model_path)
    print("✅ 模型載入成功！")

    labels = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}
    cap = cv2.VideoCapture(0)

    # ROI 座標
    roi_top, roi_bottom, roi_right, roi_left = 100, 350, 150, 400

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (roi_right, roi_top), (roi_left, roi_bottom), (0, 255, 0), 2)
        roi = frame[roi_top:roi_bottom, roi_right:roi_left]

        try:
            # === Canny 邊緣檢測前處理 ===
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 150)
            resized = cv2.resize(edges, (64, 64))
            
            # 【觀測視窗】 
            debug_view = cv2.resize(resized, (250, 250))
            cv2.imshow("Edge Vision (Debug)", debug_view)

            # 將圖片轉為 OpenCV DNN 所需的 blob 格式
            # 格式要求: (1, 1, 64, 64)，且正規化除以 255
            blob = cv2.dnn.blobFromImage(resized, scalefactor=1/255.0, size=(64, 64), mean=0, swapRB=False, crop=False)
            
            # 設定輸入並取得預測結果
            net.setInput(blob)
            outputs = net.forward()
            
            # 取得最大值的索引作為預測結果
            pred_idx = np.argmax(outputs[0])
            prediction = labels.get(pred_idx, "Unknown")
            
            cv2.putText(frame, f"CNN: {prediction}", (roi_right, roi_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            pass

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
