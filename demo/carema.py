import cv2
import numpy as np
import joblib
import os

def main():
    model_path = 'rps_svm_model.pkl'
    
    # 檢查模型是否存在
    if not os.path.exists(model_path):
        print(f"❌ 錯誤：找不到模型檔案 '{model_path}'")
        print("請確認是否已經在上一層執行過訓練程式，並將模型放置於 demo 資料夾中。")
        return

    print("⏳ 載入模型中...")
    clf = joblib.load(model_path)
    print("✅ 模型載入成功！")

    # 標籤對應 (與訓練時相同)
    labels = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}

    # 開啟攝影機
    cap = cv2.VideoCapture(0)

    # 設定感興趣區域 (ROI) 的座標 [y_top:y_bottom, x_right:x_left]
    # 我們將在畫面上畫一個框，請將手放在這個框內
    roi_top, roi_bottom, roi_right, roi_left = 100, 350, 150, 400

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取攝影機畫面。")
            break
            
        # 將畫面水平翻轉 (這樣會像照鏡子一樣，比較直覺)
        frame = cv2.flip(frame, 1)

        # 畫出 ROI 綠色矩形框
        cv2.rectangle(frame, (roi_right, roi_top), (roi_left, roi_bottom), (0, 255, 0), 2)

        # 擷取 ROI 區域的影像
        roi = frame[roi_top:roi_bottom, roi_right:roi_left]

        try:
            # 影像前處理 (必須與訓練時一模一樣)
            # 1. 轉灰階
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # 2. 高斯模糊與 Canny 邊緣檢測
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 150)
            # 3. 縮放至 64x64
            resized = cv2.resize(edges, (64, 64))
            
            # 【SVM 眼中的畫面】
            debug_view = cv2.resize(resized, (250, 250))
            cv2.imshow("SVM Vision (Edges)", debug_view)

            # 4. 攤平
            flattened = resized.flatten()
            
            # 正規化
            features = np.array([flattened]) / 255.0
            
            # 模型預測
            pred_idx = clf.predict(features)[0]
            prediction = labels.get(pred_idx, "Unknown")
            
            # 將預測結果顯示在畫面上
            cv2.putText(frame, prediction, (roi_right, roi_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            # 防止預處理過程中出錯導致程式崩潰
            pass

        # 顯示畫面
        cv2.imshow("Camera", frame)

        # 按 'q' 鍵離開
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()