"""
train_advanced.py
使用 MediaPipe 關鍵點資料訓練兩個模型:
  Model A: SVM (RBF Kernel)
  Model B: Random Forest
比較兩者的 Accuracy, Precision, Recall, F1-Score，
並自動選擇最佳模型儲存至 demo 資料夾。
"""
import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


def evaluate_model(name, model, X_test, y_test):
    """評估模型效能並印出詳細報告"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n{'=' * 45}")
    print(f"  📊 {name} Results")
    print(f"{'=' * 45}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    labels = ['Rock', 'Paper', 'Scissors']
    print(f"  {'':>10s} {'Pred_R':>8s} {'Pred_P':>8s} {'Pred_S':>8s}")
    for i, row in enumerate(cm):
        print(f"  {labels[i]:>10s} {row[0]:>8d} {row[1]:>8d} {row[2]:>8d}")
    print(f"\n  Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    return {'name': name, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'model': model}


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'train', 'landmark_data.joblib')

    if not os.path.exists(data_path):
        print(f"❌ Error: Landmark data not found at {data_path}")
        print("   Please run extract_landmarks.py first!")
        return

    print("=" * 50)
    print("  Advanced Model Training (Landmark-based)")
    print("=" * 50)

    # === Load Data ===
    print("\n⏳ Loading landmark data...")
    data = joblib.load(data_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    print(f"   Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"   Features per sample: {X_train.shape[1]}")

    # === Model A: SVM (RBF Kernel) ===
    print("\n🔧 Training Model A: SVM (RBF Kernel, C=10.0)...")
    svm_model = SVC(kernel='rbf', probability=True, C=10.0, gamma='scale')
    svm_model.fit(X_train, y_train)
    results_svm = evaluate_model("SVM (RBF)", svm_model, X_test, y_test)

    # === Model B: Random Forest ===
    print("\n🔧 Training Model B: Random Forest (n=200, depth=15)...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1  # 使用所有 CPU 核心
    )
    rf_model.fit(X_train, y_train)
    results_rf = evaluate_model("Random Forest", rf_model, X_test, y_test)

    # === Comparison Summary ===
    print("\n" + "=" * 50)
    print("  🏆 Model Comparison Summary")
    print("=" * 50)
    print(f"  {'Metric':<12s} {'SVM':>10s} {'RF':>10s} {'Winner':>10s}")
    print(f"  {'-' * 42}")

    metrics = [('Accuracy', 'acc'), ('Precision', 'prec'), ('Recall', 'rec'), ('F1-Score', 'f1')]
    for label, key in metrics:
        svm_val = results_svm[key]
        rf_val = results_rf[key]
        winner = "SVM" if svm_val >= rf_val else "RF"
        print(f"  {label:<12s} {svm_val:>10.4f} {rf_val:>10.4f} {winner:>10s}")

    # === Save Best Model ===
    best = results_svm if results_svm['acc'] >= results_rf['acc'] else results_rf
    print(f"\n  🥇 Best Model: {best['name']} (Accuracy: {best['acc']:.4f})")

    demo_dir = os.path.join(base_dir, 'demo')
    os.makedirs(demo_dir, exist_ok=True)

    model_save_path = os.path.join(demo_dir, 'best_landmark_model.pkl')
    joblib.dump(best['model'], model_save_path)
    print(f"\n✅ Best model saved to: {model_save_path}")
    print("   → Use this model with carema_landmark.py for real-time demo.")


if __name__ == "__main__":
    main()
