import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    auc
)

plt.rcParams["font.family"] = "DejaVu Sans"


def evaluate(model, X_test, y_test, model_name="model"):

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    # =============================
    # TÍNH METRICS
    # =============================

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print("\n📊 Kết quả đánh giá:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # =============================
    # CONFUSION MATRIX
    # =============================

    cm = confusion_matrix(y_test, y_pred)

    os.makedirs("outputs/figures", exist_ok=True)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"Ma trận nhầm lẫn - {model_name}")

    plt.ylabel("Giá trị thực tế")

    plt.xlabel("Giá trị dự đoán")

    plt.savefig(f"outputs/figures/confusion_matrix_{model_name}.png", dpi=300)

    plt.close()

    # =============================
    # PHÂN TÍCH LỖI
    # =============================

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds"
    )

    plt.title("Phân tích lỗi mô hình")

    plt.ylabel("Thực tế")

    plt.xlabel("Dự đoán")

    plt.savefig("outputs/figures/confusion_matrix_analysis.png", dpi=300)

    plt.close()

    # =============================
    # PRECISION RECALL CURVE
    # =============================

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)

    pr_auc = auc(recall_vals, precision_vals)

    plt.figure(figsize=(6,5))

    plt.plot(recall_vals, precision_vals)

    plt.xlabel("Recall (Độ bao phủ)")

    plt.ylabel("Precision (Độ chính xác)")

    plt.title(f"Đường cong Precision-Recall - {model_name}")

    plt.legend([f"PR curve (diện tích = {pr_auc:.4f})"])

    plt.savefig(f"outputs/figures/pr_curve_{model_name}.png", dpi=300)

    plt.close()

    # =============================
    # PR AUC CHART
    # =============================

    plt.figure(figsize=(6,5))

    plt.plot(recall_vals, precision_vals, color="orange")

    plt.xlabel("Recall (Độ bao phủ)")

    plt.ylabel("Precision (Độ chính xác)")

    plt.title("Đường cong Precision-Recall (PR-AUC)")

    plt.legend([f"PR curve (diện tích = {pr_auc:.4f})"])

    plt.savefig("outputs/figures/pr_curve_auc.png", dpi=300)

    plt.close()

    return metrics


def save_model(model, path):

    with open(path, "wb") as f:
        pickle.dump(model, f)

    print(f"\n💾 Đã lưu mô hình tại: {path}")