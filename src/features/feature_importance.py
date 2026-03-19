import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = "DejaVu Sans"


def plot_feature_importance(model, feature_names, output_dir="outputs/figures"):

    if not hasattr(model, "feature_importances_"):
        print("Mô hình không hỗ trợ Feature Importance")
        return

    os.makedirs(output_dir, exist_ok=True)

    importance = model.feature_importances_

    df = pd.DataFrame({
        "Đặc trưng": feature_names,
        "Mức độ quan trọng": importance
    })

    # ======================
    # DỊCH SANG TIẾNG VIỆT
    # ======================

    mapping = {
        "age": "Tuổi khách hàng",
        "balance": "Số dư tài khoản",
        "day": "Ngày liên hệ",
        "campaign": "Số lần liên hệ",
        "pdays": "Số ngày từ lần liên hệ trước",
        "previous": "Số lần liên hệ trước",
        "balance_level": "Mức số dư",
        "housing_yes": "Có vay mua nhà",
        "loan_yes": "Có vay cá nhân",
        "month_may": "Tháng 5"
    }

    df["Đặc trưng"] = df["Đặc trưng"].replace(mapping)

    df = df.sort_values("Mức độ quan trọng", ascending=False)

    # chỉ lấy top 10
    df = df.head(10)

    # ======================
    # VẼ BIỂU ĐỒ
    # ======================

    plt.figure(figsize=(10,6))

    plt.barh(df["Đặc trưng"], df["Mức độ quan trọng"])

    plt.gca().invert_yaxis()

    plt.title(
        "Top 10 đặc trưng ảnh hưởng mạnh nhất đến dự đoán",
        fontsize=14,
        pad=15
    )

    plt.xlabel("Mức độ quan trọng", fontsize=12)

    plt.ylabel("Đặc trưng", fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # căn lề cho chữ dài
    plt.subplots_adjust(left=0.35)

    # tự cân layout
    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Đã lưu biểu đồ feature importance.")