import sys
import os
import yaml
import logging
import shutil

# ==============================
# 1. THIẾT LẬP LOGGING
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline_execution.log", mode="w", encoding="utf-8")
    ]
)

# ==============================
# 2. XỬ LÝ ĐƯỜNG DẪN PROJECT
# ==============================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# ==============================
# 3. IMPORT MODULE
# ==============================
from src.data.data_cleaner import load_data, clean_data
from src.features.feature_builder import build_features, encode_features
from src.models.trainer import train_model
from src.evaluation.evaluator import evaluate, save_model

# NEW MODULES
from src.mining.segmentation import customer_segmentation
from src.mining.association_rules import find_cross_sell_rules
from src.features.feature_importance import plot_feature_importance


def main():

    logging.info("=" * 50)
    logging.info("🚀 KHỞI ĐỘNG PIPELINE KHAI PHÁ DỮ LIỆU")
    logging.info("=" * 50)

    # ==============================
    # ⭐ RESET THƯ MỤC BIỂU ĐỒ
    # ==============================
    figures_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")

    try:
        if os.path.exists(figures_dir):
            shutil.rmtree(figures_dir)

        os.makedirs(figures_dir, exist_ok=True)

        logging.info("Đã reset thư mục outputs/figures")

    except Exception as e:
        logging.warning(f"Không thể reset thư mục figures: {e}")

    # ==============================
    # 4. LOAD CONFIG
    # ==============================
    config_path = os.path.join(PROJECT_ROOT, "configs", "params.yaml")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        logging.info("Đã tải file cấu hình params.yaml")

    except FileNotFoundError:
        logging.error("Không tìm thấy file configs/params.yaml")
        sys.exit(1)

    # ==============================
    # 5. LOAD DATA
    # ==============================
    data_path = os.path.join(PROJECT_ROOT, config["paths"]["raw_data"])

    try:
        logging.info(f"Đang load dữ liệu: {data_path}")
        data = load_data(data_path)
        logging.info(f"Kích thước dữ liệu gốc: {data.shape}")

    except Exception as e:
        logging.error(f"Lỗi khi load data: {e}")
        sys.exit(1)

    # ==============================
    # 6. CLEAN DATA
    # ==============================
    logging.info("Đang làm sạch dữ liệu...")
    data_cleaned = clean_data(data)

    # ==============================
    # ⭐ 7. CUSTOMER SEGMENTATION
    # ==============================
    logging.info("Đang phân cụm khách hàng (Customer Segmentation)...")
    try:
        data_segmented = customer_segmentation(data_cleaned)
    except Exception as e:
        logging.warning(f"Segmentation lỗi: {e}")
        data_segmented = data_cleaned

    # ==============================
    # ⭐ 8. ASSOCIATION RULES
    # ==============================
    logging.info("Đang tìm luật cross-sell...")
    try:
        rules = find_cross_sell_rules(data_segmented)
        logging.info("Top cross-sell rules:")
        logging.info(rules.head())
    except Exception as e:
        logging.warning(f"Association rules lỗi: {e}")

    # ==============================
    # 9. FEATURE ENGINEERING
    # ==============================
    logging.info("Đang tạo đặc trưng mới...")
    data_features = build_features(data_segmented)

    # ==============================
    # 10. ENCODE FEATURES
    # ==============================
    logging.info("Đang mã hóa categorical features...")
    data_encoded = encode_features(data_features)

    # ==============================
    # 11. TRAIN MODELS
    # ==============================
    logging.info("Đang train models...")
    models, X_test, y_test = train_model(data_encoded)

    # ==============================
    # 12. EVALUATE MODELS
    # ==============================
    best_model = None
    best_score = 0
    best_model_name = None

    for name, model in models.items():

        logging.info(f"===== Evaluating {name} =====")

        metrics = evaluate(model, X_test, y_test)

        logging.info(f"Metrics: {metrics}")

        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_model = model
            best_model_name = name

    logging.info(f"Best model: {best_model_name}")
    logging.info(f"Best F1 Score: {best_score}")

    # ==============================
    # ⭐ 13. FEATURE IMPORTANCE
    # ==============================
    try:
        logging.info("Đang phân tích Feature Importance...")
        plot_feature_importance(best_model, X_test.columns)
    except Exception as e:
        logging.warning(f"Feature importance lỗi: {e}")

    # ==============================
    # 14. SAVE BEST MODEL
    # ==============================
    model_output = os.path.join(
        PROJECT_ROOT,
        config["paths"]["models"],
        "best_model.pkl"
    )

    os.makedirs(os.path.dirname(model_output), exist_ok=True)

    save_model(best_model, model_output)

    logging.info(f"Đã lưu best model tại: {model_output}")

    logging.info("=" * 50)
    logging.info("✅ PIPELINE HOÀN TẤT")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()