import pandas as pd
import numpy as np
import logging


def load_data(path: str) -> pd.DataFrame:
    """Đọc dữ liệu từ file CSV."""

    try:
        # Dataset Bank Marketing dùng dấu ;
        data = pd.read_csv(path, sep=';')

        logging.info(f"Đã load dữ liệu thành công. Kích thước ban đầu: {data.shape}")

        return data

    except FileNotFoundError:
        logging.error(f"Không tìm thấy file tại đường dẫn: {path}")
        raise

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch dữ liệu: xử lý missing, unknown, duplicates và chống data leakage."""

    df = data.copy()

    # 1. Thống kê ban đầu
    logging.info(f"Số lượng dòng trùng lặp ban đầu: {df.duplicated().sum()}")

    # 2. Chuyển 'unknown' -> NaN
    df.replace("unknown", np.nan, inplace=True)

    missing_count = df.isnull().sum().sum()
    logging.info(f"Tổng số giá trị missing/unknown: {missing_count}")

    # 3. Xử lý missing values
    for col in df.columns:

        if df[col].isnull().sum() > 0:

            # kiểm tra numeric
            if pd.api.types.is_numeric_dtype(df[col]):

                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

            else:

                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)

    # 4. Xóa duplicate
    df = df.drop_duplicates()

    # 5. Chống Data Leakage (cột duration)
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
        logging.info("Đã xóa cột 'duration' để tránh data leakage")

    # 6. Thống kê sau xử lý
    logging.info("Hoàn tất bước làm sạch dữ liệu")
    logging.info(f"Kích thước dữ liệu sau khi clean: {df.shape}")

    return df