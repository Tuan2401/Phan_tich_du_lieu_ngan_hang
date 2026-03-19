import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo thêm đặc trưng mới để cải thiện mô hình.
    """
    df = data.copy()

    # Khách hàng từng được contact trước chưa
    if "pdays" in df.columns:
        df["contacted_before"] = df["pdays"].apply(lambda x: 0 if x == -1 else 1)

    # Phân nhóm mức balance
    if "balance" in df.columns:
        df["balance_level"] = pd.qcut(df["balance"], q=4, labels=False)

    logging.info("Đã tạo thêm các feature mới.")

def encode_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Mã hóa đặc trưng và Chuẩn hóa dữ liệu (Encoding & Scaling).
    Giải quyết triệt để biến phân loại và biến số học.
    """
    df = data.copy()
    
    # 1. Mã hóa riêng biệt biến mục tiêu 'y' (nếu có trong tập dữ liệu)
    if 'y' in df.columns:
        # Bank Marketing dataset có nhãn y là 'yes'/'no'
        df['y'] = df['y'].map({'yes': 1, 'no': 0})
        logging.info("Đã mã hóa biến mục tiêu 'y' (yes/no -> 1/0).")

    # 2. Phân loại các cột theo kiểu dữ liệu để xử lý riêng
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Loại bỏ biến mục tiêu 'y' khỏi danh sách tính toán để không làm hỏng nhãn
    if 'y' in categorical_cols: categorical_cols.remove('y')
    if 'y' in numerical_cols: numerical_cols.remove('y')

    # 3. XỬ LÝ BIẾN PHÂN LOẠI: Dùng One-Hot Encoding thay vì LabelEncoder
    if categorical_cols:
        # drop_first=True giúp tránh bẫy biến giả (Dummy Variable Trap)
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Chuyển đổi kiểu boolean (True/False) của get_dummies sang int (1/0)
        dummy_cols = [col for col in df.columns if col not in numerical_cols + ['y']]
        df[dummy_cols] = df[dummy_cols].astype(int)
        
        logging.info(f"Đã áp dụng One-Hot Encoding cho {len(categorical_cols)} biến phân loại.")

    # 4. XỬ LÝ BIẾN SỐ HỌC: Chuẩn hóa StandardScaler (Bắt buộc cho K-Means, Ridge)
    if numerical_cols:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        logging.info(f"Đã chuẩn hóa (StandardScaler) cho {len(numerical_cols)} biến số học.")

    logging.info(f"Kích thước dữ liệu sau khi mã hóa và chuẩn hóa: {df.shape}")
    return df
def build_features(df):

    df["contacted_before"] = df["pdays"].apply(lambda x: 0 if x == -1 else 1)

    df["is_married"] = (df["marital"] == "married").astype(int)

    df["balance_level"] = pd.qcut(df["balance"], q=4, labels=False)

    return df    