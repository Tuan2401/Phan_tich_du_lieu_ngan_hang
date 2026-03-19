🏦 Bank Transaction Analysis & Customer Prediction Syste

📌 Giới thiệu đề tài

Dự án này xây dựng một hệ thống phân tích dữ liệu giao dịch ngân hàng nhằm:

Dự đoán khách hàng có khả năng đăng ký sản phẩm (term deposit)

Phân tích hành vi khách hàng

Hỗ trợ chiến lược marketing ngân hàng

Hệ thống sử dụng các kỹ thuật Machine Learning và Data Mining như:

Classification (phân loại)

Association Rules (luật kết hợp)

Customer Segmentation (phân cụm)


🎯 Mục tiêu

Xây dựng pipeline xử lý dữ liệu hoàn chỉnh

So sánh nhiều mô hình ML

Tìm ra mô hình tối ưu

Phân tích insight khách hàng từ dữ liệu

📊 Mô tả dữ liệu

Nguồn: Bank Marketing Dataset

Số mẫu ban đầu: 45,211

Số thuộc tính ban đầu: 17

Sau xử lý: Loại bỏ duration (tránh data leakage)

Còn: 45,211 × 16

Dữ liệu bao gồm:

Thông tin cá nhân (age, job, marital…)

Thông tin tài chính (balance, loan…)

Thông tin chiến dịch marketing (contact, campaign…)

⚙️ Quy trình xử lý dữ liệu

1. Data Cleaning
   
Thay thế "unknown" → NaN

Xóa các dòng thiếu dữ liệu

Loại bỏ cột duration


2. Feature Engineering

One-Hot Encoding cho biến phân loại

Chuẩn hóa dữ liệu số (StandardScaler)

Mã hóa biến mục tiêu:
yes → 1,
no → 0

👉 Sau xử lý: Kích thước dữ liệu: 45,211 × 41


3. Data Splitting & Balancing
   
Train: 36,168

Test: 9,043

⚠️ Mất cân bằng:

Lớp 0: 31,937

Lớp 1: 4,231

✅ Sau SMOTE: 2 lớp đều: 31,937

🤖 Mô hình sử dụng

Logistic Regression

Random Forest 

XGBoost
