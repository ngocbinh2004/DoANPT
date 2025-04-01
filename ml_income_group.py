
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

# Đọc dữ liệu từ file CSV
du_lieu = pd.read_csv('Country_cleaned.csv')

# Lấy các cột cần dùng: vùng, đơn vị tiền tệ, hệ thống thương mại và nhóm thu nhập
du_lieu_loc = du_lieu[['Region', 'CurrencyUnit', 'SystemOfTrade', 'IncomeGroup']].dropna()

# Mã hóa các cột đầu vào (biến phân loại thành biến số)
X = pd.get_dummies(du_lieu_loc[['Region', 'CurrencyUnit', 'SystemOfTrade']])

# Mã hóa cột nhãn (IncomeGroup) thành số
encoder = LabelEncoder()
y = encoder.fit_transform(du_lieu_loc['IncomeGroup'])

# Chia dữ liệu thành tập huấn luyện và kiểm tra (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình Random Forest
mo_hinh = RandomForestClassifier(random_state=42)
mo_hinh.fit(X_train, y_train)

# Dự đoán trên dữ liệu kiểm tra
y_du_doan = mo_hinh.predict(X_test)

# In ra độ chính xác và báo cáo phân loại
print("🎯 Độ chính xác (Accuracy):", accuracy_score(y_test, y_du_doan))
print("📊 Báo cáo phân loại:")
print(classification_report(y_test, y_du_doan, target_names=encoder.classes_))

# === 1. Lưu mô hình và encoder vào file ===
joblib.dump(mo_hinh, 'random_forest_income_model.pkl')
joblib.dump(encoder, 'label_encoder.pkl')
print("✅ Đã lưu mô hình và encoder vào file.")

# === 2. Tải lại mô hình và encoder từ file ===
mo_hinh_tai_lai = joblib.load('random_forest_income_model.pkl')
encoder_tai_lai = joblib.load('label_encoder.pkl')

# === 3. Dự đoán với một mẫu mới (giá trị giả định) ===
# Tạo mẫu mới giả sử có cùng cấu trúc như X
mau_moi = pd.DataFrame([{
    'Region_East Asia & Pacific': 1,
    'CurrencyUnit_U.S. dollars': 1,
    'SystemOfTrade_Special': 1
}], columns=X.columns).fillna(0)

du_doan_moi = mo_hinh_tai_lai.predict(mau_moi)
nhom_thu_nhap = encoder_tai_lai.inverse_transform(du_doan_moi)

print("🌍 Nhóm thu nhập dự đoán cho mẫu mới là:", nhom_thu_nhap[0])
