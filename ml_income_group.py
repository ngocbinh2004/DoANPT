import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

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
print("Độ chính xác (Accuracy):", accuracy_score(y_test, y_du_doan))
print("Báo cáo phân loại:")
print(classification_report(y_test, y_du_doan, target_names=encoder.classes_))
