import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Đọc dữ liệu
df = pd.read_csv('Country_cleaned.csv')
df_ml = df[['Region', 'CurrencyUnit', 'SystemOfTrade', 'IncomeGroup']].dropna()

# Mã hóa dữ liệu đầu vào
X = pd.get_dummies(df_ml[['Region', 'CurrencyUnit', 'SystemOfTrade']])
y = df_ml['IncomeGroup']

# Loại bỏ outlier bằng Z-score
z_scores = np.abs(stats.zscore(X))
mask = (z_scores < 3).all(axis=1)
X_clean = X[mask]
y_clean = y[mask]

# Chia tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = model.predict(X_test)
print("Độ chính xác sau khi loại outlier:", accuracy_score(y_test, y_pred))
