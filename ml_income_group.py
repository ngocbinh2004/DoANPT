import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# 1. Đọc dữ liệu
df = pd.read_csv('Country_cleaned.csv')
df = df[['Region', 'CurrencyUnit', 'SystemOfTrade', 'IncomeGroup']].dropna()

# 2. Phân phối nhãn
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='IncomeGroup')
plt.title('Phân phối nhóm thu nhập')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Tiền xử lý
X = pd.get_dummies(df[['Region', 'CurrencyUnit', 'SystemOfTrade']])
encoder = LabelEncoder()
y = encoder.fit_transform(df['IncomeGroup'])

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Huấn luyện mô hình Random Forest đơn giản hơn (giảm overfit)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# 6. Đánh giá mô hình
y_pred = model.predict(X_test)

print("🎯 Độ chính xác (Accuracy):", round(accuracy_score(y_test, y_pred), 4))
print("📊 Báo cáo phân loại:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# 7. Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title("Ma trận nhầm lẫn")
plt.tight_layout()
plt.show()

# 8. Feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 5))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Độ quan trọng của đặc trưng (Feature Importance)")
plt.tight_layout()
plt.show()

# 9. Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("📉 Cross-validation accuracy trung bình:", round(np.mean(cv_scores), 4))

# 10. Lưu mô hình
joblib.dump(model, 'rf_income_model_fixed.pkl')
joblib.dump(encoder, 'label_encoder_fixed.pkl')

# 11. Dự đoán mẫu mới
sample = pd.DataFrame([{
    'Region_East Asia & Pacific': 1,
    'CurrencyUnit_U.S. dollars': 1,
    'SystemOfTrade_Special': 1
}], columns=X.columns).fillna(0)

model_loaded = joblib.load('rf_income_model_fixed.pkl')
encoder_loaded = joblib.load('label_encoder_fixed.pkl')
prediction = model_loaded.predict(sample)
print("🌍 Nhóm thu nhập dự đoán cho mẫu mới là:", encoder_loaded.inverse_transform(prediction)[0])
