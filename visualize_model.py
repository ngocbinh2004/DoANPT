import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd

# === 1. Tải mô hình và encoder đã lưu ===
mo_hinh = joblib.load('random_forest_income_model.pkl')
encoder = joblib.load('label_encoder.pkl')

# === 2. Tải lại dữ liệu để lấy tên các đặc trưng ===
du_lieu = pd.read_csv('Country_cleaned.csv')
du_lieu_loc = du_lieu[['Region', 'CurrencyUnit', 'SystemOfTrade', 'IncomeGroup']].dropna()
X = pd.get_dummies(du_lieu_loc[['Region', 'CurrencyUnit', 'SystemOfTrade']])
feature_names = X.columns

# === 3. Vẽ biểu đồ độ quan trọng của đặc trưng ===
feature_importances = mo_hinh.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel("Feature Importance")
plt.title("📊 Độ quan trọng của các đặc trưng (Random Forest)")
plt.tight_layout()
plt.show()

# === 4. Vẽ 1 cây quyết định trong Random Forest ===
plt.figure(figsize=(20, 10))
plot_tree(mo_hinh.estimators_[0], 
          feature_names=feature_names,
          class_names=encoder.classes_,
          filled=True, max_depth=3)
plt.title("🌳 Một cây quyết định trong Random Forest (giới hạn depth=3)")
plt.show()
