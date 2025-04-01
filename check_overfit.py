import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu
df = pd.read_csv('Country_cleaned.csv')
df_ml = df[['Region', 'CurrencyUnit', 'SystemOfTrade', 'IncomeGroup']].dropna()

# Mã hóa dữ liệu
X = pd.get_dummies(df_ml[['Region', 'CurrencyUnit', 'SystemOfTrade']])
le = LabelEncoder()
y = le.fit_transform(df_ml['IncomeGroup'])

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mô hình
model = RandomForestClassifier()
model.fit(X_train, y_train)

# So sánh train/test accuracy
print("Độ chính xác trên tập train:", model.score(X_train, y_train))
print("Độ chính xác trên tập test :", model.score(X_test, y_test))
