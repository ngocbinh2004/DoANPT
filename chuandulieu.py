import pandas as pd
import numpy as np

# 1. Đọc dữ liệu từ file Country.csv
df = pd.read_csv("Country.csv")

# 2. Kiểm tra số lượng giá trị thiếu ở mỗi cột
print("Số lượng giá trị thiếu (NaN) ở mỗi cột:")
print(df.isna().sum())

# 3. Xử lý dữ liệu

# 3a. Xóa các cột có quá nhiều giá trị thiếu (hơn 50% số dòng)
threshold = 0.5 * len(df)  # ngưỡng 50%
cols_to_drop = [col for col in df.columns if df[col].isna().sum() > threshold]
df.drop(columns=cols_to_drop, inplace=True)
print(f"\nĐã xóa {len(cols_to_drop)} cột có quá nhiều giá trị thiếu.")

# 3b. Điền giá trị thiếu cho cột số (numeric) bằng median
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# 3c. Điền giá trị thiếu cho cột dạng object (category/string) bằng mode
cat_cols = df.select_dtypes(include=[object]).columns
for col in cat_cols:
    if df[col].isna().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Kiểm tra lại sau khi xử lý
print("\nSố lượng giá trị thiếu sau khi xử lý:")
print(df.isna().sum())

# 4. Lưu dữ liệu đã làm sạch vào file Country_cleaned.csv
df.to_csv("Country_cleaned.csv", index=False)
print("\nĐã lưu dữ liệu sạch vào file Country_cleaned.csv")
