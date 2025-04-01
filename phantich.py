import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Đọc dữ liệu từ file input (Country_cleaned
df = pd.read_csv("Country_cleaned.csv")
print("Dữ liệu input từ file Country_cleaned.csv:")
print(df.head())

# -------------------------------
# Phân tích 1: Số lượng quốc gia theo Vùng (Region)
# -------------------------------

# Đếm số quốc gia theo từng vùng
region_counts = df['Region'].value_counts()
print("\nSố lượng quốc gia theo vùng:")
print(region_counts)

# Vẽ biểu đồ cột (bar chart)
plt.figure(figsize=(10,6))
sns.barplot(x=region_counts.index, y=region_counts.values, palette="viridis")
plt.title("Số lượng quốc gia theo vùng")
plt.xlabel("Vùng")
plt.ylabel("Số lượng quốc gia")
plt.xticks(rotation=45)
plt.tight_layout()

# Lưu biểu đồ dưới dạng file PNG
plt.savefig("Country_by_Region.png")
plt.show()

# -------------------------------
# Phân tích 2: Số lượng quốc gia theo Nhóm thu nhập (IncomeGroup)
# -------------------------------

# Đếm số quốc gia theo nhóm thu nhập
income_counts = df['IncomeGroup'].value_counts()
print("\nSố lượng quốc gia theo nhóm thu nhập:")
print(income_counts)

# Vẽ biểu đồ cột (bar chart)
plt.figure(figsize=(10,6))
sns.barplot(x=income_counts.index, y=income_counts.values, palette="magma")
plt.title("Số lượng quốc gia theo nhóm thu nhập")
plt.xlabel("Nhóm thu nhập")
plt.ylabel("Số lượng quốc gia")
plt.xticks(rotation=45)
plt.tight_layout()

# Lưu biểu đồ dưới dạng file PNG
plt.savefig("Country_by_IncomeGroup.png")
plt.show()

