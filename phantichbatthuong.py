# phantichbatthuong.py
# “Nhóm em phát hiện các điểm bất thường trong dữ liệu như dân số âm hoặc thương mại quá cao. 
# Dùng z-score và boxplot để lọc ra.”
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

df = pd.read_csv("Country_cleaned.csv")
num_cols = df.select_dtypes(include='number').columns
z_scores = df[num_cols].apply(zscore)

# Phân tích tương quan
corr = df[num_cols].corr()
print("\nMa trận tương quan:")
print(corr)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Ma trận tương quan giữa các biến số")
plt.tight_layout()
plt.savefig("correlation_matrix.png")

# Boxplot, Violin plot, Z-Score
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot - {col}")
    plt.savefig(f"boxplot_{col}.png")

    plt.figure(figsize=(6, 4))
    sns.violinplot(x=df[col])
    plt.title(f"Violin plot - {col}")
    plt.savefig(f"violinplot_{col}.png")

    plt.figure(figsize=(6, 4))
    plt.plot(z_scores[col])
    plt.axhline(y=3, color='r', linestyle='--')
    plt.title(f"Z-Score Outlier - {col}")
    plt.savefig(f"zscore_{col}.png")
