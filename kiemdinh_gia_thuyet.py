# kiemdinh_gia_thuyet.py
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, normaltest
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Country_cleaned.csv")

# Chọn 2 nhóm để kiểm định: High income vs Low income
high_income = df[df['IncomeGroup'].str.contains("High", na=False)]
low_income = df[df['IncomeGroup'].str.contains("Low", na=False)]

# Chọn một số chỉ số để kiểm định (nếu có)
numeric_cols = df.select_dtypes(include='number').columns

for col in numeric_cols:
    # Kiểm định phân phối chuẩn
    _, p_high = normaltest(high_income[col])
    _, p_low = normaltest(low_income[col])
    
    print(f"--- {col} ---")
    print(f"Normal test High Income (p={p_high:.3f}) | Low Income (p={p_low:.3f})")

    # Kiểm định giả thuyết: Trung bình 2 nhóm có khác biệt không
    stat, p_val = ttest_ind(high_income[col], low_income[col], nan_policy='omit')
    print(f"T-test p-value = {p_val:.4f}")
    if p_val < 0.05:
        print("→ Có sự khác biệt có ý nghĩa thống kê giữa 2 nhóm.")
    else:
        print("→ Không có sự khác biệt rõ ràng.")
    print()

# Vẽ Line Chart một vài cột (nếu có dữ liệu dạng năm)
# Ở đây ví dụ giả lập một line chart cho chỉ số ngẫu nhiên
sample_col = numeric_cols[0] if len(numeric_cols) > 0 else None
if sample_col:
    df_sorted = df.sort_values(by=sample_col)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_sorted, x=range(len(df_sorted)), y=sample_col)
    plt.title(f"Line Chart - {sample_col}")
    plt.xlabel("Quốc gia (sắp theo giá trị)")
    plt.ylabel(sample_col)
    plt.tight_layout()
    plt.savefig(f"linechart_{sample_col}.png")
