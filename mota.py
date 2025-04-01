# mota.py
import pandas as pd
import numpy as np

# Đọc dữ liệu sau khi làm sạch
df = pd.read_csv("Country_cleaned.csv")
print(df.describe())
