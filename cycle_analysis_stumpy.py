# nhóm em dùng thư viện STUMPY để phát hiện chu kỳ tăng trưởng 
# hoặc khủng hoảng GDP theo thời gian
# xong => file mi_income_group
import pandas as pd
import stumpy
import matplotlib.pyplot as plt

df = pd.read_csv("WDI_GDP_timeseries.csv")

# Chuyển chuỗi thời gian về dạng float64
ts = df['Argentina'].astype('float64').values

# Tính matrix profile
mp = stumpy.stump(ts, m=5)

# Vẽ biểu đồ kết quả
plt.figure(figsize=(10, 4))
plt.plot(mp[:, 0])
plt.title("Biểu đồ Matrix Profile - GDP Argentina (cửa sổ = 5)")
plt.xlabel("Chỉ số thời gian (Time Index)")
plt.ylabel("Giá trị Matrix Profile")
plt.tight_layout()
plt.show()

