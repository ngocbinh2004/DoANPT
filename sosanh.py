import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Country_cleaned.csv")
df["YearDiff"] = df["LatestTradeData"] - df["LatestWaterWithdrawalData"]

# 1. Trung bình chênh lệch năm
print("Chênh lệch trung bình (Trade - Water):", df["YearDiff"].mean())

# 2. Số quốc gia có dữ liệu trade mới hơn
print("Số quốc gia có dữ liệu thương mại mới hơn nước:",
      (df["YearDiff"] > 0).sum())

# 3. Quốc gia có chênh lệch lớn nhất
print("Top quốc gia có chênh lệch lớn:")
print(df[["ShortName", "LatestTradeData", "LatestWaterWithdrawalData", "YearDiff"]]
      .sort_values("YearDiff", ascending=False).head())


plt.figure(figsize=(10,6))
plt.scatter(df["LatestWaterWithdrawalData"], df["LatestTradeData"])
plt.xlabel("Năm dữ liệu Water")
plt.ylabel("Năm dữ liệu Trade")
plt.title("So sánh năm cập nhật dữ liệu Trade và Water")
plt.grid(True)
plt.savefig("Compare_Trade_Water.png")
plt.show()
