import pandas as pd
import matplotlib.pyplot as plt
import stumpy

# Đọc dữ liệu GDP theo năm (giả sử đã được chuẩn hóa theo dạng thời gian)
df = pd.read_csv('WDI_GDP_timeseries.csv')

# Lọc dữ liệu GDP của Việt Nam
gdp_vietnam = df[df['Country Name'] == 'Vietnam'].iloc[:, 4:]  # bỏ cột CountryCode, Name, etc.

# Chuyển về mảng 1 chiều
gdp_series = gdp_vietnam.values.flatten()

# Tính matrix profile (m = chiều dài chuỗi con muốn tìm)
mp = stumpy.stump(gdp_series, m=5)

# Hiển thị biểu đồ chu kỳ
plt.plot(mp[:, 0])
plt.title("Phân tích chu kỳ GDP Việt Nam (STUMPY)")
plt.xlabel("Chỉ số thời gian")
plt.ylabel("Khoảng cách")
plt.show()
