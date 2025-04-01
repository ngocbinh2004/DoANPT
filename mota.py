# mota.py
# chủ đề : Phân tích dữ liệu chỉ số phát triển kinh tế thế giới Từ 1960 đến 2020.
import pandas as pd
import numpy as np

# Đọc dữ liệu sau khi làm sạch
df = pd.read_csv("Country_cleaned.csv")
print(df.describe())

# count :	Số lượng quốc gia có dữ liệu cho cột đó (ở đây là 247 quốc gia)
# mean :	Trung bình năm có dữ liệu (ví dụ: trung bình năm thương mại là ~2012)
# std :	Độ lệch chuẩn – cho biết dữ liệu trải rộng bao nhiêu (càng nhỏ, dữ liệu càng đồng đều)
# min : Năm sớm nhất có dữ liệu (vd: thương mại từ 1995, nước từ 1975)
# 25% / 50% / 75%	: Các mốc phân vị: 
# 25% quốc gia có dữ liệu trước mốc đó, 
# 50% là median (trung vị), 
# 75% là năm mà 3/4 quốc gia có dữ liệu trước đó

# 📌 Mục tiêu: cho thấy dữ liệu có thật, sạch sẽ, đã mô tả rõ.
# => tới bước 2: phantichbatthuong,outlier_filter.