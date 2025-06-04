import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Đọc dữ liệu
data = pd.read_csv('train.csv')

# Xử lý thiếu dữ liệu đơn giản
data = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'Neighborhood', 'SalePrice']]
data.dropna(inplace=True)

# Đổi tên cột cho đồng bộ
data.columns = ['area', 'bedrooms', 'bathrooms', 'location', 'price']




# One-hot encoding cho location
encoder = OneHotEncoder(sparse_output=False)

encoded_location = encoder.fit_transform(data[['location']])
encoded_df = pd.DataFrame(encoded_location, columns=encoder.get_feature_names_out(['location']))

# Kết hợp đặc trưng
X = pd.concat([data[['area', 'bedrooms', 'bathrooms']], encoded_df], axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)


st.title("Dự đoán giá nhà")

area = st.slider("Diện tích (sqft)", 500, 5000, 1500)
bedrooms = st.slider("Số phòng ngủ", 1, 10, 3)
bathrooms = st.slider("Số phòng tắm", 1, 5, 2)
location = st.selectbox("Khu vực", encoder.categories_[0])

# Chuyển input người dùng thành định dạng phù hợp
input_data = np.array([[area, bedrooms, bathrooms]])
location_encoded = encoder.transform([[location]])
input_full = np.concatenate([input_data, location_encoded], axis=1)

# Dự đoán
predicted_price = model.predict(input_full)[0]
st.success(f"Giá nhà dự đoán: ${predicted_price:,.0f}")

st.subheader("Phân phối giá nhà theo khu vực")
fig, ax = plt.subplots(figsize=(10, 4))
sns.boxplot(x='location', y='price', data=data, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)
