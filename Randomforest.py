
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Đọc dữ liệu từ file CSV
data = pd.read_csv('data.csv',encoding='utf-8')

label_encoders = {}

for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])


data = data.dropna()

# Phân chia features và target
X = data.drop(columns=['package', 'dateTime', 'isUpgrade', 'email'], axis=1) 

y = data['package']
print(X)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Huấn luyện mô hình Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Hàm để dự đoán cho dòng dữ liệu mới
def predict_new_data(new_data):
    new_data_scaled = scaler.transform([new_data])
    predicted_package = model.predict(new_data_scaled)
    return predicted_package[0]

# Dùng hàm để dự đoán cho dòng dữ liệu mới
new_data_row = ['Học sinh - sinh viên', '3 - 5 triệu', 1.75, 'Nam', 68, '1 - 3 buổi / tuần']  # Thay thế giá trị của các biến với dữ liệu mới
predicted_package = predict_new_data(new_data_row)
print("Predicted package for new data:", predicted_package)
