
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Đọc dữ liệu từ file CSV
data = pd.read_csv('data.csv',encoding='utf-8')

label_encoders = {}

for column, dtype in data.dtypes.items():
    print(column)
    if(dtype == 'object') :
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])


data = data.dropna()

# Phân chia features và target
X = data.drop(columns=['package','isGym', 'dateTime', 'isUpgrade', 'email'], axis=1) 

y = data['package']
print(y)
print(X)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Huấn luyện mô hình Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Hàm để dự đoán cho dòng dữ liệu mới
def predict_new_data(new_data):
    new_data_encoded = []  # Danh sách để lưu trữ các giá trị được mã hóa

    # Duyệt qua từng cột trong new_data
     # Duyệt qua từng cột trong new_data
    for column in new_data.keys():
        # Kiểm tra nếu cột là kiểu 'object' thì mới mã hóa
        if column in label_encoders:
            # Lấy giá trị của cột từ new_data, nếu không tồn tại thì lấy giá trị mặc định là ''
            value = new_data.get(column, '')
            
            # Mã hóa giá trị của cột và thêm vào danh sách new_data_encoded
            encoded_value = label_encoders[column].transform([value])[0]
            
            new_data_encoded.append(encoded_value)
        else:
            # Nếu là kiểu số thì giữ nguyên giá trị và thêm vào danh sách new_data_encoded
            new_data_encoded.append(new_data[column])

    new_data_scaled = scaler.transform([new_data_encoded])
    predicted_package = model.predict(new_data_scaled)
    return predicted_package[0]

# Dùng hàm để dự đoán cho dòng dữ liệu mới

new_data_row = {
    'job': 'Học sinh - sinh viên',
    'income': 'Trên 20 triệu',
    'height': 1.75,
    'gender': 'Nam',
    'weight': 68,
    'workout_frequency': '1 - 3 buổi / tuần'
}
print(new_data_row.keys())

predicted_package = predict_new_data(new_data_row)
print("Predicted package for new data:", predicted_package)
