import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Veri yükleme ve işleme
@st.cache_data
def load_data():
    file_path = r"C:\Users\hp\Desktop\veri_klasörü\Health_insurance.csv"
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Eksik verilerin kontrolü
    if data.isnull().sum().any():
        st.warning("Veri setinde eksik değerler bulunuyor. Lütfen kontrol edin!")

    # Kategorik değişkenlerin sayısallaştırılması
    data["sex"] = data["sex"].map({"female": 0, "male": 1})
    data["smoker"] = data["smoker"].map({"no": 0, "yes": 1})
    data["region"] = data["region"].map({"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3})
    return data

# Eğitim ve tahmin
@st.cache_resource
def train_model(data):
    x = data[["age", "sex", "bmi", "children", "smoker", "region"]]
    y = data["charges"]

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(xtrain, ytrain)
    return model

# Tahmin fonksiyonu
def predict_premium(model, age, sex, bmi, children, smoker, region):
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit uygulaması
st.title("Sağlık Sigortası Prim Tahmini")

# Veriyi yükleme ve işleme
data = load_data()
data = preprocess_data(data)

# Modeli eğitme
model = train_model(data)

# Kullanıcı girişi alma
st.header("Kişisel Bilgileriniz")
age = st.number_input("Yaş", min_value=0, max_value=120, step=1, key="age")
sex = st.selectbox("Cinsiyet", options=[0, 1], index=0, format_func=lambda x: "Kadın" if x == 0 else "Erkek", key="sex")
bmi = st.number_input("Vücut Kitle İndeksi (BMI)", min_value=0.0, max_value=100.0, step=0.1, key="bmi")
children = st.number_input("Çocuk Sayısı", min_value=0, max_value=20, step=1, key="children")
smoker = st.selectbox("Sigara İçiyor musunuz?", options=[0, 1], index=0, format_func=lambda x: "Hayır" if x == 0 else "Evet", key="smoker")
region = st.selectbox("Bölge", options=[0, 1, 2, 3], index=0, format_func=lambda x: ["Northeast", "Northwest", "Southeast", "Southwest"][x], key="region")

# Tahmin yapma
if st.button("Tahmini Hesapla"):
    if age is None or bmi is None:
        st.error("Lütfen yaş ve BMI bilgilerini doldurun!")
    else:
        premium = predict_premium(model, age, sex, bmi, children, smoker, region)
        st.success(f"Tahmini Sigorta Primi: {premium:.2f} TL")












