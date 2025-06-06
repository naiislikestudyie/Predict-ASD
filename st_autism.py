import streamlit as st
import numpy as np
import pandas as pd
import pickle
#library standarscaler

try: 
    from ann_mlp_manual import ANN
except ImportError:
    class ANN:
        pass

st.set_page_config(page_title="Prediksi Autisme dengan ANN", layout="centered")
st.title("Prediksi Autisme (ASD) dengan algoritma ANN")
st.write("Aplikasi ini menggunakan algoritma Artificial Neural Network (ANN) MLP untuk memprediksi kemungkinan seseorang mengidap Autisme berdasarkan data kuisioner.")

@st.cache_resource
def load_model():
    try:
        with open("ANN_model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

saved_model = load_model()

st.markdown("---")
st.subheader("Test ASD")

if saved_model:
    st.markdown("Masukkan jawaban dibawah ini")
    input_values = []

    age = st.text_input("Usia")
    gender = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
    jundice = st.selectbox("Apakah pernah kuning (jaundice)?", ['Ya', 'Tidak'])
    autism = st.selectbox("Riwayat autisme keluarga?", ['Ya', 'Tidak'])

    input_values.append(age)
    input_values.append(1 if gender == 'Laki-laki' else 0)
    input_values.append(1 if jundice == 'Ya' else 0)
    input_values.append(1 if autism == 'Ya' else 0)

    st.markdown("---")

    pertanyaan = [
        "Saya sering memperhatikan suara-suara kecil yang tidak diperhatikan orang lain.",
        "Saya biasanya lebih fokus pada keseluruhan gambar daripada detail kecil.",
        "Dalam kelompok sosial, saya dapat dengan mudah mengikuti beberapa percakapan sekaligus.",
        "Saya merasa mudah untuk beralih antara berbagai aktivitas.",
        "Saya menemukan bahwa saya lebih suka melakukan hal-hal dengan cara yang sama berulang kali.",
        "Saya sering merasa sulit untuk memahami maksud orang lain.",
        "Saya lebih suka melakukan sesuatu sendiri daripada bersama orang lain.",
        "Saya merasa sulit untuk membayangkan bagaimana rasanya menjadi orang lain.",
        "Saya sering memperhatikan detail kecil yang diabaikan orang lain.",
        "Saya merasa sulit untuk memahami lelucon atau sindiran."
    ]

    for i, q in enumerate(pertanyaan, start=1):
        jawaban = st.radio(
            f"{q}",
            ["Setuju", "Tidak setuju"],
            key=f"a{i}",
            label_visibility="visible",
            horizontal=True,
        )
        input_values.append(1 if jawaban == "Setuju" else 0)

    if st.button("Prediksi Autisme", key="prediksi_autisme_btn"):
        try:
            df_raw = pd.read_csv("cleaned_data.csv")
            fitur = df_raw[['age','gender','jundice','autism',
                            'A1_Score','A2_Score','A3_Score','A4_Score','A5_Score',
                            'A6_Score','A7_Score','A8_Score','A9_Score','A10_Score']]
            
            # scaler = StandardScaler()
            # fitur_scaled = scaler.fit_transform(fitur)

            # input_array = np.array(input_values, dtype=float).reshape(1, -1)
            # input_scaled = scaler.transform(input_array)[0]

            pred, prob = saved_model.predict(jawaban.tolist()) #(input_scaled.tolist())
            if pred == 1:
                st.error(f"Hasil: Kemungkinan MENGIDAP Autisme (Probabilitas: {prob:.2f})")
            else:
                st.success(f"Hasil: Tidak mengidap Autisme (Probabilitas: {prob:.2f})")
        except Exception as e:
            st.error(f"Terjadi error saat memproses prediksi: {e}")
else:
    st.warning("Model belum berhasil dimuat. Pastikan file 'ANN_model.pkl' tersedia dan dapat diakses.")
