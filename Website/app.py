import streamlit as st
import pandas as pd
import numpy as np
import joblib
from data_kecamatan import load_kecamatan_data

def round_price_to_millions(price, base=5_000_000):
    return int(np.round(price / base) * base)

# --- Load database kecamatan ---
df_kec = load_kecamatan_data()
df_kec.columns = df_kec.columns.str.strip()  # jaga-jaga kolom ada spasi

# --- Konfigurasi tampilan ---
st.set_page_config(page_title="Prediksi Harga Sewa Ruko", layout="centered")

st.markdown("<h1 style='color:green;'>🧮 Prediksi Harga Sewa Ruko</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:18px;'>Isi data properti Anda dan sistem akan memilih model terbaik secara otomatis.</p>", unsafe_allow_html=True)

# --- Input user ---
kota = st.selectbox("Pilih Kota", df_kec["Kota"].unique())
filtered_kec = df_kec[df_kec["Kota"] == kota]["Kecamatan"].unique()
kecamatan = st.selectbox("Pilih Kecamatan", filtered_kec)

# Lookup kepadatan
kepadatan = df_kec[(df_kec["Kota"] == kota) & (df_kec["Kecamatan"] == kecamatan)]["Kepadatan"].values[0]

luas_tanah = st.number_input("Luas Tanah (m²)", min_value=0)
luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=0)
jumlah_lantai = st.number_input("Jumlah Lantai", min_value=1)
carport = st.number_input("Jumlah Carport", min_value=0)

# Fasilitas dinamis
fasilitas = []
fasilitas_count = st.number_input("Jumlah Fasilitas (Misal: AC, Ruko 2 in 1, dll)", min_value=0, max_value=20)
for i in range(fasilitas_count):
    fasilitas_input = st.text_input(f"Fasilitas #{i+1}")
    if fasilitas_input:
        fasilitas.append(fasilitas_input)

jumlah_fasilitas = len(fasilitas)

# --- Logika pemilihan model otomatis ---
model_choice = "random_forest" if kota.lower() in ["jakarta", "surabaya"] else "svr"

# Tombol prediksi
if st.button("🔍 Prediksi Harga Sewa"):
    try:
        # Load model
        model_path = "surabaya.pkl"
        model = joblib.load(model_path)

        # Buat dataframe input
        input_data = pd.DataFrame([{
            "Kecamatan": kecamatan,
            "Luas Bangunan": luas_bangunan,
            "Luas Tanah": luas_tanah,
            "Jumlah Lantai": jumlah_lantai,
            "Kepadatan": kepadatan,
            "Carport": carport,
            "Fasilitas": jumlah_fasilitas
        }])

        # Prediksi
        pred = model.predict(input_data)
        raw_harga = np.expm1(pred[0]) if "svr" in model_path else pred[0]
        harga = round_price_to_millions(raw_harga, base=5_000_000) 

        # Tampilkan hasil
        st.markdown(f"""
        <div style='
            background-color: #e8f5e9;
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid green;
            margin-top: 20px;
        '>
            <h3 style='color:#1b5e20;'>💰 Estimasi Harga Sewa:</h3>
            <h1 style='color:#2e7d32;'>Rp {harga:,.0f} / Tahun</h1>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Terjadi error saat memproses: {e}")
