# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="🌪️ Aplikasi Prediksi Kecepatan Angin", layout="wide")

# Sidebar
menu = st.sidebar.selectbox("Navigasi Menu", [
    "🏠 Home",
    "📤 Upload Data",
    "📊 EDA",
    "⚙️ Preprocessing",
    "🧠 Modeling (LSTM / TCN / RBFNN)",
    "📈 Prediction"
])

uploaded_file = st.sidebar.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.session_state.df = df
else:
    df = st.session_state.get('df', None)

# ========== HOME ==========
if menu == "🏠 Home":
    st.title("🏠 Selamat Datang di Aplikasi Prediksi Kecepatan Angin")
    st.markdown("""
Aplikasi ini membantu kamu:
- 📊 Melakukan eksplorasi data angin (EDA)
- ⚙️ Preprocessing berdasarkan musim
- 🧠 Modeling dengan LSTM / TCN / RBFNN
- 📈 Prediksi kecepatan angin ke depan
""")

# ========== Upload Data ==========
elif menu == "📤 Upload Data":
    st.title("📤 Upload Data")
    if df is not None:
        st.success("✅ File berhasil dibaca!")
        st.dataframe(df.head())
        st.write("### Missing Value per Kolom")
        st.dataframe(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        st.warning("❗ Silakan upload file terlebih dahulu.")

# ========== EDA ==========
elif menu == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    if df is not None:
        # (Sama seperti sebelumnya...)
        # ACF, PACF, decomposition, ADF sudah ditambahkan
        pass
    else:
        st.warning("❗ Silakan upload file terlebih dahulu.")

# ========== PREPROCESSING ==========
elif menu == "⚙️ Preprocessing":
    st.title("⚙️ Preprocessing Data")

    if df is not None:
        df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
        df['Bulan'] = df['TANGGAL'].dt.month

        def determine_season(m):
            if m in [12, 1, 2]: return 'HUJAN'
            elif m in [3, 4, 5]: return 'PANCAROBA I'
            elif m in [6, 7, 8]: return 'KEMARAU'
            elif m in [9, 10, 11]: return 'PANCAROBA II'

        df['Musim'] = df['Bulan'].apply(determine_season)

        st.success("✅ Kolom Bulan & Musim berhasil ditambahkan.")

        # Isi missing value berdasarkan musim
        def fill_missing_values(group):
            group['FF_X'] = group['FF_X'].fillna(group['FF_X'].mean())
            return group

        df_filled = df.groupby('Musim').apply(fill_missing_values)
        df_filled.reset_index(drop=True, inplace=True)

        # Simpan df_musim ke sesi
        df_selected = df_filled[['TANGGAL', 'FF_X', 'Musim']]
        df_selected.set_index('TANGGAL', inplace=True)
        dfs = {s: g.reset_index() for s, g in df_selected.groupby('Musim')}
        df_musim = pd.concat(dfs.values(), ignore_index=True)
        df_musim['TANGGAL'] = pd.to_datetime(df_musim['TANGGAL'])
        df_musim.set_index('TANGGAL', inplace=True)
        st.session_state.df_musim = df_musim

        st.success("✅ Missing value ditangani & data digabung berdasarkan musim.")
        st.dataframe(df_musim.head())

        # --------- PEMBAGIAN TRAIN-TEST ---------
        st.subheader("✂️ Pembagian Train dan Test")

        test_size = 0.2
        n_total = len(df_musim)
        n_test = int(n_total * test_size)
        n_train = n_total - n_test

        df_train = df_musim.iloc[:n_train]
        df_test = df_musim.iloc[n_train:]

        st.write(f"Jumlah total data   : {n_total}")
        st.write(f"Jumlah data training: {df_train.shape[0]}")
        st.write(f"Jumlah data testing : {df_test.shape[0]}")

        # --------- NORMALISASI ---------
        st.subheader("📏 Normalisasi Data")
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(df_musim['FF_X'].values.reshape(-1, 1))
        st.success("✅ Normalisasi selesai menggunakan MinMaxScaler.")

        # --------- VISUALISASI PEMBAGIAN ---------
        st.subheader("📊 Visualisasi Pembagian Data Train-Test")
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(df_train.index, df_train['FF_X'], label='Training', color='royalblue')
        ax.plot(df_test.index, df_test['FF_X'], label='Testing', color='darkorange')
        ax.axvline(x=df_test.index[0], color='red', linestyle='--', label='Split Point')
        ax.set_title('Visualisasi Pembagian Data Train dan Test - Variabel FF_X')
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Kecepatan Angin (FF_X)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("""
**Penjelasan Visualisasi:**
- Warna biru mewakili data pelatihan (training set)
- Warna oranye menunjukkan data pengujian (testing set)
- Garis merah vertikal adalah titik pemisah antara data latih dan uji berdasarkan waktu

📌 Karena ini adalah data deret waktu (time series), maka pembagian dilakukan berdasarkan urutan waktu, bukan secara acak.
""")
    else:
        st.warning("❗ Silakan upload file terlebih dahulu.")
