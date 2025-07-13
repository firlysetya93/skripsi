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
        st.subheader("🔍 Data Awal")
        st.dataframe(df.head())

        st.subheader("📋 Struktur Data")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.subheader("📈 Statistik Deskriptif")
        st.dataframe(df.describe())

        st.subheader("📉 Visualisasi Kolom Numerik")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            selected = st.selectbox("Pilih kolom:", num_cols)
            st.line_chart(df[selected])
    else:
        st.warning("❗ Silakan upload file terlebih dahulu.")

# ========== Preprocessing ==========
elif menu == "⚙️ Preprocessing":
    st.title("⚙️ Preprocessing Data")
    if df is not None:
        df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
        df['Bulan'] = df['TANGGAL'].dt.month

        def determine_season(m):
            if m in [12, 1, 2]: return 'HUJAN'
            elif m in [3, 4, 5]: return 'PANCAROBA I'
            elif m in [6, 7, 8]: return 'KEMARAU'
            else: return 'PANCAROBA II'

        df['Musim'] = df['Bulan'].apply(determine_season)
        st.success("✅ Kolom Bulan & Musim berhasil ditambahkan.")
        st.dataframe(df[['TANGGAL', 'Bulan', 'Musim']].head())

        st.subheader("🔍 Missing Value Sebelum Penanganan")
        st.dataframe(df.isnull().sum()[df.isnull().sum() > 0])

        def fill_missing_values(group):
            group['FF_X'] = group['FF_X'].fillna(group['FF_X'].mean())
            return group

        df_filled = df.groupby('Musim').apply(fill_missing_values)
        df_filled.reset_index(drop=True, inplace=True)

        df_selected = df_filled[['TANGGAL', 'FF_X', 'Musim']].set_index('TANGGAL')
        dfs = {s: g.reset_index() for s, g in df_selected.groupby('Musim')}
        df_musim = pd.concat(dfs.values(), ignore_index=True)

        st.session_state.df_musim = df_musim
        st.success("✅ Missing value berhasil diisi berdasarkan musim.")
        st.dataframe(df_musim.head())
    else:
        st.warning("❗ Silakan upload file terlebih dahulu.")

# ========== Modeling ==========
elif menu == "🧠 Modeling (LSTM / TCN / RBFNN)":
    st.title("🧠 Modeling dengan LSTM / TCN / RBFNN")
    st.info("Modul modeling belum diisi. Silakan tentukan parameter dan pelatihan di sini.")
    st.markdown("""
Kamu bisa menambahkan:
- Pelatihan per musim
- Visualisasi hasil training
- Evaluasi MAE / RMSE / R²
""")

# ========== Prediction ==========
elif menu == "📈 Prediction":
    st.title("📈 Prediksi Kecepatan Angin")
    st.info("Modul prediksi belum diisi. Di sini kamu bisa input data dan menampilkan hasil prediksi.")
    st.markdown("""
- Masukkan data input prediksi
- Load model terlatih (jika ada)
- Tampilkan hasil prediksi vs aktual
""")
