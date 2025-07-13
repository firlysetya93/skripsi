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

        st.subheader("📉 ACF & PACF")
        if 'TANGGAL' in df.columns and 'FF_X' in df.columns:
            df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
            df.set_index('TANGGAL', inplace=True)
            ts = df['FF_X'].dropna()
            fig, axes = plt.subplots(3, 1, figsize=(14, 10))
            plot_acf(ts, lags=50, ax=axes[0])
            axes[0].set_title("Autocorrelation Function (ACF)")
            plot_pacf(ts, lags=50, ax=axes[1], method='ywm')
            axes[1].set_title("Partial Autocorrelation Function (PACF)")
            axes[2].plot(ts)
            axes[2].set_title("Time Series Plot")
            st.pyplot(fig)

            st.markdown("""
**Interpretasi Visualisasi:**

- **ACF (Autocorrelation Function):**
  Menunjukkan korelasi antara nilai saat ini dengan nilai sebelumnya (lag). Pola ACF membantu dalam mengidentifikasi komponen MA (Moving Average) pada data.

- **PACF (Partial Autocorrelation Function):**
  Menunjukkan korelasi langsung antara nilai sekarang dan nilai lag-n, setelah menghilangkan pengaruh dari lag-lag di antaranya. Pola PACF bermanfaat untuk menentukan komponen AR (AutoRegressive).

- **Time Series Plot:**
  Menampilkan pola keseluruhan data kecepatan angin dari waktu ke waktu. Dapat digunakan untuk melihat tren dan musiman.

💡 Jika ACF/PACF menunjukkan puncak berulang secara berkala, itu menandakan adanya pola musiman.
""")

        st.subheader("📉 Seasonal Decomposition")
        if 'FF_X' in df.columns:
            ts = df['FF_X'].dropna()
            result = seasonal_decompose(ts, model='additive', period=30)
            fig, axes = plt.subplots(4, 1, figsize=(16, 12))
            axes[0].plot(result.observed)
            axes[0].set_title("Observed")
            axes[1].plot(result.trend)
            axes[1].set_title("Trend")
            axes[2].plot(result.seasonal)
            axes[2].set_title("Seasonal")
            axes[3].plot(result.resid)
            axes[3].set_title("Residual")
            st.pyplot(fig)
            st.markdown("""
**Penjelasan Komponen Dekomposisi:**

- **Observed:** Data asli gabungan antara tren, musiman, dan noise.
- **Trend:** Menunjukkan kecenderungan jangka panjang (naik atau turun).
- **Seasonal:** Menggambarkan fluktuasi musiman berulang, misalnya bulanan atau musiman.
- **Residual:** Sisa dari data setelah tren dan musiman dihilangkan (noise).
""")

        st.subheader("📈 Uji Stasioneritas (ADF)")
        result = adfuller(ts, autolag='AIC')
        st.write(f"ADF Statistic : {result[0]:.4f}")
        st.write(f"p-value       : {result[1]:.4f}")
        st.write("Critical Values:")
        for key, value in result[4].items():
            st.write(f"   {key} : {value:.4f}")
        if result[1] <= 0.05:
            st.success("✅ Data stasioner (tolak H0): Pola statistik tetap sepanjang waktu")
        else:
            st.warning("⚠️ Data tidak stasioner (gagal tolak H0): Pola statistik berubah sepanjang waktu")
    else:
        st.warning("❗ Silakan upload file terlebih dahulu.")
