# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="🌪️ Prediksi Kecepatan Angin", layout="wide")

menu = st.sidebar.radio("Navigasi", [
    "🏠 Home",
    "📤 Upload Data",
    "⚙️ Preprocessing",
    "📊 EDA Time Series"
])

# Shared states
df_musim = None

def determine_season(month):
    if month in [12, 1, 2]: return 'HUJAN'
    elif month in [3, 4, 5]: return 'PANCAROBA I'
    elif month in [6, 7, 8]: return 'KEMARAU'
    else: return 'PANCAROBA II'

def fill_missing(group):
    group['FF_X'] = group['FF_X'].fillna(group['FF_X'].mean())
    return group

if menu == "🏠 Home":
    st.title("🌪️ Aplikasi Prediksi Kecepatan Angin")
    st.markdown("""
    Selamat datang! Aplikasi ini digunakan untuk:

    - 📤 Mengunggah data kecepatan angin
    - ⚙️ Melakukan preprocessing berdasarkan musim
    - 📊 Melakukan analisis time series
    - 🧠 Memprediksi kecepatan angin menggunakan LSTM (dan model lain)
    
    Pastikan Anda mengunggah file Excel dengan kolom **TANGGAL** dan **FF_X**.
    """)

elif menu == "📤 Upload Data":
    st.title("📤 Upload Data")
    uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("✅ File berhasil dibaca!")
        st.dataframe(df.head())
        st.write("### Missing Value per Kolom")
        st.write(df.isnull().sum())
    else:
        st.info("Silakan upload file terlebih dahulu.")

elif menu == "⚙️ Preprocessing":
    st.title("⚙️ Preprocessing Berdasarkan Musim")

    if 'df' in st.session_state:
        df = st.session_state.df.copy()

        df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
        df['Bulan'] = df['TANGGAL'].dt.month
        df['Musim'] = df['Bulan'].apply(determine_season)

        df_filled = df.groupby('Musim').apply(fill_missing)
        df_filled.reset_index(drop=True, inplace=True)

        df_selected = df_filled[['TANGGAL', 'FF_X', 'Musim']].set_index('TANGGAL')
        dfs_reset = {season: group.reset_index() for season, group in df_selected.groupby('Musim')}
        df_musim = pd.concat(dfs_reset.values(), ignore_index=True)

        st.session_state.df_musim = df_musim

        st.success("✅ Preprocessing selesai!")
        st.dataframe(df_musim.head())
    else:
        st.warning("Silakan upload data terlebih dahulu.")

elif menu == "📊 EDA Time Series":
    st.title("📊 Analisis Time Series Kecepatan Angin")

    if 'df_musim' in st.session_state:
        df_musim = st.session_state.df_musim.copy()
        df_musim['TANGGAL'] = pd.to_datetime(df_musim['TANGGAL'])
        df_musim = df_musim.set_index('TANGGAL')

        ts = df_musim['FF_X'].dropna()

        st.subheader("📉 Seasonal Decomposition")
        result = seasonal_decompose(ts, model='additive', period=30)

        fig, axes = plt.subplots(4, 1, figsize=(18, 10), sharex=True)
        result.observed.plot(ax=axes[0], title='Observed', color='blue')
        result.trend.plot(ax=axes[1], title='Trend', color='orange')
        result.seasonal.plot(ax=axes[2], title='Seasonal', color='green')
        result.resid.plot(ax=axes[3], title='Residual', color='red')
        st.pyplot(fig)

        st.subheader("📌 Uji Stasioneritas - ADF Test")
        result_adf = adfuller(ts, autolag='AIC')
        st.write(f"ADF Statistic : {result_adf[0]:.4f}")
        st.write(f"p-value       : {result_adf[1]:.4f}")
        for key, value in result_adf[4].items():
            st.write(f"   {key} : {value:.4f}")

        if result_adf[1] <= 0.05:
            st.success("✅ Data stasioner (tolak H0)")
        else:
            st.warning("⚠️ Data tidak stasioner (gagal tolak H0)")

        st.subheader("📊 Plot ACF, PACF dan Time Series")
        fig, axes = plt.subplots(3, 1, figsize=(18, 14))
        plot_acf(ts, lags=50, ax=axes[0])
        plot_pacf(ts, lags=50, ax=axes[1], method='ywm')
        axes[2].plot(ts, color='blue')
        axes[2].set_title('Time Series Plot - FF_X')
        axes[2].set_xlabel('Tanggal')
        axes[2].set_ylabel('FF_X')
        st.pyplot(fig)

        st.subheader("🔀 Train-Test Split")
        test_size = 0.2
        n_total = len(df_musim)
        n_test = int(n_total * test_size)
        n_train = n_total - n_test
        df_train = df_musim.iloc[:n_train]
        df_test = df_musim.iloc[n_train:]

        fig_split = plt.figure(figsize=(22, 6))
        plt.plot(df_train.index, df_train['FF_X'], label='Training', color='royalblue')
        plt.plot(df_test.index, df_test['FF_X'], label='Testing', color='darkorange')
        plt.axvline(x=df_test.index[0], color='red', linestyle='--', label='Split Point')
        plt.title("Visualisasi Pembagian Data Train dan Test")
        plt.xlabel("Tanggal")
        plt.ylabel("FF_X")
        plt.legend()
        plt.grid(True)
        st.pyplot(fig_split)
    else:
        st.warning("Silakan lakukan preprocessing terlebih dahulu.")
