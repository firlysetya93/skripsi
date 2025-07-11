import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Judul aplikasi
st.set_page_config(page_title="Wind Speed Data App", layout="wide")
st.title("📊 Aplikasi Analisis Data Kecepatan Angin")

# Upload file
uploaded_file = st.file_uploader("Silakan unggah file Excel Anda (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Baca data
    try:
        df = pd.read_excel(uploaded_file)
        st.success("File berhasil dibaca!")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
        st.stop()

    # Tampilkan 5 data awal
    st.subheader("🔍 Data Awal")
    st.dataframe(df.head())

    # Informasi dataset
    st.subheader("📋 Informasi Dataset")
    buffer = []
    df.info(buf=buffer)
    s = "\n".join(buffer)
    st.text(s)

    # Statistik deskriptif
    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(df.describe())

    # Cek missing values
    st.subheader("⚠️ Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.dataframe(missing)
    else:
        st.success("Tidak ada missing values!")

    # Visualisasi variabel numerik
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.subheader("📉 Visualisasi Garis")
        selected_col = st.selectbox("Pilih kolom numerik untuk divisualisasikan:", numeric_cols)
        st.line_chart(df[selected_col])
    else:
        st.warning("Tidak ditemukan kolom numerik untuk divisualisasikan.")
else:
    st.warning("Silakan upload file Excel terlebih dahulu (.xlsx)")
