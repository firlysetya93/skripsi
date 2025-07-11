import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set judul halaman
st.set_page_config(page_title="Data Eksplorasi - Skripsi", layout="wide")

st.title("📊 Eksplorasi Data Awal - Wind Speed Prediction")

# Upload file
uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx"])
if uploaded_file is not None:
    # Load data
    df = pd.read_excel(uploaded_file)

    # Tampilkan data awal
    st.subheader("🧾 Tampilkan 5 Data Teratas")
    st.dataframe(df.head())

    # Info struktur data
    st.subheader("📋 Informasi Dataset")
    buffer = []
    df.info(buf=buffer)
    s = "\n".join(buffer)
    st.text(s)

    # Statistik deskriptif
    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(df.describe())

    # Cek missing values
    st.subheader("🔍 Missing Values")
    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0])

    # Pilih kolom numerik untuk visualisasi
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.subheader("📉 Visualisasi Garis Variabel Numerik")
        col_to_plot = st.selectbox("Pilih kolom:", numeric_cols)
        st.line_chart(df[col_to_plot])
    else:
        st.warning("Tidak ada kolom numerik yang tersedia untuk divisualisasikan.")
else:
    st.info("Silakan unggah file Excel terlebih dahulu.")

