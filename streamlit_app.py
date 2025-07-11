import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io


# Konfigurasi halaman
st.set_page_config(page_title="Pipeline Analisis Data Angin", layout="wide")
st.title("🌪️ Prediksi Data Kecepatan Angin")

# Upload file
uploaded_file = st.file_uploader("📤 Unggah file Excel Anda (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Load data
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ File berhasil dibaca!")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat membaca file: {e}")
        st.stop()

    # Tampilkan data awal
    st.subheader("🔍 Data Awal")
    st.dataframe(df.head())

    # Informasi struktur dataset
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)


    # Statistik deskriptif
    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(df.describe())

    # ============================
    # 🔍 Missing Value Analysis
    # ============================
    st.subheader("⚠️ Analisis Missing Values")

    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        st.write("Kolom yang memiliki missing values:")
        st.dataframe(missing_values)

        if 'FF_X' in df.columns:
            rows_with_missing_ffx = df[df['FF_X'].isnull()]
            if not rows_with_missing_ffx.empty:
                st.markdown("#### 🔎 Baris dengan nilai hilang di kolom `FF_X`")
                st.dataframe(rows_with_missing_ffx)
            else:
                st.success("Tidak ada nilai hilang di kolom `FF_X`.")
        else:
            st.warning("⚠️ Kolom `FF_X` tidak ditemukan.")
    else:
        st.success("🎉 Tidak ada missing values dalam dataset!")

    # ============================
    # 🧹 Tombol untuk hapus missing
    # ============================
    if st.button("🧹 Hapus semua baris dengan missing values"):
        df = df.dropna()
        st.success("Baris dengan missing values telah dihapus.")
        st.dataframe(df.head())

    # ============================
    # 📊 Visualisasi kolom numerik
    # ============================
    st.subheader("📉 Visualisasi Kolom Numerik")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Pilih kolom untuk visualisasi:", numeric_cols)
        st.line_chart(df[selected_col])
    else:
        st.warning("Tidak ada kolom numerik tersedia.")
else:
    st.info("💡 Silakan upload file Excel terlebih dahulu untuk memulai.")
