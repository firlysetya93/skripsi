import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Konfigurasi halaman
st.set_page_config(page_title="🌪️ Analisis Data Kecepatan Angin", layout="wide")
st.title("🌪️ Prediksi & Analisis Data Kecepatan Angin")

# Upload file Excel
uploaded_file = st.file_uploader("📤 Unggah file Excel Anda (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ File berhasil dibaca!")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat membaca file: {e}")
        st.stop()

    # Tampilkan data awal
    st.subheader("🔍 Data Awal")
    st.dataframe(df.head())

    # Struktur Dataset
    st.subheader("📋 Struktur Dataset (`df.info()`)")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    # Statistik Deskriptif
    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(df.describe())

    # Analisis Missing Values
    st.subheader("⚠️ Analisis Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        st.write("Kolom yang memiliki nilai hilang:")
        st.dataframe(missing)

        # Tampilkan baris yang hilang pada kolom FF_X jika ada
        if 'FF_X' in df.columns:
            rows_with_missing_ffx = df[df['FF_X'].isnull()]
            if not rows_with_missing_ffx.empty:
                st.markdown("#### 🔎 Baris dengan nilai hilang di kolom `FF_X`")
                st.dataframe(rows_with_missing_ffx)
            else:
                st.success("Kolom `FF_X` tidak memiliki nilai yang hilang.")
    else:
        st.success("🎉 Tidak ada missing values dalam dataset!")

    # Tombol hapus missing
    if st.button("🧹 Hapus semua baris yang memiliki missing values"):
        df = df.dropna()
        st.success("Baris yang mengandung missing values telah dihapus.")
        st.dataframe(df.head())

    # Visualisasi Garis
    st.subheader("📉 Visualisasi Perubahan Kolom Numerik")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Pilih kolom numerik:", numeric_cols)
        st.line_chart(df[selected_col])

        # Penjelasan visualisasi
        st.markdown("### ℹ️ Penjelasan Visualisasi")
        st.markdown(f"""
Visualisasi di atas menunjukkan **perubahan nilai `{selected_col}` dari waktu ke waktu**.

- **Sumbu X**: Baris data atau waktu (jika tersedia)
- **Sumbu Y**: Nilai dari `{selected_col}`
- Grafik ini membantu kamu mengenali tren umum, fluktuasi, atau lonjakan ekstrim

📌 **Contoh**: Jika kolom `{selected_col}` adalah `FF_X`, maka:
- Lonjakan → kemungkinan angin kencang
- Stabil → periode tenang

""")
    else:
        st.warning("Tidak ada kolom numerik untuk divisualisasikan.")
else:
    st.info("💡 Silakan unggah file Excel terlebih dahulu.")
