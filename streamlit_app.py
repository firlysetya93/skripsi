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

    # ============================
    # 🔍 Data Awal
    # ============================
    st.subheader("🔍 Data Awal")
    st.dataframe(df.head())

    # ============================
    # 📋 Struktur Dataset
    # ============================
    st.subheader("📋 Struktur Dataset (`df.info()`)")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    # ============================
    # 📈 Statistik Deskriptif
    # ============================
    st.subheader("📊 Statistik Deskriptif")
    st.dataframe(df.describe())

    # ============================
    # 🗓️ Tambahkan Kolom Bulan & Musim
    # ============================
    st.subheader("📆 Menambahkan Kolom Bulan dan Musim")
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
    df['Bulan'] = df['TANGGAL'].dt.month

    def determine_season(month):
        if month in [12, 1, 2]:
            return 'HUJAN'
        elif month in [3, 4, 5]:
            return 'PANCAROBA I'
        elif month in [6, 7, 8]:
            return 'KEMARAU'
        elif month in [9, 10, 11]:
            return 'PANCAROBA II'

    df['Musim'] = df['Bulan'].apply(determine_season)
    st.dataframe(df[['TANGGAL', 'Bulan', 'Musim']].head())

    # ============================
    # 📊 Statistik FF_X per Musim
    # ============================
    st.subheader("📈 Statistik `FF_X` per Musim")
    grouped = df.groupby('Musim')['FF_X'].agg(['mean', 'max', 'min']).reset_index()
    st.dataframe(grouped)

    # ============================
    # ⚠️ Analisis Missing Value
    # ============================
    st.subheader("🔍 Missing Value Berdasarkan Musim")
    missing_rows = df[df.isnull().any(axis=1)]

    if missing_rows.empty:
        st.success("✅ Tidak ada missing values dalam dataset.")
    else:
        missing_by_group = missing_rows.groupby('Musim')
        for group, rows in missing_by_group:
            st.markdown(f"**Musim `{group}` memiliki {len(rows)} baris dengan missing value:**")
            st.dataframe(rows)

    # ============================
    # 🧼 Handling Missing per Musim
    # ============================
    st.subheader("🧹 Penanganan Missing Value Berdasarkan Rata-Rata Musim")

    def fill_missing_values(group):
        group['FF_X'] = group['FF_X'].fillna(group['FF_X'].mean())
        return group

    df_filled = df.groupby('Musim').apply(fill_missing_values)
    df_filled.reset_index(drop=True, inplace=True)

    st.write("✅ Dataset setelah missing value ditangani:")
    st.dataframe(df_filled.head())

    # Cek ulang sisa missing
    remaining_missing = df_filled.isnull().sum()
    st.write("🔍 Sisa missing values setelah penanganan:")
    st.dataframe(remaining_missing[remaining_missing > 0])

    # Penjelasan edukatif
    st.markdown("### ℹ️ Mengapa Missing Value Diisi Per Musim?")
    st.markdown("""
Penanganan missing value dilakukan **berdasarkan musim**, karena:

- Kecepatan angin (`FF_X`) dipengaruhi oleh faktor musiman seperti hujan, pancaroba, atau kemarau.
- Rata-rata `FF_X` di musim kemarau bisa sangat berbeda dengan musim hujan.
- Jika kita isi missing value dengan rata-rata global, data bisa bias dan analisis jadi menyesatkan.

🧠 Dengan mengisi berdasarkan **rata-rata per musim**, kita menjaga konteks dan pola musiman tetap terjaga.
""")

    # ============================
    # 📉 Visualisasi Kolom Numerik
    # ============================
    st.subheader("📉 Visualisasi Perubahan Kolom Numerik")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Pilih kolom numerik:", numeric_cols)
        st.line_chart(df[selected_col])

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
