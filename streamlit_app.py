import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="📊 Preprocessing Musiman", layout="wide")
st.title("📊 Preprocessing Kecepatan Angin per Musim")

uploaded_file = st.sidebar.file_uploader("📤 Upload File Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ File berhasil dibaca!")

    # Tampilkan beberapa baris awal
    st.subheader("🔍 Data Awal")
    st.dataframe(df.head())

    # Cek missing value per kolom
    st.subheader("📌 Missing Value Tiap Kolom")
    missing = df.isnull().sum()
    st.write(missing[missing > 0])

    # Konversi TANGGAL dan ekstrak Bulan
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
    df['Bulan'] = df['TANGGAL'].dt.month

    # Tambah kolom musim
    def determine_season(month):
        if month in [12, 1, 2]: return 'HUJAN'
        elif month in [3, 4, 5]: return 'PANCAROBA I'
        elif month in [6, 7, 8]: return 'KEMARAU'
        else: return 'PANCAROBA II'

    df['Musim'] = df['Bulan'].apply(determine_season)

    # Tampilkan statistik per musim
    st.subheader("📊 Statistik Per Musim")
    grouped = df.groupby('Musim').agg({ 'FF_X': ['mean', 'max', 'min'] }).reset_index()
    st.dataframe(grouped)

    # Cek missing value per musim
    st.subheader("🔎 Missing Value per Musim")
    missing_rows = df[df.isnull().any(axis=1)]
    if not missing_rows.empty:
        for musim, rows in missing_rows.groupby('Musim'):
            st.markdown(f"#### Musim: {musim}")
            st.dataframe(rows)
    else:
        st.success("Tidak ada missing value yang ditemukan.")

    # Imputasi nilai hilang dengan mean per musim
    st.subheader("🧹 Imputasi Missing Value")
    def fill_missing_values(group):
        group['FF_X'] = group['FF_X'].fillna(group['FF_X'].mean())
        return group

    df_filled = df.groupby('Musim').apply(fill_missing_values)
    df_filled.reset_index(drop=True, inplace=True)

    st.write("Setelah imputasi, jumlah missing value:")
    st.write(df_filled.isnull().sum())

    # Pisahkan dataframe per musim
    st.subheader("📁 Dataframe Per Musim")
    dfs = {}
    for season, group in df_filled.groupby('Musim'):
        dfs[season] = group
        st.markdown(f"#### {season}")
        st.dataframe(group.head())

    # Gabungkan ulang setelah reset index
    st.subheader("🔄 Gabungan Semua Data Musiman dengan Reset Index")
    df_selected = df_filled[['TANGGAL', 'FF_X', 'Musim']].set_index('TANGGAL')
    dfs_reset = {season: group.reset_index() for season, group in df_selected.groupby('Musim')}
    df_musim = pd.concat(dfs_reset.values(), ignore_index=True)

    st.dataframe(df_musim.head(50))
else:
    st.warning("❗ Silakan upload file .xlsx terlebih dahulu.")
