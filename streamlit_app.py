import streamlit as st
import pandas as pd
import numpy as np
import io


# ========================== SETUP ==========================
st.set_page_config(page_title="🌪️ Prediksi Kecepatan Angin", layout="wide")
menu = st.sidebar.selectbox("Navigasi Menu", [
    "🏠 Home", "📊 EDA", "⚙️ Preprocessing", "🧠 Modeling (LSTM)", "📈 Prediction"
])

# ========================== UPLOAD ==========================
st.sidebar.header("📤 Upload File Excel")
uploaded_file = st.sidebar.file_uploader("Pilih file .xlsx", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
else:
    df = None

# ========================== HOME ==========================
if menu == "🏠 Home":
    st.title("🌪️ Aplikasi Prediksi Kecepatan Angin")
    st.markdown("""
        Aplikasi ini melakukan analisis dan prediksi kecepatan angin berdasarkan urutan waktu. 
        Data akan diproses per musim dan dilatih menggunakan model LSTM.
    """)

# ========================== EDA ==========================
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
    else:
        st.warning("❗ Silakan upload file terlebih dahulu.")

# ========================== PREPROCESSING ==========================
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

        # Handle Missing
        def fill_missing(group):
            group['FF_X'] = group['FF_X'].fillna(group['FF_X'].mean())
            return group

        df_filled = df.groupby('Musim').apply(fill_missing)
        df_filled.reset_index(drop=True, inplace=True)

        df_selected = df_filled[['TANGGAL', 'FF_X', 'Musim']].set_index('TANGGAL')
        dfs = {s: g.reset_index() for s, g in df_selected.groupby('Musim')}

        for s, d in dfs.items():
            st.markdown(f"### 🍂 Data Musim: `{s}`")
            st.dataframe(d.head())

        df_musim = pd.concat(dfs.values(), ignore_index=True)
        st.markdown("### 🔄 Gabungan Semua Data Musiman")
        st.dataframe(df_musim.head())
    else:
        st.warning("❗ Silakan upload file terlebih dahulu.")
