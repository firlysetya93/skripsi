import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from math import sqrt

# ========== Set Seed ==========
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# ========== UI ==========
st.title("Analisis dan Prediksi dengan LSTM")
st.write("Aplikasi ini memuat data dan menampilkan informasi awal termasuk missing value.")

# ========== Upload File ==========
uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
if uploaded_file is not None:
    # Load dataset
    df = pd.read_excel(uploaded_file)
    
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Cek missing value
    st.subheader("Cek Missing Value")
    missing_info = df.isnull().sum()
    missing_info = missing_info[missing_info > 0]
    if not missing_info.empty:
        st.write("Kolom dengan missing value:")
        st.dataframe(missing_info)
    else:
        st.success("Tidak ada missing value dalam dataset.")

    # Optional: tampilkan deskripsi statistik
    if st.checkbox("Tampilkan Statistik Deskriptif"):
        st.write(df.describe())
else:
    st.warning("Silakan upload file Excel terlebih dahulu.")
