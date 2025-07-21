# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tcn import TCN
import io
import math

st.set_page_config(page_title="🌪️ Aplikasi Prediksi Kecepatan Angin", layout="wide")

# Sidebar
menu = st.sidebar.selectbox("Navigasi Menu", [
    "🏠 Home",
    "📄 Upload Data",
    "📊 EDA",
    "⚙️ Preprocessing",
    "🧠 Modeling (LSTM)",
    "📈 Prediction"
])

uploaded_file = st.sidebar.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.session_state.df = df
else:
    df = st.session_state.get('df', None)

# ======================= HOME =======================
if menu == "🏠 Home":
    st.title("🏠 Selamat Datang di Aplikasi Prediksi Kecepatan Angin")
    st.markdown("""
Aplikasi ini membantu kamu:
- 📊 Melakukan eksplorasi data angin (EDA)
- ⚙️ Preprocessing berdasarkan musim
- 🧠 Modeling dengan LSTM
- 📈 Prediksi kecepatan angin ke depan
""")

# ======================= EDA =======================
elif menu == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    if df is not None:
        st.subheader("🔍 Data Awal")
        st.dataframe(df.head())
        st.subheader("📈 Statistik Deskriptif")
        st.dataframe(df.describe())

        if 'TANGGAL' in df.columns and 'FF_X' in df.columns:
            df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
            df.set_index('TANGGAL', inplace=True)
            ts = df['FF_X'].dropna()

            fig, ax = plt.subplots(figsize=(20, 5))
            ax.plot(ts)
            ax.set_title("Time Series Plot")
            st.pyplot(fig)

            result = adfuller(ts)
            st.write(f"ADF Statistic: {result[0]:.4f}")
            st.write(f"p-value: {result[1]:.4f}")
    else:
        st.warning("Upload data terlebih dahulu.")

# ======================= PREPROCESSING =======================
elif menu == "⚙️ Preprocessing":
    st.title("⚙️ Preprocessing Data")
    if df is not None:
        df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
        df['Bulan'] = df['TANGGAL'].dt.month
        
        def season(bulan):
            if bulan in [12,1,2]: return 'HUJAN'
            elif bulan in [3,4,5]: return 'PANCAROBA I'
            elif bulan in [6,7,8]: return 'KEMARAU'
            else: return 'PANCAROBA II'

        df['Musim'] = df['Bulan'].apply(season)
        df['FF_X'] = df.groupby('Musim')['FF_X'].transform(lambda x: x.fillna(x.mean()))
        df.set_index('TANGGAL', inplace=True)
        st.session_state.df_musim = df
        st.dataframe(df.head())
    else:
        st.warning("Upload data terlebih dahulu.")

# ======================= MODELING =======================
elif menu == "🧠 Modeling":
    st.title("🧠 Modeling")
    model_choice = st.selectbox("Pilih Model", ["LSTM", "TCN", "RBFNN"])

    if 'df_musim' in st.session_state:
        df = st.session_state.df_musim
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['FF_X']])

        def series_to_supervised(data, n_in=1, n_out=1):
            df = pd.DataFrame(data)
            cols = [df.shift(i) for i in range(n_in, 0, -1)] + [df.shift(-i) for i in range(n_out)]
            df_supervised = pd.concat(cols, axis=1)
            df_supervised.dropna(inplace=True)
            return df_supervised

        n_lag = 6
        supervised = series_to_supervised(scaled, n_lag, 1).values
        train_size = int(len(supervised)*0.8)
        train, test = supervised[:train_size], supervised[train_size:]
        X_train, y_train = train[:, :-1], train[:, -1]
        X_test, y_test = test[:, :-1], test[:, -1]

        if model_choice in ["LSTM", "TCN"]:
            X_train = X_train.reshape((X_train.shape[0], n_lag, 1))
            X_test = X_test.reshape((X_test.shape[0], n_lag, 1))

            model = Sequential()
            if model_choice == "LSTM":
                model.add(LSTM(64, input_shape=(n_lag, 1)))
            else:
                model.add(TCN(input_shape=(n_lag, 1)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

            y_pred = model.predict(X_test)

        elif model_choice == "RBFNN":
            from sklearn.linear_model import LinearRegression
            class RBF:
                def __init__(self, X, y, n_centers):
                    self.kmeans = KMeans(n_clusters=n_centers).fit(X)
                    self.centers = self.kmeans.cluster_centers_
                    self.sigma = np.mean([np.linalg.norm(c - X.mean(axis=0)) for c in self.centers])
                    self.lr = LinearRegression()
                    self.X = X
                    self.y = y

                def _rbf(self, x, center):
                    return np.exp(-np.linalg.norm(x - center)**2 / (2 * self.sigma**2))

                def transform(self, X):
                    return np.array([[self._rbf(x, c) for c in self.centers] for x in X])

                def fit(self):
                    X_rbf = self.transform(self.X)
                    self.lr.fit(X_rbf, self.y)

                def predict(self, X):
                    return self.lr.predict(self.transform(X))

            rbf = RBF(X_train, y_train, n_centers=10)
            rbf.fit()
            y_pred = rbf.predict(X_test)

        y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

        st.session_state.y_pred_inv = y_pred_inv
        st.session_state.y_test_inv = y_test_inv
        st.success(f"Model {model_choice} selesai dilatih.")

    else:
        st.warning("Lakukan preprocessing terlebih dahulu.")

# ======================= PREDICTION =======================
elif menu == "📈 Prediction":
    st.title("📈 Hasil Prediksi")
    if 'y_test_inv' in st.session_state and 'y_pred_inv' in st.session_state:
        y_test_inv = st.session_state.y_test_inv
        y_pred_inv = st.session_state.y_pred_inv

        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(y_test_inv, label='Actual')
        ax.plot(y_pred_inv, label='Predicted')
        ax.legend()
        st.pyplot(fig)

        def calculate_metrics(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            return pd.DataFrame({
                'MAE': [mae], 'RMSE': [rmse], 'R2 Score': [r2], 'MAPE': [mape]
            })

        st.subheader("📊 Evaluasi Model")
        st.dataframe(calculate_metrics(y_test_inv, y_pred_inv))
    else:
        st.warning("Belum ada hasil prediksi. Silakan lakukan modeling terlebih dahulu.")

