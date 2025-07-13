# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import io
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import optuna

st.set_page_config(page_title="🌪️ Aplikasi Prediksi Kecepatan Angin", layout="wide")

# Sidebar
menu = st.sidebar.selectbox("Navigasi Menu", [
    "🏠 Home",
    "📄 Upload Data",
    "📊 EDA",
    "⚙️ Preprocessing"
])

uploaded_file = st.sidebar.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.session_state.df = df
else:
    df = st.session_state.get('df', None)

if menu == "🏠 Home":
    st.title("🏠 Selamat Datang di Aplikasi Prediksi Kecepatan Angin")
    st.markdown("""
Aplikasi ini membantu kamu:
- 📊 Eksplorasi Data (EDA)
- ⚙️ Preprocessing berdasarkan musim
- 🧠 Pemodelan dengan LSTM
- 📈 Prediksi kecepatan angin
""")

elif menu == "📄 Upload Data":
    st.title("📄 Upload Data")
    if df is not None:
        st.success("File berhasil dibaca!")
        st.dataframe(df.head())
        st.dataframe(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        st.warning("Silakan upload file terlebih dahulu.")

elif menu == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    if df is not None:
        st.write(df.describe())
        if 'TANGGAL' in df.columns:
            df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
            df.set_index('TANGGAL', inplace=True)
            st.line_chart(df['FF_X'])
    else:
        st.warning("Silakan upload file terlebih dahulu.")

elif menu == "⚙️ Preprocessing":
    st.title("⚙️ Preprocessing")
    if df is not None:
        df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
        df['Bulan'] = df['TANGGAL'].dt.month

        def musim(bulan):
            if bulan in [12,1,2]: return 'HUJAN'
            elif bulan in [3,4,5]: return 'PANCAROBA I'
            elif bulan in [6,7,8]: return 'KEMARAU'
            else: return 'PANCAROBA II'

        df['Musim'] = df['Bulan'].apply(musim)

        df['FF_X'] = df.groupby('Musim')['FF_X'].transform(lambda x: x.fillna(x.mean()))
        df.set_index('TANGGAL', inplace=True)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['FF_X']])

        def series_to_supervised(data, n_in=1, n_out=1):
            df = pd.DataFrame(data)
            cols = [df.shift(i) for i in range(n_in, 0, -1)] + [df.shift(-i) for i in range(n_out)]
            agg = pd.concat(cols, axis=1)
            agg.dropna(inplace=True)
            return agg

        n_days = 6
        supervised = series_to_supervised(scaled, n_days, 1)
        values = supervised.values
        n_features = 1
        n_obs = n_days * n_features

        train_size = int(len(values) * 0.8)
        train, test = values[:train_size], values[train_size:]
        train_X, train_y = train[:, :n_obs], train[:, -1]
        test_X, test_y = test[:, :n_obs], test[:, -1]
        X_train = train_X.reshape((train_X.shape[0], n_days, n_features))
        X_test = test_X.reshape((test_X.shape[0], n_days, n_features))
        y_train = train_y.reshape(-1, 1)
        y_test = test_y.reshape(-1, 1)

        def objective(trial):
            model = Sequential()
            model.add(LSTM(trial.suggest_int('lstm_units', 10, 200), return_sequences=True, input_shape=(n_days, n_features)))
            model.add(Dropout(trial.suggest_float('dropout', 0.0, 0.5)))
            model.add(LSTM(trial.suggest_int('lstm_units2', 10, 100)))
            model.add(Dropout(trial.suggest_float('dropout2', 0.0, 0.5)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(X_train, y_train, validation_data=(X_test, y_test),
                      epochs=50, batch_size=32, verbose=0, callbacks=[early_stopping], shuffle=False)
            loss = model.evaluate(X_test, y_test, verbose=0)
            return loss

        st.info("Melakukan tuning hyperparameter...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)

        st.success("Tuning selesai.")
        st.json(study.best_params)

        # Final training
        best_params = study.best_params
        model = Sequential()
        model.add(LSTM(best_params['lstm_units'], return_sequences=True, input_shape=(n_days, n_features)))
        model.add(Dropout(best_params['dropout']))
        model.add(LSTM(best_params['lstm_units2']))
        model.add(Dropout(best_params['dropout2']))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=100, batch_size=32, verbose=1,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                            shuffle=False)

        # Predict
        y_pred = model.predict(X_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_test_inv = scaler.inverse_transform(y_test)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(y_test_inv, label='Aktual')
        ax.plot(y_pred_inv, label='Prediksi')
        ax.set_title("Prediksi vs Aktual")
        ax.legend()
        st.pyplot(fig)

        # Metrics
        def calculate_metrics(y_true, y_pred):
            y_true, y_pred = y_true.flatten(), y_pred.flatten()
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            return mae, rmse, r2, mape

        mae, rmse, r2, mape = calculate_metrics(y_test_inv, y_pred_inv)
        st.write(f"**MAE:** {mae:.3f}")
        st.write(f"**RMSE:** {rmse:.3f}")
        st.write(f"**R²:** {r2:.3f}")
        st.write(f"**MAPE:** {mape:.2f}%")
    else:
        st.warning("Silakan upload file terlebih dahulu.")
