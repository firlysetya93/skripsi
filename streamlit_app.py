# streamlit_app.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import optuna

st.set_page_config(page_title="🌪️ Aplikasi Prediksi Kecepatan Angin", layout="wide")

# Sidebar
menu = st.sidebar.selectbox("Navigasi Menu", [
    "🏠 Home",
    "📤 Upload Data",
    "📊 EDA",
    "⚙️ Preprocessing",
    "🧠 Modeling (LSTM / TCN / RBFNN)",
    "📈 Prediction"
])

uploaded_file = st.sidebar.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.session_state.df = df
else:
    df = st.session_state.get('df', None)

# ========== HOME ==========
if menu == "🏠 Home":
    st.title("🏠 Selamat Datang di Aplikasi Prediksi Kecepatan Angin")
    st.markdown("""
Aplikasi ini membantu kamu:
- 📊 Melakukan eksplorasi data angin (EDA)
- ⚙️ Preprocessing berdasarkan musim
- 🧠 Modeling dengan LSTM / TCN / RBFNN
- 📈 Prediksi kecepatan angin ke depan
""")

# ========== Upload Data ==========
elif menu == "📤 Upload Data":
    st.title("📤 Upload Data")
    if df is not None:
        st.success("✅ File berhasil dibaca!")
        st.dataframe(df.head())
        st.write("### Missing Value per Kolom")
        st.dataframe(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        st.warning("❗ Silakan upload file terlebih dahulu.")

# ========== EDA ==========
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

        st.subheader("📉 Visualisasi Kolom Numerik")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            selected = st.selectbox("Pilih kolom:", num_cols)
            st.line_chart(df[selected])

        st.subheader("📉 ACF & PACF")
        if 'TANGGAL' in df.columns and 'FF_X' in df.columns:
            df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
            df.set_index('TANGGAL', inplace=True)
            ts = df['FF_X'].dropna()
            fig, axes = plt.subplots(3, 1, figsize=(14, 10))
            plot_acf(ts, lags=50, ax=axes[0])
            axes[0].set_title("Autocorrelation Function (ACF)")
            plot_pacf(ts, lags=50, ax=axes[1], method='ywm')
            axes[1].set_title("Partial Autocorrelation Function (PACF)")
            axes[2].plot(ts)
            axes[2].set_title("Time Series Plot")
            st.pyplot(fig)

            st.markdown("""
**Interpretasi Visualisasi:**

- **ACF (Autocorrelation Function):**
  Menunjukkan korelasi antara nilai saat ini dengan nilai sebelumnya (lag). Pola ACF membantu dalam mengidentifikasi komponen MA (Moving Average) pada data.

- **PACF (Partial Autocorrelation Function):**
  Menunjukkan korelasi langsung antara nilai sekarang dan nilai lag-n, setelah menghilangkan pengaruh dari lag-lag di antaranya. Pola PACF bermanfaat untuk menentukan komponen AR (AutoRegressive).

- **Time Series Plot:**
  Menampilkan pola keseluruhan data kecepatan angin dari waktu ke waktu. Dapat digunakan untuk melihat tren dan musiman.

💡 Jika ACF/PACF menunjukkan puncak berulang secara berkala, itu menandakan adanya pola musiman.
""")

        st.subheader("📉 Seasonal Decomposition")
        if 'FF_X' in df.columns:
            ts = df['FF_X'].dropna()
            result = seasonal_decompose(ts, model='additive', period=30)
            fig, axes = plt.subplots(4, 1, figsize=(16, 12))
            axes[0].plot(result.observed)
            axes[0].set_title("Observed")
            axes[1].plot(result.trend)
            axes[1].set_title("Trend")
            axes[2].plot(result.seasonal)
            axes[2].set_title("Seasonal")
            axes[3].plot(result.resid)
            axes[3].set_title("Residual")
            st.pyplot(fig)
            st.markdown("""
**Penjelasan Komponen Dekomposisi:**

- **Observed:** Data asli gabungan antara tren, musiman, dan noise.
- **Trend:** Menunjukkan kecenderungan jangka panjang (naik atau turun).
- **Seasonal:** Menggambarkan fluktuasi musiman berulang, misalnya bulanan atau musiman.
- **Residual:** Sisa dari data setelah tren dan musiman dihilangkan (noise).
""")

        st.subheader("📈 Uji Stasioneritas (ADF)")
        result = adfuller(ts, autolag='AIC')
        st.write(f"ADF Statistic : {result[0]:.4f}")
        st.write(f"p-value       : {result[1]:.4f}")
        st.write("Critical Values:")
        for key, value in result[4].items():
            st.write(f"   {key} : {value:.4f}")
        if result[1] <= 0.05:
            st.success("✅ Data stasioner (tolak H0): Pola statistik tetap sepanjang waktu")
        else:
            st.warning("⚠️ Data tidak stasioner (gagal tolak H0): Pola statistik berubah sepanjang waktu")
    else:
        st.warning("❗ Silakan upload file terlebih dahulu.")
# ========== PREPROCESSING ==========
elif menu == "⚙️ Preprocessing":
    st.title("⚙️ Preprocessing Data")

    if df is not None:
        df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
        df['Bulan'] = df['TANGGAL'].dt.month

        def determine_season(m):
            if m in [12, 1, 2]: return 'HUJAN'
            elif m in [3, 4, 5]: return 'PANCAROBA I'
            elif m in [6, 7, 8]: return 'KEMARAU'
            elif m in [9, 10, 11]: return 'PANCAROBA II'

        df['Musim'] = df['Bulan'].apply(determine_season)

        st.success("✅ Kolom Bulan & Musim berhasil ditambahkan.")

        # Isi missing value berdasarkan musim
        def fill_missing_values(group):
            group['FF_X'] = group['FF_X'].fillna(group['FF_X'].mean())
            return group

        df_filled = df.groupby('Musim').apply(fill_missing_values)
        df_filled.reset_index(drop=True, inplace=True)

        # Simpan df_musim ke sesi
        df_selected = df_filled[['TANGGAL', 'FF_X', 'Musim']]
        df_selected.set_index('TANGGAL', inplace=True)
        dfs = {s: g.reset_index() for s, g in df_selected.groupby('Musim')}
        df_musim = pd.concat(dfs.values(), ignore_index=True)
        df_musim['TANGGAL'] = pd.to_datetime(df_musim['TANGGAL'])
        df_musim.set_index('TANGGAL', inplace=True)
        st.session_state.df_musim = df_musim

        st.success("✅ Missing value ditangani & data digabung berdasarkan musim.")
        st.dataframe(df_musim.head())

        # --------- PEMBAGIAN TRAIN-TEST ---------
        st.subheader("✂️ Pembagian Train dan Test")

        test_size = 0.2
        n_total = len(df_musim)
        n_test = int(n_total * test_size)
        n_train = n_total - n_test

        df_train = df_musim.iloc[:n_train]
        df_test = df_musim.iloc[n_train:]

        st.write(f"Jumlah total data   : {n_total}")
        st.write(f"Jumlah data training: {df_train.shape[0]}")
        st.write(f"Jumlah data testing : {df_test.shape[0]}")

        # --------- NORMALISASI ---------
        st.subheader("📏 Normalisasi Data")
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(df_musim['FF_X'].values.reshape(-1, 1))
        st.success("✅ Normalisasi selesai menggunakan MinMaxScaler.")

        # --------- VISUALISASI PEMBAGIAN ---------
        st.subheader("📊 Visualisasi Pembagian Data Train-Test")
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(df_train.index, df_train['FF_X'], label='Training', color='royalblue')
        ax.plot(df_test.index, df_test['FF_X'], label='Testing', color='darkorange')
        ax.axvline(x=df_test.index[0], color='red', linestyle='--', label='Split Point')
        ax.set_title('Visualisasi Pembagian Data Train dan Test - Variabel FF_X')
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Kecepatan Angin (FF_X)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("""
**Penjelasan Visualisasi:**
- Warna biru mewakili data pelatihan (training set)
- Warna oranye menunjukkan data pengujian (testing set)
- Garis merah vertikal adalah titik pemisah antara data latih dan uji berdasarkan waktu

📌 Karena ini adalah data deret waktu (time series), maka pembagian dilakukan berdasarkan urutan waktu, bukan secara acak.
""")
    else:
        st.warning("❗ Silakan upload file terlebih dahulu.")
 # Normalisasi
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['FF_X']])

    # Series to supervised
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

    # Optuna tuning
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

    st.info("🔍 Proses tuning hyperparameter menggunakan Optuna...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    st.success("✅ Tuning selesai.")
    best_params = study.best_params
    st.json(best_params)

    # Final model training
    final_model = Sequential()
    final_model.add(LSTM(best_params['lstm_units'], return_sequences=True, input_shape=(n_days, n_features)))
    final_model.add(Dropout(best_params['dropout']))
    final_model.add(LSTM(best_params['lstm_units2']))
    final_model.add(Dropout(best_params['dropout2']))
    final_model.add(Dense(1))
    final_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = final_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                              epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping], shuffle=False)

    # Predict
    y_pred = final_model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test)

    # Plot actual vs prediction
    st.subheader("📈 Visualisasi Prediksi vs Aktual")
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(y_test_inv, label='Aktual')
    ax.plot(y_pred_inv, label='Prediksi')
    ax.set_title("Prediksi vs Aktual Kecepatan Angin")
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
    st.subheader("📊 Evaluasi Model")
    st.markdown(f"**MAE:** {mae:.3f}")
    st.markdown(f"**RMSE:** {rmse:.3f}")
    st.markdown(f"**R²:** {r2:.3f}")
    st.markdown(f"**MAPE:** {mape:.2f}%")
