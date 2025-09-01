import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras_tcn import TCN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
import tempfile
from datetime import timedelta



# === Sidebar menu ===
# === Sidebar menu ===
st.sidebar.title("📂 Menu")

# Cek apakah ada state menu yang tersimpan
if "menu" not in st.session_state:
    st.session_state["menu"] = "Preprocessing & Analisis Musim"

menu = st.sidebar.radio(
    "Pilih Halaman", 
    ["Preprocessing & Analisis Musim", 
     "Transformasi Supervised & Splitting", 
     "Hyperparameter Tuning (TCN)", 
     "Evaluasi Model"],
    index=["Preprocessing & Analisis Musim", 
           "Transformasi Supervised & Splitting", 
           "Hyperparameter Tuning (TCN)", 
           "Evaluasi Model"].index(st.session_state["menu"])  # simpan posisi terakhir
)

# === Menu 1: Preprocessing & Analisis Musim ===
if menu == "Preprocessing & Analisis Musim":
    st.title("📊 Analisis Kecepatan Angin")
    uploaded_file = st.file_uploader("Unggah file Excel", type=['xlsx'])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.subheader("📊 Preview Data (5 Baris Pertama)")
        st.dataframe(df.head())

        st.subheader("🧩 Jumlah Missing Values per Kolom")
        missing_values = df.isnull().sum()
        st.dataframe(missing_values[missing_values > 0])

        if 'FF_X' in df.columns:
            st.subheader("🔎 Baris dengan Missing Values pada Kolom 'FF_X'")
            st.dataframe(df[df['FF_X'].isnull()])
        else:
            st.warning("⚠️ Kolom 'FF_X' tidak ditemukan dalam dataset.")

        if 'TANGGAL' in df.columns:
            try:
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
                    else:
                        return 'UNKNOWN'

                df['Musim'] = df['Bulan'].apply(determine_season)
                df['tahun'] = df['TANGGAL'].dt.year

                st.session_state['df'] = df
                st.subheader("📋 Data Setelah Ditambah Kolom Bulan & Musim")
                st.dataframe(df.head())

                st.subheader("📊 Statistik Kecepatan Angin Berdasarkan Musim")
                grouped = df.groupby('Musim').agg({'FF_X': ['mean', 'max', 'min']}).reset_index()
                grouped.columns = ['Musim', 'FF_X Mean', 'FF_X Max', 'FF_X Min']
                st.dataframe(grouped)

                df_selected = df[['TANGGAL', 'FF_X', 'Musim']].copy()
                df_selected = df_selected.set_index('TANGGAL')

                dfs = {}
                for season, group in df_selected.groupby('Musim'):
                    dfs[season] = group.reset_index()

                st.subheader("🗂️ Data Per Musim")
                for season, df_season in dfs.items():
                    st.markdown(f"### Musim: {season}")
                    st.dataframe(df_season.head())

                df_musim = pd.concat(dfs.values(), ignore_index=True)
                df_musim = df_musim.sort_values('TANGGAL').reset_index(drop=True)
                st.session_state['df_musim'] = df_musim

                st.subheader("📅 Data Gabungan (Diurutkan Berdasarkan Tanggal)")
                st.dataframe(df_musim.head(1000))

                # --- Analisis Time Series ---
                st.subheader("📈 Rata-Rata Kecepatan Angin per Tahun")
                rata_tahunan = df.groupby('tahun')['FF_X'].mean()
                fig1, ax1 = plt.subplots(figsize=(10, 5))
                ax1.plot(rata_tahunan.index, rata_tahunan.values, marker='o')
                ax1.set_xlabel('Tahun')
                ax1.set_ylabel('Rata-rata Kecepatan Angin (m/s)')
                ax1.set_title('Rata-Rata Kecepatan Angin per Tahun')
                ax1.grid(True)
                ax1.set_xticks(rata_tahunan.index)
                st.pyplot(fig1)
            # --- Uji Stasioneritas ADF dan ACF/PACF Seluruh Data ---
                st.subheader("📉 Uji Stasioneritas (ADF Test) per Musim")
                adf_results = []
                for season, df_season in dfs.items():
                    series = df_season['FF_X'].dropna()
                    adf_result = adfuller(series)
                    adf_results.append({
                        'Musim': season,
                        'ADF Statistic': adf_result[0],
                        'p-value': adf_result[1],
                        'Critical Value 5%': adf_result[4]['5%']
                    })
                st.dataframe(pd.DataFrame(adf_results))

                st.subheader("🔁 ACF dan PACF Plot per Musim (100 Lags)")
                # Pastikan kolom 'TANGGAL' menjadi datetime
                if 'TANGGAL' in df_musim.columns:
                    df_musim['TANGGAL'] = pd.to_datetime(df_musim['TANGGAL'])
                    df_musim.set_index('TANGGAL', inplace=True)
        
                # Ambil hanya kolom FF_X dan drop NaN
                if 'FF_X' in df_musim.columns:
                    ts = df_musim['FF_X'].dropna()
        
                    # --- Uji Stasioneritas: Augmented Dickey-Fuller (ADF) ---
                    result = adfuller(ts, autolag='AIC')
        
                    st.markdown("### Hasil Uji ADF untuk FF_X")
                    st.write(f"**ADF Statistic** : {result[0]:.4f}")
                    st.write(f"**p-value**       : {result[1]:.4f}")
                    st.write("**Critical Values:**")
                    for key, value in result[4].items():
                        st.write(f"   {key} : {value:.4f}")
                    if result[1] <= 0.05:
                        st.success("✅ Data stasioner (tolak H0)")
                    else:
                        st.warning("⚠️ Data tidak stasioner (gagal tolak H0)")
        
                    # --- Plot ACF, PACF, dan Time Series ---
                    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
                    plt.subplots_adjust(hspace=0.5)
        
                    # Plot ACF
                    plot_acf(ts, lags=50, ax=axes[0])
                    axes[0].set_title("Autocorrelation Function (ACF) - FF_X")
        
                    # Plot PACF
                    plot_pacf(ts, lags=50, ax=axes[1], method='ywm')
                    axes[1].set_title("Partial Autocorrelation Function (PACF) - FF_X")
        
                    # Plot Time Series
                    axes[2].plot(ts, color='blue')
                    axes[2].set_title("Time Series Plot - FF_X")
                    axes[2].set_xlabel("Tanggal")
                    axes[2].set_ylabel("Kecepatan Angin (FF_X)")
        
                    st.pyplot(fig)
                st.success("✅ Preprocessing dan analisis musiman selesai! Data siap digunakan di menu berikutnya.")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat memproses tanggal: {e}")
        else:
            st.warning("⚠️ Kolom 'TANGGAL' tidak ditemukan dalam dataset.")
                            # Jika berhasil selesai preprocessing
        st.success("✅ Preprocessing dan analisis musiman selesai! Data siap digunakan di menu berikutnya.")

        # Tombol ke tahap berikutnya
        if st.button("➡️ Lanjut ke Tahap Berikutnya"):
            st.session_state["menu"] = "Transformasi Supervised & Splitting"
            st.rerun()   # refresh app agar langsung pindah menu
    else:
        st.info("⬆️ Silakan upload file Excel (.xlsx) terlebih dahulu.")
    
# === Menu 4: Transformasi Supervised & Splitting ===
if menu == "Transformasi Supervised & Splitting":
    st.title("🔁 Transformasi Supervised Learning")

    if "df_musim" not in st.session_state:
        st.warning("❗ Data musim belum tersedia. Silakan lakukan preprocessing terlebih dahulu.")
        st.stop()
    
    df_musim = st.session_state["df_musim"].copy()

    # --- Normalisasi ---
    st.subheader("📊 Normalisasi Data")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(df_musim[['FF_X']].values.astype('float32'))
    df_musim['FF_X_scaled'] = scaled_values
    st.session_state['scaler'] = scaler

    # --- Train-test split ---
    st.subheader("✂️ Pembagian Data Train dan Test")
    df_train, df_test = train_test_split(df_musim, test_size=0.2, shuffle=False)
    st.session_state['df_train'] = df_train
    st.session_state['df_test'] = df_test

    # --- Visualisasi ---
    features = ['FF_X']
    for feature in features:
        fig, ax = plt.subplots(figsize=(22, 6))
        ax.plot(df_train.index, df_train[feature], label='Training', color='blue')
        ax.plot(df_test.index, df_test[feature], label='Testing', color='orange')
        ax.set_title(f'Pembagian Data Train dan Test pada Variabel {feature}')
        ax.legend()
        st.pyplot(fig)

    st.success("✅ Data telah dinormalisasi dan dibagi menjadi train/test.")
    # === Fungsi Transformasi Supervised ===
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        df = pd.DataFrame(data)
        n_vars = df.shape[1]
        cols, names = [], []

        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [f'var{j+1}(t-{i})' for j in range(n_vars)]

        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [f'var{j+1}(t)' for j in range(n_vars)]
            else:
                names += [f'var{j+1}(t+{i})' for j in range(n_vars)]

        agg = pd.concat(cols, axis=1)
        agg.columns = names

        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # === Normalisasi ulang FF_X jika belum disimpan ===
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df_train[['FF_X']])
    st.session_state['scaler'] = scaler

    # === Parameter Lag ===
    n_days = st.slider("⏳ Jumlah Hari Input (Lag)", min_value=1, max_value=30, value=6)
    n_features = st.session_state.get("n_features", 1)

    st.session_state['n_days'] = n_days
    st.session_state['n_features'] = n_features

    # === Transformasi ke Supervised Format ===
    reframed = series_to_supervised(scaled, n_in=n_days, n_out=1)
    st.session_state['reframed'] = reframed

    st.success(f"✅ Data berhasil diubah ke format supervised dengan {n_days} lag hari.")
    st.subheader("📄 Contoh Data Supervised")
    st.dataframe(reframed.head(10))

    # === Splitting Train-Test ===
    st.subheader("✂️ Pembagian Data Train dan Test")
    values = reframed.values
    train, test = train_test_split(values, test_size=0.2, shuffle=False)

    # Ambil indeks tanggal sesuai reframed
    date_reframed = df_train.index[reframed.index]
    date_train = date_reframed[:len(train)]
    date_test = date_reframed[len(train):]

    n_obs = n_days * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features:]
    test_X, test_y = test[:, :n_obs], test[:, -n_features:]

    # Reshape ke 3D
    X_train = train_X.reshape((train_X.shape[0], n_days, n_features))
    X_test = test_X.reshape((test_X.shape[0], n_days, n_features))

    # Simpan ke session state
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = train_y
    st.session_state['y_test'] = test_y
    st.session_state['date_train'] = date_train
    st.session_state['date_test'] = date_test

    st.success("✅ Data berhasil dibagi ke bentuk input-output LSTM.")

    # === Ringkasan Dimensi ===
    st.markdown("**🧾 Ringkasan Dimensi Data:**")
    st.write("Total features:", n_features)
    st.write("X_train:", X_train.shape)
    st.write("X_test:", X_test.shape)
    st.write("y_train:", train_y.shape)
    st.write("y_test:", test_y.shape)

    # === Visualisasi Contoh Data ===
    st.subheader("🔍 Contoh Data Input dan Output")

    with st.expander("📌 Contoh struktur input X_train[0]"):
        st.write(X_train[0])

    with st.expander("📌 Contoh target y_train[0]"):
        st.write(train_y[0])

    # === Tanggal Train-Test ===
    st.subheader("🗓️ Tanggal Data Train dan Test")
    st.write("Tanggal train:", date_train.min(), "→", date_train.max())
    st.write("Tanggal test:", date_test.min(), "→", date_test.max())
        # === Tombol ke tahap berikutnya ===
    if st.button("➡️ Lanjut ke Hyperparameter Tuning (TCN)"):
        st.session_state["menu"] = "Hyperparameter Tuning (TCN)"
        st.rerun()
    
if menu == "Hyperparameter Tuning (TCN)":
    st.title("🎯 Hyperparameter Tuning dengan Optuna (TCN)")

    if 'X_train' not in st.session_state or 'y_train' not in st.session_state:
        st.warning("🚨 Data belum diproses! Silakan lakukan preprocessing, transformasi supervised, dan splitting terlebih dahulu.")
    else:
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        n_features = st.session_state.n_features
        df_train = st.session_state['df_train']
        df_test = st.session_state['df_test']

        n_trials = st.number_input("🔁 Jumlah Percobaan (Trials)", min_value=10, max_value=100, value=50, step=10)

        def objective(trial):
            nb_filters = trial.suggest_int('filters', 16, 128)
            kernel_size = trial.suggest_int('kernel_size', 2, 6)
            dilations_level = trial.suggest_int('dilations_level', 2, 5)
            dilations = [2 ** i for i in range(dilations_level)]
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            dense_units = trial.suggest_int('dense_units', 10, 200)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            epochs = trial.suggest_int('epochs', 20, 100)
            batch_size = trial.suggest_int('batch_size', 16, 128)

            model = Sequential()
            model.add(TCN(
                nb_filters=nb_filters,
                kernel_size=kernel_size,
                dilations=dilations,
                dropout_rate=dropout_rate,
                return_sequences=False,
                input_shape=(X_train.shape[1], X_train.shape[2])
            ))
            model.add(Dropout(dropout_rate))
            model.add(Dense(dense_units, activation='relu'))
            model.add(Dense(n_features))

            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mean_squared_error')

            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                      validation_data=(X_test, y_test),
                      callbacks=[early_stopping], verbose=0, shuffle=False)

            loss = model.evaluate(X_test, y_test, verbose=0)
            return loss

        if st.button("🚀 Jalankan Tuning"):
            with st.spinner("🔍 Mencari kombinasi hyperparameter terbaik..."):
                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=n_trials)

                st.success("🎯 Tuning selesai!")
                st.write("Best loss:", study.best_value)
                st.json(study.best_params)

                # Simpan ke session state
                st.session_state.best_params_tcn = study.best_params
                best_params = study.best_params
                dilations = [2 ** i for i in range(best_params['dilations_level'])]

                # Final Model
                final_model = Sequential()
                final_model.add(TCN(
                    nb_filters=best_params['filters'],
                    kernel_size=best_params['kernel_size'],
                    dilations=dilations,
                    dropout_rate=best_params['dropout_rate'],
                    return_sequences=False,
                    input_shape=(X_train.shape[1], X_train.shape[2])
                ))
                final_model.add(Dropout(best_params['dropout_rate']))
                final_model.add(Dense(best_params['dense_units'], activation='relu'))
                final_model.add(Dense(n_features))

                optimizer = Adam(learning_rate=best_params['learning_rate'])
                final_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                history = final_model.fit(X_train, y_train,
                                          epochs=best_params['epochs'],
                                          batch_size=best_params['batch_size'],
                                          validation_data=(X_test, y_test),
                                          callbacks=[early_stopping],
                                          verbose=0, shuffle=False)

                test_loss, test_mae = final_model.evaluate(X_test, y_test, verbose=0)
                st.success(f"✅ **Test Loss:** {test_loss:.4f} | **Test MAE:** {test_mae:.4f}")

                # Simpan model ke session_state
                st.session_state['tcn_model'] = final_model

                # Plot loss
                st.subheader("📉 Grafik Loss Training vs Validation")
                fig, ax = plt.subplots()
                ax.plot(history.history['loss'], label='Training Loss')
                ax.plot(history.history['val_loss'], label='Validation Loss')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.set_title('Training vs Validation Loss')
                ax.legend()
                st.pyplot(fig)

                # ---------------- INVERSE TRANSFORM & PREDIKSI ---------------- #

                def inverse_transform_and_plot_tcn(y_true, y_pred, scaler, feature_name='FF_X'):
                    inv_true = scaler.inverse_transform(y_true)
                    inv_pred = scaler.inverse_transform(y_pred)

                    st.subheader(f"📊 Actual vs Predicted - {feature_name}")
                    fig, ax = plt.subplots(figsize=(14, 5))
                    ax.plot(inv_true, label='Actual')
                    ax.plot(inv_pred, label='Predicted')
                    ax.set_title(f'Actual vs Predicted: {feature_name}')
                    ax.set_xlabel('Time')
                    ax.set_ylabel(feature_name)
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

                    return inv_true, inv_pred

                # --- Buat DataFrame Prediksi vs Aktual ---
                def create_predictions_dataframe_tcn(y_true, y_pred, feature_name='Target'):
                    y_true_flat = y_true.flatten()
                    y_pred_flat = np.round(y_pred.flatten(), 3)

                    df = pd.DataFrame({
                        f'{feature_name}': y_true_flat,
                        f'{feature_name}_pred': y_pred_flat
                    })
                    return df

                def calculate_metrics_tcn(y_true, y_pred, feature_name='FF_X'):
                    y_true = y_true.flatten()
                    y_pred = y_pred.flatten()

                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    r2 = r2_score(y_true, y_pred)

                    mask = y_true != 0
                    if np.any(mask):
                        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                    else:
                        mape = np.nan

                    metrics = {
                        'Feature': [feature_name],
                        'MAE': [mae],
                        'RMSE': [rmse],
                        'R2': [r2],
                        'MAPE': [mape]
                    }

                    return pd.DataFrame(metrics)

                
                # Ambil variabel dari session state
                final_model = st .session_state['tcn_model']
                X_test = st.session_state['X_test']
                y_test = st.session_state['y_test']
                scaler = st.session_state['scaler']
                # 1. Prediksi
                y_pred_tcn = final_model.predict(X_test)

                # 2. Nama fitur target
                features = ['FF_X']  # sesuaikan jika berbeda
                st.session_state.features = ['FF_X']

                # 3. Inverse transform dan visualisasi
                y_test_inv, y_pred_inv = inverse_transform_and_plot_tcn(y_test, y_pred_tcn, scaler, features)
                st.session_state.y_test_inv = y_test_inv
                st.session_state.y_pred_inv = y_pred_inv

                # 4. Tabel hasil prediksi
                st.subheader("🧾 Tabel Hasil Prediksi (Model TCN)")
                preds_df = create_predictions_dataframe_tcn(y_test_inv, y_pred_inv, feature_name=features[0])
                st.dataframe(preds_df.head())

                # 5. Evaluasi metrik
                st.subheader("📊 Evaluasi Akurasi (Model TCN)")
                metrics_df = calculate_metrics_tcn(y_test_inv, y_pred_inv, feature_name=features[0])
                st.dataframe(metrics_df)
                st.session_state.preds_df = preds_df
                st.session_state.metrics_df = metrics_df
                model = st.session_state['tcn_model']
                scaler = st.session_state['scaler']
                X_test = st.session_state['X_test']
                df_musim = st.session_state['df_musim']
                df_train = st.session_state['df_train']
                df_test = st.session_state['df_test']

                # --- Parameter ---
                n_days = st.session_state['n_days']
                n_features = st.session_state['n_features']
                n_forecast_days = 30

                # --- Ambil sequence terakhir dari data test ---
                last_sequence = X_test[-1].reshape(1, n_days, n_features)

                # --- Prediksi iteratif 30 hari ---
                forecast = []
                current_seq = last_sequence
                for _ in range(n_forecast_days):
                    pred = model.predict(current_seq, verbose=0)
                    forecast.append(pred[0])

                    # Update sequence autoregresif
                    new_seq = np.append(current_seq[:, 1:, :], pred.reshape(1, 1, n_features), axis=1)
                    current_seq = new_seq

                # --- Inverse transform hasil prediksi ---
                forecast_array = np.array(forecast)
                forecast_inverse = scaler.inverse_transform(forecast_array)
                forecast_inverse = np.abs(forecast_inverse)

                # --- Buat dataframe hasil prediksi ---
                date_range = pd.date_range(start=df_musim.index[-1] + pd.Timedelta(days=1), periods=n_forecast_days)
                forecast_df = pd.DataFrame(forecast_inverse, index=date_range, columns=['FF_X'])
                st.session_state['forecast_df'] = forecast_df

                # --- Plot hasil peramalan ---
                feature = 'FF_X'
                fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

                fig.add_trace(go.Scatter(
                    x=df_train.index,
                    y=df_train[feature],
                    mode='lines',
                    name='Data Training',
                    line=dict(color='green')
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=df_test.index,
                    y=df_test[feature],
                    mode='lines',
                    name='Data Test',
                    line=dict(color='orange')
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df[feature],
                    mode='lines',
                    name='Peramalan TCN',
                    line=dict(color='blue')
                ), row=1, col=1)

                if not df_train.empty and not df_test.empty:
                    fig.add_trace(go.Scatter(
                        x=[df_train.index[-1], df_test.index[0]],
                        y=[df_train[feature].iloc[-1], df_test[feature].iloc[0]],
                        mode='lines',
                        line=dict(color='orange'),
                        showlegend=False
                    ), row=1, col=1)

                if not df_test.empty and not forecast_df.empty:
                    fig.add_trace(go.Scatter(
                        x=[df_test.index[-1], forecast_df.index[0]],
                        y=[df_test[feature].iloc[-1], forecast_df[feature].iloc[0]],
                        mode='lines',
                        line=dict(color='blue'),
                        showlegend=False
                    ), row=1, col=1)

                fig.update_layout(
                    title=f'Peramalan {feature} Menggunakan TCN untuk {n_forecast_days} Hari ke Depan',
                    yaxis_title=feature,
                    width=1100,
                    height=500,
                    margin=dict(l=50, r=30, t=60, b=40),
                    legend=dict(orientation="h", y=1.12, x=0.01)
                )

                st.plotly_chart(fig, use_container_width=True)

                # --- Tabel hasil ---
                st.subheader("🧾 Tabel Hasil Prediksi 30 Hari")
                st.dataframe(forecast_df)
                                # === Tombol ke tahap berikutnya ===
                if st.button("➡️ Evaluasi Model"):
                    st.session_state["menu"] = "Evaluasi Model"
                    st.rerun()

      
if menu == "Evaluasi Model":
    st.title("📊 Evaluasi & Peramalan Model TCN")
    # Fungsi untuk ubah ke supervised
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        df = pd.DataFrame(data)
        cols, names = [], []
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [f'var(t-{i})']
        for i in range(n_out):
            cols.append(df.shift(-i))
            names += ['var(t)'] if i == 0 else [f'var(t+{i})']
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    
    # Judul
    st.subheader("📦 Prediksi & Visualisasi Kecepatan Angin (Upload Model tcn Saja)")
    
    # --- Cek apakah df_musim sudah ada ---
    if 'df_musim' not in st.session_state or 'df_train' not in st.session_state or 'df_test' not in st.session_state:
        st.warning("Silakan lakukan preprocessing dan pembagian data terlebih dahulu.")
        st.stop()

    df_musim = st.session_state['df_musim']
    df_train = st.session_state['df_train']
    df_test = st.session_state['df_test']

    # --- Upload model ---
    st.subheader("📥 Upload Model TCN (.h5)")
    uploaded_model = st.file_uploader("Upload file model TCN (.h5)", type=["h5"])

    if uploaded_model is None:
        st.info("Silakan upload model TCN terlebih dahulu.")
        st.stop()

    # --- Simpan model sementara ---
    model_path = "temp_uploaded_model.h5"
    with open(model_path, "wb") as f:
        f.write(uploaded_model.read())

    try:
        loaded_model = load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    # --- Data uji ---
    test_data = df_musim[['FF_X']].values.astype('float32')

    # --- Scaling ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    test_data_scaled = scaler.fit_transform(test_data)

    # --- Parameter ---
    n_days = 6
    n_features = test_data.shape[1]
    n_forecast_days = 30

    # --- Supervised Transform ---
    test_data_supervised = series_to_supervised(test_data_scaled, n_days, 1)
    test_data_sequences = test_data_supervised.values[:, :n_days * n_features]

    # --- Ambil sequence terakhir untuk prediksi iteratif ---
    last_sequence = test_data_sequences[-1].reshape((1, n_days, n_features))

    # --- Prediksi iteratif 30 hari ---
    forecast = []
    for _ in range(n_forecast_days):
        predicted = loaded_model.predict(last_sequence)
        forecast.append(predicted[0])

        predicted_reshaped = predicted.reshape((1, 1, n_features))
        last_sequence = np.append(last_sequence[:, 1:, :], predicted_reshaped, axis=1)

    # --- Inverse transform hasil prediksi ---
    forecast_array = np.array(forecast)
    forecast_inverse = scaler.inverse_transform(forecast_array)
    forecast_inverse = np.abs(forecast_inverse)

    # --- Buat dataframe hasil prediksi ---
    date_range = pd.date_range(start=df_musim.index[-1], periods=n_forecast_days + 1)
    forecast_df = pd.DataFrame(forecast_inverse, index=date_range[1:], columns=['FF_X'])

    # --- Simpan hasil prediksi ke session_state (opsional) ---
    st.session_state['forecast_df'] = forecast_df

    # --- Plot Hasil ---
    features = ['FF_X']

    for feature in features:
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

        fig.add_trace(go.Scatter(
            x=df_train.index,
            y=df_train[feature],
            mode='lines',
            name='Data Training',
            line=dict(color='green')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_test.index,
            y=df_test[feature],
            mode='lines',
            name='Data Test',
            line=dict(color='orange')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df[feature],
            mode='lines',
            name='Peramalan TCN',
            line=dict(color='blue')
        ), row=1, col=1)

        if not df_train.empty and not df_test.empty:
            fig.add_trace(go.Scatter(
                x=[df_train.index[-1], df_test.index[0]],
                y=[df_train[feature].iloc[-1], df_test[feature].iloc[0]],
                mode='lines',
                line=dict(color='orange'),
                showlegend=False
            ), row=1, col=1)

        if not df_test.empty and not forecast_df.empty:
            fig.add_trace(go.Scatter(
                x=[df_test.index[-1], forecast_df.index[0]],
                y=[df_test[feature].iloc[-1], forecast_df[feature].iloc[0]],
                mode='lines',
                line=dict(color='blue'),
                showlegend=False
            ), row=1, col=1)

        fig.update_layout(
            title=f'Peramalan {feature} Menggunakan TCN untuk {n_forecast_days} Hari ke Depan',
            yaxis_title=feature,
            width=1100,
            height=500,
            margin=dict(l=50, r=30, t=60, b=40),
            legend=dict(orientation="h", y=1.12, x=0.01)
        )

        st.plotly_chart(fig, use_container_width=True)

    # --- Tampilkan tabel hasil prediksi ---
    st.subheader("Hasil Peramalan TCN 30 Hari")
    st.dataframe(forecast_df)

