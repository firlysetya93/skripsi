# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
        # (Sama seperti sebelumnya...)
        # ACF, PACF, decomposition, ADF sudah ditambahkan
        pass
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
# ========== MODELING ==========
elif menu == "🧠 Modeling (LSTM / TCN / RBFNN)":
    st.title("🧠 Transformasi Supervised Learning (Lag Feature)")

    if 'df_musim' in st.session_state:
        df_musim = st.session_state.df_musim

        # Menampilkan nilai min dan max
        st.subheader("📏 Statistik Kolom FF_X")
        min_val = df_musim['FF_X'].min()
        max_val = df_musim['FF_X'].max()
        st.write(f"**Nilai Minimum FF_X:** {min_val}")
        st.write(f"**Nilai Maksimum FF_X:** {max_val}")

        # Normalisasi
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(df_musim[['FF_X']])
        st.success("✅ Data telah dinormalisasi menggunakan MinMaxScaler.")

        # Fungsi konversi ke supervised
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

        # Input lag dari user
        n_days = st.slider("🔢 Pilih jumlah lag (hari)", 1, 30, 6)
        n_features = 1

        reframed = series_to_supervised(scaled, n_days, 1)

        st.subheader("🧾 Data Setelah Diubah ke Supervised Format")
        st.write(f"Shape: {reframed.shape}")
        st.dataframe(reframed.head(10))

        # Simpan ke session state jika ingin digunakan selanjutnya
        st.session_state.reframed = reframed
    else:
        st.warning("❗ Silakan lakukan preprocessing terlebih dahulu di menu '⚙️ Preprocessing'.")
        # ------------------------- TRAIN TEST SPLIT -------------------------
        st.subheader("🧪 Split Data untuk LSTM")

        # Ambil nilai array dari dataframe hasil reframing
        values = reframed.values

        # Simpan index tanggal dari hasil reframing
        date_reframed = df_musim.index[reframed.index]

        # Split manual (tanpa shuffle)
        train_size = int(len(values) * 0.8)
        train, test = values[:train_size], values[train_size:]

        # Bagi juga tanggal
        date_train = date_reframed[:len(train)]
        date_test = date_reframed[len(train):]

        st.write(f"Jumlah data: {len(values)}")
        st.write(f"Jumlah train: {len(train)} ({date_train.min().date()} s.d. {date_train.max().date()})")
        st.write(f"Jumlah test : {len(test)} ({date_test.min().date()} s.d. {date_test.max().date()})")

        # ------------------------- PISAHKAN X dan y -------------------------
        n_obs = n_days * n_features
        train_X, train_y = train[:, :n_obs], train[:, -1]
        test_X, test_y = test[:, :n_obs], test[:, -1]

        # Reshape untuk LSTM
        X_train = train_X.reshape((train_X.shape[0], n_days, n_features))
        X_test = test_X.reshape((test_X.shape[0], n_days, n_features))
        y_train = train_y.reshape(-1, 1)
        y_test = test_y.reshape(-1, 1)

        st.write("📐 Shape X_train:", X_train.shape)
        st.write("📐 Shape y_train:", y_train.shape)
        st.write("📐 Shape X_test :", X_test.shape)
        st.write("📐 Shape y_test :", y_test.shape)

        # ------------------------- MODEL LSTM -------------------------
        st.subheader("🧠 Arsitektur Model LSTM")

        model1 = Sequential()
        model1.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model1.add(Dropout(0.3))
        model1.add(LSTM(10, return_sequences=False))
        model1.add(Dropout(0.3))
        model1.add(Flatten())
        model1.add(Dense(64, activation="relu"))
        model1.add(Dense(16, activation="relu"))
        model1.add(Dense(n_features))

        optimizer = Adam(learning_rate=0.001)
        model1.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        # Tampilkan summary
        with st.expander("📋 LSTM Model Summary"):
            st.text(model1.summary())
            # ------------------------- TRAINING DENGAN CALLBACK -------------------------
        st.subheader("🏋️ Training dengan Scheduler & Checkpoint")

        def train_model_streamlit(model, X_train, y_train, X_test, y_test,
                                  learning_rate=0.001, batch_size=32, epochs=50,
                                  patience=5, filepath='best_model.h5'):
            # Scheduler untuk learning rate
            def scheduler(epoch, lr):
                if epoch < 10:
                    return lr
                else:
                    return float(lr * tf.math.exp(-0.1 * (epoch - 9)))

            lr_scheduler = LearningRateScheduler(scheduler)
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)

            # Training
            with st.spinner("⏳ Model sedang dilatih..."):
                history = model.fit(X_train, y_train,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_data=(X_test, y_test),
                                    callbacks=[lr_scheduler, early_stopping, checkpointer],
                                    verbose=0,
                                    shuffle=False)

            st.success("✅ Training selesai!")

            # Evaluasi
            loss, mae = model.evaluate(X_test, y_test, verbose=0)
            st.metric("MSE (Test Loss)", f"{loss:.5f}")
            st.metric("MAE (Test MAE)", f"{mae:.5f}")

            # Plot history
            fig, ax = plt.subplots()
            ax.plot(history.history['loss'], label='Training Loss')
            ax.plot(history.history['val_loss'], label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training History')
            ax.legend()
            st.pyplot(fig)

            return history, loss

        # Tombol untuk mulai training
        if st.button("🚀 Mulai Training (dengan Scheduler & Checkpoint)"):
            history1, test_loss1 = train_model_streamlit(model1, X_train, y_train, X_test, y_test)
            st.session_state.history1 = history1



