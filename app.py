import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import gdown
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import re

# ---------------- Session State ----------------
if "users" not in st.session_state:
    st.session_state["users"] = {
        "admin@gmail.com": {"password": "admin123", "role": "Admin"},
        "user1@gmail.com": {"password": "user123", "role": "User"},
    }

if "logged_in" not in st.session_state:
    st.session_state.update({"logged_in": False, "role": None, "email": None})

if "login_history" not in st.session_state:
    st.session_state["login_history"] = []

if "model_status" not in st.session_state:
    st.session_state["model_status"] = {"SVM": True, "MLP": True, "LSTM": True}

# ---------------- Email Validation ----------------
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

# ---------------- Download & Load Models ----------------
# LSTM (.h5)
lstm_url = "https://drive.google.com/uc?export=download&id=15vtdUpzVW21yvGizNDTfpvp3zfAZzq8T"
lstm_path = "lstm_stock.h5"
if not os.path.exists(lstm_path):
    gdown.download(lstm_url, lstm_path, quiet=False)
lstm_model = tf.keras.models.load_model(lstm_path, compile=False)

# MLP
mlp_url = "https://drive.google.com/uc?export=download&id=12FtUiL_PKXfo1Z6Nv7adds3NOta_NICr"
mlp_path = "mlp_model.pkl"
if not os.path.exists(mlp_path):
    gdown.download(mlp_url, mlp_path, quiet=False)
mlp_model = joblib.load(mlp_path)

# SVM
svm_url = "https://drive.google.com/uc?export=download&id=1bOhNKntdNX5xEv5kv33QQKrbdiDSaiI7"
svm_path = "svm_model.pkl"
if not os.path.exists(svm_path):
    gdown.download(svm_url, svm_path, quiet=False)
svm_model = joblib.load(svm_path)

# ---------------- LOGIN ----------------
if not st.session_state["logged_in"]:
    st.subheader("🔑 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if email in st.session_state["users"] and st.session_state["users"][email]["password"] == password:
            st.session_state.update({
                "logged_in": True,
                "role": st.session_state["users"][email]["role"],
                "email": email
            })
            st.session_state["login_history"].append({
                "Email": email,
                "Role": st.session_state["role"],
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success(f"✅ Welcome {st.session_state['role']} - {email}")
            st.experimental_rerun()
        else:
            st.error("❌ Invalid email or password")

# ---------------- DASHBOARDS ----------------
else:
    st.sidebar.header(f"{st.session_state['role']} Dashboard")
    if st.sidebar.button("Logout"):
        st.session_state.update({"logged_in": False, "role": None, "email": None})
        st.experimental_rerun()

    # ---------------- ADMIN ----------------
    if st.session_state["role"] == "Admin":
        st.title("📊 Admin Dashboard")

        # Enable/Disable models
        st.subheader("✅ Enable / Disable Models")
        for model in st.session_state["model_status"]:
            st.session_state["model_status"][model] = st.checkbox(
                model, value=st.session_state["model_status"][model]
            )

        # Register new users
        st.subheader("👥 Register New User")
        new_email = st.text_input("User Email", key="reg_email")
        new_pass = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Register User"):
            if not is_valid_email(new_email):
                st.error("Invalid email")
            elif new_email in st.session_state["users"]:
                st.error("User already exists")
            elif len(new_pass) < 5:
                st.error("Password too short")
            else:
                st.session_state["users"][new_email] = {"password": new_pass, "role": "User"}
                st.success(f"User {new_email} registered successfully")

        # Registered users table
        st.subheader("📜 Registered Users")
        users_df = pd.DataFrame([{"Email": e, "Role": v["role"]} for e, v in st.session_state["users"].items()])
        st.table(users_df)

        # Login history
        st.subheader("📊 Login History")
        if st.session_state["login_history"]:
            st.dataframe(pd.DataFrame(st.session_state["login_history"]))
        else:
            st.info("No login history")

    # ---------------- USER ----------------
    st.title(f"📈 Stock Prediction Dashboard ({st.session_state['role']})")
    uploaded_file = st.file_uploader("Upload CSV (must contain 'close')", type="csv")
    enabled_models = [m for m, status in st.session_state["model_status"].items() if status]

    if uploaded_file is not None and enabled_models:
        df = pd.read_csv(uploaded_file)
        if "close" not in df.columns:
            st.error("CSV must contain 'close'")
        else:
            st.dataframe(df.head())
            model_choice = st.radio("Select model", enabled_models)

            combined = None
            # ---------------- LSTM Prediction ----------------
            if model_choice == "LSTM":
                seq_len = 60
                scaler = MinMaxScaler()
                data_scaled = scaler.fit_transform(df[["close"]])
                if len(data_scaled) <= seq_len:
                    st.warning("Not enough data for LSTM (need >60 rows).")
                else:
                    n_features = lstm_model.input_shape[-1]
                    X_seq = [np.repeat(data_scaled[i:i+seq_len], n_features, axis=1)
                             if data_scaled[i:i+seq_len].shape[1] < n_features
                             else data_scaled[i:i+seq_len]
                             for i in range(len(data_scaled)-seq_len)]
                    X_seq = np.array(X_seq)
                    try:
                        pred_scaled = lstm_model.predict(X_seq, verbose=0)
                        pred = scaler.inverse_transform(pred_scaled[:, :1])
                        y_true = df["close"].values[seq_len:]
                        combined = pd.DataFrame({"Actual": y_true.flatten(), "Predicted": pred.flatten()})
                        st.line_chart(combined)
                        st.write(f"MSE: {mean_squared_error(y_true, pred):.4f}")
                        st.write(f"MAE: {mean_absolute_error(y_true, pred):.4f}")
                    except ValueError as e:
                        st.error(f"LSTM prediction error: {e}")

            # ---------------- MLP Prediction ----------------
            elif model_choice == "MLP":
                window = 5
                y_true = df["close"].values
                X = np.array([y_true[i:i+window] for i in range(len(y_true)-window)])
                pred = mlp_model.predict(X)
                y_true_aligned = y_true[window:]
                combined = pd.DataFrame({"Actual": y_true_aligned, "Predicted": pred.flatten()})
                st.line_chart(combined)
                st.write(f"MSE: {mean_squared_error(y_true_aligned, pred):.4f}")
                st.write(f"MAE: {mean_absolute_error(y_true_aligned, pred):.4f}")

            # ---------------- SVM Prediction ----------------
            elif model_choice == "SVM":
                window = 5
                y_true = df["close"].values
                scaler = StandardScaler()
                y_scaled = scaler.fit_transform(y_true.reshape(-1, 1)).flatten()
                X = np.array([y_scaled[i:i+window] for i in range(len(y_scaled)-window)])
                pred = svm_model.predict(X)
                pred_dir = ["Down" if p == 0 else "Up" for p in pred]
                st.write(pred_dir)
                st.line_chart([0 if p == "Down" else 1 for p in pred_dir])

            # ---------------- Download Predictions ----------------
            if combined is not None:
                csv = combined.to_csv(index=False)
                st.download_button(
                    "📥 Download Predictions",
                    data=csv,
                    file_name=f"{model_choice}_predictions.csv",
                    mime="text/csv"
                )
    else:
        if not enabled_models:
            st.warning("No models enabled")
