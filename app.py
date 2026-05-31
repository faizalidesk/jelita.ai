from flask import Flask, render_template, request, redirect, url_for, session
from pipeline import run_pipeline
from newspaper import Article
from bs4 import BeautifulSoup
from functools import wraps
import requests
import pickle
import pandas as pd
import numpy as np
import json
import os

app = Flask(__name__)
app.secret_key = "jelita_secret_key_2025"

# =========================
# CONSTANTS
# =========================

MODEL_PATH      = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"
DATASET_PATH    = "dataset/DataHoaxIndonesia_Cleaned.csv"

VALID_ACCOUNTS = {
    "admin":    "admin123",
    "mahasiswa": "hoax2025",
}

# =========================
# GLOBAL VARS
# =========================

hasil_model = {}
model       = None
vectorizer  = None

# =========================
# LOAD MODEL & VECTORIZER
# =========================

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model berhasil dimuat")
except Exception as e:
    print("Gagal load model:", e)

try:
    if os.path.exists(VECTORIZER_PATH):
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        print("Vectorizer berhasil dimuat")
except Exception as e:
    print("Gagal load vectorizer:", e)

# =========================
# HELPER: LOGIN REQUIRED
# =========================

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# =========================
# HELPER: CONVERT NUMPY
# =========================

def convert_numpy(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, list): return [convert_numpy(i) for i in obj]
    if isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
    return obj

# =========================
# LOGIN
# =========================

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username in VALID_ACCOUNTS and VALID_ACCOUNTS[username] == password:
            session["logged_in"] = True
            session["username"]  = username
            return redirect(url_for("index"))
        else:
            error = "Username atau password salah."
    return render_template("login.html", error=error)

# =========================
# LOGOUT
# =========================

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# =========================
# DASHBOARD
# =========================

@app.route("/")
@login_required
def index():
    hasil = {}
    if os.path.exists("hasil_cache.json"):
        with open("hasil_cache.json", "r") as f:
            hasil = json.load(f)

    dataset_stats = {"total": 0, "hoaks": 0, "valid": 0, "rate": 0}
    dataset_preview = []  # ← tambahkan ini

    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        if "Label" in df.columns:
            df.rename(columns={"Label": "label"}, inplace=True)
        if "label" in df.columns:
            total   = len(df)
            n_hoax  = int((df["label"] == 1).sum())
            n_valid = int((df["label"] == 0).sum())
            rate    = round(n_hoax / total * 100, 1) if total else 0
            dataset_stats = {"total": total, "hoaks": n_hoax, "valid": n_valid, "rate": rate}

            # Ambil 10 baris pertama untuk preview ← tambahkan ini
            preview_cols = [c for c in df.columns if c in ("Headline", "Body", "text", "label")]
            df_preview = df[preview_cols].head(10)
            dataset_preview = df_preview.to_dict(orient="records")

    return render_template("index.html", hasil=hasil, stats=dataset_stats, preview=dataset_preview)

# =========================
# RUN PIPELINE
# =========================

@app.route("/run")
@login_required
def run():
    global hasil_model, model, vectorizer

    hasil_model = run_pipeline()
    if hasil_model is None:
        hasil_model = {}

    hasil_model.setdefault("acc_nb",       0)
    hasil_model.setdefault("acc_svm",      0)
    hasil_model.setdefault("prec_nb",      0)
    hasil_model.setdefault("prec_svm",     0)
    hasil_model.setdefault("rec_nb",       0)
    hasil_model.setdefault("rec_svm",      0)
    hasil_model.setdefault("f1_nb",        0)
    hasil_model.setdefault("f1_svm",       0)
    hasil_model.setdefault("cm_nb",        [[0,0],[0,0]])
    hasil_model.setdefault("cm_svm",       [[0,0],[0,0]])
    hasil_model.setdefault("label_counts", {})
    hasil_model.setdefault("data",         [])
    hasil_model.setdefault("realtime_news",[])

    # Simpan ke cache (tanpa data besar)
    cache = {k: v for k, v in hasil_model.items() if k not in ("data", "realtime_news")}
    cache_clean = convert_numpy(cache)
    with open("hasil_cache.json", "w") as f:
        json.dump(cache_clean, f)

    # Reload model setelah training
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    if os.path.exists(VECTORIZER_PATH):
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)

    return redirect(url_for("index"))

# =========================
# CHECKER
# =========================

@app.route("/checker")
@login_required
def checker():
    return render_template("checker.html")

# =========================
# PREDICT
# =========================

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    global model, vectorizer

    text = request.form.get("text")
    link = request.form.get("link")
    final_text = ""

    if link and link.strip():
        try:
            try:
                article = Article(link)
                article.download()
                article.parse()
                final_text = article.text
            except:
                headers  = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(link, headers=headers, timeout=15)
                soup     = BeautifulSoup(response.text, "html.parser")
                final_text = " ".join([p.get_text(strip=True) for p in soup.find_all("p")])

            if not final_text.strip():
                return render_template("checker.html", error="Isi berita tidak dapat diambil.")
        except Exception as e:
            return render_template("checker.html", error=f"Gagal mengambil berita: {e}")

    elif text and text.strip():
        final_text = text
    else:
        return render_template("checker.html", error="Masukkan link atau teks berita.")

    if model is None or vectorizer is None:
        return render_template("checker.html", error="Model belum tersedia. Jalankan /run terlebih dahulu.")

    try:
        text_vector = vectorizer.transform([final_text])
        prediction  = model.predict(text_vector)[0]
        label       = "HOAKS" if prediction == 1 else "VALID"
        confidence  = 0
        if hasattr(model, "predict_proba"):
            prob       = model.predict_proba(text_vector)[0]
            confidence = round(max(prob) * 100, 2)
        return render_template("checker.html", result=label, confidence=confidence, news_text=final_text[:3000])
    except Exception as e:
        return render_template("checker.html", error=f"Terjadi error prediksi: {e}")

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    app.run(debug=True)