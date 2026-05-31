import requests
import pandas as pd
import re
import os
import pickle

# MACHINE LEARNING
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# VISUALISASI
import matplotlib.pyplot as plt

API_KEY     = "b552bb321d8749268adabdcd689683ac"
STATIC_FILE = "dataset/DataHoaxIndonesia_Cleaned.csv"

# =========================
# CLEANING
# =========================

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def limit_text(text, max_len=300):
    return str(text)[:max_len]

# =========================
# LOAD DATASET
# =========================

def load_static_data():
    if not os.path.exists(STATIC_FILE):
        print("⚠️ Dataset tidak ditemukan, memakai dummy")
        return pd.DataFrame({
            "text":   ["contoh berita hoaks", "contoh berita valid"],
            "label":  [1, 0],
            "source": ["dummy", "dummy"],
            "type":   ["static", "static"]
        })

    df = pd.read_csv(STATIC_FILE)
    print("Kolom dataset:", df.columns.tolist())

    if "Label" in df.columns:
        df.rename(columns={"Label": "label"}, inplace=True)

    if "Headline" in df.columns and "Body" in df.columns:
        df["text"] = df["Headline"].fillna('') + " " + df["Body"].fillna('')
    elif "text" not in df.columns:
        raise Exception("❌ Dataset tidak memiliki kolom text")

    df = df[["text", "label"]]
    df["source"] = "dataset"
    df["type"]   = "static"
    return df

# =========================
# REALTIME NEWS
# =========================

def get_realtime_data():
    realtime_data = []
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q=indonesia&language=id&pageSize=20&"
        f"sortBy=publishedAt&apiKey={API_KEY}"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get("status") == "ok":
            for art in data.get("articles", []):
                text = (
                    f"{art.get('title', '')} "
                    f"{art.get('description', '')}"
                ).strip()
                if text:
                    realtime_data.append({
                        "text":   text,
                        "label":  0,
                        "source": "newsapi",
                        "type":   "realtime"
                    })
        else:
            print("⚠️ API ERROR:", data)
    except Exception as e:
        print("⚠️ ERROR API:", e)

    if not realtime_data:
        realtime_data.append({
            "text":   "contoh berita realtime indonesia",
            "label":  0,
            "source": "dummy",
            "type":   "realtime"
        })

    return pd.DataFrame(realtime_data)

# =========================
# MAIN PIPELINE
# =========================

def run_pipeline():
    print("🚀 Menjalankan Pipeline")

    # Load data
    df_static   = load_static_data()
    df_realtime = get_realtime_data()

    # Cleaning
    for df in [df_static, df_realtime]:
        df["clean_text"] = df["text"].apply(clean_text)
        df["text"]       = df["text"].apply(limit_text)

    # Hapus duplikat
    df_static.drop_duplicates(subset="clean_text", inplace=True)

    # Merge
    df_final = pd.concat([df_static, df_realtime], ignore_index=True)
    df_final["label_name"] = df_final["label"].apply(
        lambda x: "Hoaks" if x == 1 else "Valid"
    )
    print("Jumlah Data:", len(df_final))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df_final["clean_text"],
        df_final["label"],
        test_size=0.2,
        random_state=42
    )

    # TF-IDF
    tfidf         = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    # ── Naive Bayes ──
    nb       = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    pred_nb  = nb.predict(X_test_tfidf)

    acc_nb   = accuracy_score(y_test, pred_nb)
    prec_nb  = precision_score(y_test, pred_nb,  average="weighted", zero_division=0)
    rec_nb   = recall_score(y_test, pred_nb,     average="weighted", zero_division=0)
    f1_nb    = f1_score(y_test, pred_nb,         average="weighted", zero_division=0)
    cm_nb    = confusion_matrix(y_test, pred_nb).tolist()   # list agar JSON-safe

    # ── SVM ──
    svm      = LinearSVC(max_iter=5000)
    svm.fit(X_train_tfidf, y_train)
    pred_svm = svm.predict(X_test_tfidf)

    acc_svm  = accuracy_score(y_test, pred_svm)
    prec_svm = precision_score(y_test, pred_svm, average="weighted", zero_division=0)
    rec_svm  = recall_score(y_test, pred_svm,    average="weighted", zero_division=0)
    f1_svm   = f1_score(y_test, pred_svm,        average="weighted", zero_division=0)
    cm_svm   = confusion_matrix(y_test, pred_svm).tolist()

    print(f"NB  — Acc:{acc_nb:.4f} Prec:{prec_nb:.4f} Rec:{rec_nb:.4f} F1:{f1_nb:.4f}")
    print(f"SVM — Acc:{acc_svm:.4f} Prec:{prec_svm:.4f} Rec:{rec_svm:.4f} F1:{f1_svm:.4f}")

    # Save model (simpan NB sebagai default checker)
    os.makedirs("model", exist_ok=True)
    with open("model/model.pkl", "wb") as f:
        pickle.dump(nb, f)
    with open("model/vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    print("MODEL BERHASIL DISIMPAN")

    # Visualisasi (opsional, untuk static folder)
    os.makedirs("static", exist_ok=True)

    plt.figure()
    df_final["label_name"].value_counts().plot(kind='bar')
    plt.title("Distribusi Data")
    plt.savefig("static/chart1.png")
    plt.close()

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, pred_svm))
    disp.plot()
    plt.savefig("static/chart2.png")
    plt.close()

    plt.figure()
    plt.bar(["Naive Bayes", "SVM"], [acc_nb, acc_svm])
    plt.title("Perbandingan Akurasi")
    plt.savefig("static/chart3.png")
    plt.close()

    # Siapkan return
    label_counts    = df_final["label_name"].value_counts().to_dict()
    data_sample     = df_static[["text", "label", "source"]].head(50).to_dict(orient="records")
    realtime_sample = df_realtime[["text", "label", "source"]].head(50).to_dict(orient="records")

    return {
        # ── Accuracy ──
        "acc_nb":   round(float(acc_nb),  4),
        "acc_svm":  round(float(acc_svm), 4),

        # ── Naive Bayes metrik ──
        "prec_nb":  round(float(prec_nb), 4),
        "rec_nb":   round(float(rec_nb),  4),
        "f1_nb":    round(float(f1_nb),   4),
        "cm_nb":    cm_nb,

        # ── SVM metrik ──
        "prec_svm": round(float(prec_svm), 4),
        "rec_svm":  round(float(rec_svm),  4),
        "f1_svm":   round(float(f1_svm),   4),
        "cm_svm":   cm_svm,

        # ── Data lain ──
        "label_counts":  label_counts,
        "data":          data_sample,
        "realtime_news": realtime_sample,
    }

# =========================
# RUN MANUAL
# =========================

if __name__ == "__main__":
    result = run_pipeline()
    print("✅ SELESAI")
    print(result)