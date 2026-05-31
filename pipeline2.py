# =========================================
# IMPORT LIBRARY
# =========================================
import requests
import pandas as pd
import re
from datetime import datetime
import os

# MACHINE LEARNING
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# VISUALISASI
import matplotlib.pyplot as plt

# =========================================
# CONFIG
# =========================================
API_KEY = "b552bb321d8749268adabdcd689683ac"
STATIC_FILE = "dataset/DataHoaxIndonesia_Cleaned.csv"

# =========================================
# SCRIPT AUTOMASI PIPELINE
# =========================================
print("\n=== MENJALANKAN PIPELINE OTOMATIS ===")


# =========================================
# FUNCTION CLEANING
# =========================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def limit_text(text, max_len=30000):
    return text[:max_len]

# =========================================
# 1. EXTRACT REALTIME
# =========================================
print("Mengambil data realtime...")

realtime_data = []
query = "indonesia"

url = (
    f"https://newsapi.org/v2/everything?"
    f"q={query}&language=id&pageSize=50&"
    f"sortBy=publishedAt&apiKey={API_KEY}"
)

try:
    response = requests.get(url, timeout=10)
    data = response.json()

    if data.get("status") == "ok":
        for art in data.get("articles", []):
            title = art.get("title") or ""
            desc = art.get("description") or ""

            text = f"{title} {desc}".strip()

            if text != "":
                realtime_data.append({
                    "text": text,
                    "label": 0,
                    "source": art.get("source", {}).get("name", "newsapi"),
                    "type": "realtime"
                })
except Exception as e:
    print("Request error:", e)

df_realtime = pd.DataFrame(realtime_data)

if df_realtime.empty:
    df_realtime = pd.DataFrame([{
        "text": "Contoh berita terbaru Indonesia",
        "label": 0,
        "source": "dummy",
        "type": "realtime"
    }])

print("Realtime:", len(df_realtime))

# =========================================
# 2. DATA STATIK
# =========================================
print("Membaca data statik...")

df_static = pd.read_csv(STATIC_FILE)

if "Label" in df_static.columns:
    df_static.rename(columns={"Label": "label"}, inplace=True)

df_static["text"] = df_static["Headline"].fillna('') + " " + df_static["Body"].fillna('')
df_static = df_static[["text", "label"]]
df_static["type"] = "static"
df_static["source"] = "dataset"

print("Static:", len(df_static))

# =========================================
# 3. PREPROCESSING
# =========================================
print("Preprocessing...")

for df in [df_realtime, df_static]:
    df["clean_text"] = df["text"].apply(clean_text)
    df["text"] = df["text"].apply(limit_text)

df_realtime = df_realtime[df_realtime["clean_text"].str.strip() != ""]
df_static = df_static[df_static["clean_text"].str.strip() != ""]

df_static.drop_duplicates(subset="clean_text", inplace=True)

# =========================================
# 4. INTEGRASI
# =========================================
print("Menggabungkan data...")

df_final = pd.concat([df_static, df_realtime], ignore_index=True)

df_final["label_name"] = df_final["label"].apply(
    lambda x: "hoaks" if x == 1 else "tidak hoaks"
)

# =========================================
# 5. MACHINE LEARNING
# =========================================
print("\n=== PROSES MACHINE LEARNING ===")

X_train, X_test, y_train, y_test = train_test_split(
    df_final["clean_text"],
    df_final["label"],
    test_size=0.2,
    random_state=42
)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)

acc_nb = accuracy_score(y_test, y_pred_nb)
print("\nNaive Bayes Accuracy:", acc_nb)
print(classification_report(y_test, y_pred_nb))

# SVM
svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)

acc_svm = accuracy_score(y_test, y_pred_svm)
print("\nSVM Accuracy:", acc_svm)
print(classification_report(y_test, y_pred_svm))

# =========================================
# 6. VISUALISASI
# =========================================

# 🔹 1. Distribusi Data
plt.figure()
df_final["label_name"].value_counts().plot(kind='bar')
plt.title("Distribusi Data Hoaks vs Tidak Hoaks")
plt.xlabel("Kelas")
plt.ylabel("Jumlah")
plt.show()

# 🔹 2. Confusion Matrix SVM
cm = confusion_matrix(y_test, y_pred_svm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix SVM")
plt.show()

# 🔹 3. Perbandingan Akurasi
plt.figure()
model_names = ["Naive Bayes", "SVM"]
accuracy_values = [acc_nb, acc_svm]

plt.bar(model_names, accuracy_values)
plt.title("Perbandingan Akurasi Model")
plt.ylabel("Akurasi")
plt.show()

# =========================================
# 7. SAVE EXCEL (LENGKAP)
# =========================================
print("Menyimpan ke Excel...")

OUTPUT_FOLDER = "hasil_load_dataset"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

file_path = os.path.join(
    OUTPUT_FOLDER,
    f"dataset_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
)

# 🔹 SUMMARY
summary = df_final["type"].value_counts().reset_index()
summary.columns = ["type", "jumlah"]

with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
    df_static.to_excel(writer, sheet_name="static", index=False)
    df_realtime.to_excel(writer, sheet_name="realtime", index=False)
    df_final.to_excel(writer, sheet_name="gabungan", index=False)
    summary.to_excel(writer, sheet_name="summary", index=False)

print("✅ Berhasil:", file_path)

print("\n=== PIPELINE SELESAI DIJALANKAN OTOMATIS ===")