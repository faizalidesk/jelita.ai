import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import re

# ==================================
# CONFIG
# ==================================

st.set_page_config(
    page_title="Jelita - Jendela literasi dan validasi berita",
    page_icon="static/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================
# CONSTANTS
# ==================================

MODEL_PATH      = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"
DATASET_PATH    = "dataset/DataHoaxIndonesia_Cleaned.csv"

# Akun login sederhana (username: password)
VALID_ACCOUNTS = {
    "admin":    "admin123",
    "mahasiswa": "hoax2025",
}

# ==================================
# CSS — LIGHT THEME
# ==================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&family=Outfit:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── App background ── */
.stApp {
    background: #f0f4ff;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1.5px solid #e2e8f0 !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.05) !important;
}


/* Sembunyikan tombol collapse sidebar */
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

[data-testid="stSidebar"] * { color: #475569 !important; }
[data-testid="stSidebarNav"] { display: none; }



/* ── Block container ── */
.block-container {
    padding: 1.5rem 2.5rem;
    max-width: 1400px;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
    color: #1e293b !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    font-family: 'Nunito', sans-serif !important;
    transition: all 0.25s ease !important;
    width: 100%;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 14px rgba(37,99,235,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(37,99,235,0.35) !important;
}

/* ── Text area & input ── */
textarea, input[type="text"], input[type="password"] {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    color: #1e293b !important;
    font-family: 'Nunito', sans-serif !important;
}
textarea:focus, input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    color: #64748b !important;
    font-weight: 700 !important;
    font-family: 'Nunito', sans-serif !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2563eb !important;
    border-bottom-color: #2563eb !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1.5px solid #e2e8f0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 12px !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
}

/* ── Radio nav ── */
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] > div {
    gap: 4px !important;
    flex-direction: column !important;
}
div[data-testid="stRadio"] > div > label {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    color: #64748b !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
div[data-testid="stRadio"] > div > label:hover {
    background: #eff6ff !important;
    color: #2563eb !important;
}
div[data-testid="stRadio"] > div > label[data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #1e293b !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: #f1f5f9 !important;
    color: #2563eb !important;
    border: 1.5px solid #bfdbfe !important;
    box-shadow: none !important;
}

</style>
""", unsafe_allow_html=True)

# ==================================
# SESSION STATE
# ==================================

for key, val in {
    "page": "Dashboard",
    "pipeline_result": None,
    "logged_in": False,
    "login_user": "",
    "uploaded_df": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==================================
# HELPERS
# ==================================

@st.cache_resource
def load_model():
    model, vectorizer = None, None
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
    except Exception:
        pass
    try:
        if os.path.exists(VECTORIZER_PATH):
            with open(VECTORIZER_PATH, "rb") as f:
                vectorizer = pickle.load(f)
    except Exception:
        pass
    return model, vectorizer

@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path)
    if "Label" in df.columns:
        df.rename(columns={"Label": "label"}, inplace=True)
    if "Headline" in df.columns and "Body" in df.columns:
        df["text"] = df["Headline"].fillna("") + " " + df["Body"].fillna("")
    if "label" in df.columns:
        df["label_name"] = df["label"].map({0: "✅ Valid", 1: "⚠️ Hoaks"})
    return df

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def scrape_url(url: str) -> str:
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        if article.text.strip():
            return article.text
    except Exception:
        pass
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        return " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
    except Exception:
        return ""

def plotly_light_layout(fig, title="", height=360):
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color="#1e293b", size=14, family="Nunito"),
        ),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748b", family="Nunito"),
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        height=height,
        xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0"),
        legend=dict(font=dict(color="#475569")),
    )
    return fig

def card(content_html, padding="20px 24px", bg="#ffffff", border="#e2e8f0", radius="16px", shadow=True):
    sh = "0 2px 10px rgba(0,0,0,0.07)" if shadow else "none"
    return f"""
    <div style="
        background:{bg};border:1.5px solid {border};
        border-radius:{radius};padding:{padding};
        box-shadow:{sh};margin-bottom:16px;
    ">{content_html}</div>
    """

# ==================================
# LOGIN PAGE
# ==================================


import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
def show_login():
    col_l, col_c, col_r = st.columns([1, 1.2, 1])

    with col_c:
        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center;margin-bottom:32px;">
            <div style="
                font-family:'Outfit',sans-serif;
                font-size:26px;
                font-weight:800;
                color:#1e293b;
                margin-bottom:0px;
            ">
                Jelita Analytics AI
            </div>
            <div style="
                color:#64748b;
                font-size:14px;
            ">
                Jendela literasi dan validasi berita Artificial Intelligence
            </div>
        </div>
        """, unsafe_allow_html=True)


        st.markdown("""
            <div style="display:flex; justify-content:center; margin-bottom:15px; margin-top:-20px;">
                <img src="data:image/png;base64,{}" width="800px">
            </div>
            """.format(get_base64_image("static/Asset 1.png")), unsafe_allow_html=True)

        

        with st.form("login_form"):
            username = st.text_input(
                "👤 Username",
                placeholder="Masukkan username"
            )

            password = st.text_input(
                "🔒 Password",
                placeholder="Masukkan password",
                type="password"
            )

            submitted = st.form_submit_button("🚀 Masuk")

        if submitted:
            if username in VALID_ACCOUNTS and VALID_ACCOUNTS[username] == password:
                st.session_state.logged_in = True
                st.session_state.login_user = username
                st.rerun()
            else:
                st.error("❌ Username atau password salah.")

        st.markdown("""
        <div style="
            background:#eff6ff;
            border:1.5px solid #bfdbfe;
            border-radius:12px;
            padding:12px 16px;
            margin-top:16px;
            color:#1d4ed8;
            font-size:12px;
            line-height:1.8;
        ">
            <strong>Demo akun:</strong><br>
            👤 admin / admin123<br>
            👤 mahasiswa / hoax2025
        </div>
        """, unsafe_allow_html=True)
# ==================================
# Guard: jika belum login
# ==================================

if not st.session_state.logged_in:
    show_login()
    st.stop()


# ==================================
# SIDEBAR NAV (setelah login)
# ==================================

with st.sidebar:
    st.markdown(f"""
    <div style="padding:24px 0 28px;text-align:center;">
        <div style="
            width:52px;height:52px;
            background:linear-gradient(135deg,#2563eb,#7c3aed);
            border-radius:14px;display:inline-flex;
            align-items:center;justify-content:center;
            font-size:24px;margin-bottom:12px;
            box-shadow:0 6px 18px rgba(37,99,235,0.25);
        ">🛡️</div>
        <div style="color:#1e293b;font-size:17px;font-weight:800;font-family:'Outfit',sans-serif;margin-bottom:3px;">
            Hoax Analytics
        </div>
        <div style="color:#94a3b8;font-size:11px;font-weight:700;letter-spacing:0.5px;">
            AI INTELLIGENCE SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

    # User badge
    st.markdown(f"""
    <div style="
        background:#eff6ff;border:1.5px solid #bfdbfe;
        border-radius:10px;padding:8px 14px;margin-bottom:20px;
        display:flex;align-items:center;gap:8px;
    ">
        <span style="font-size:18px;">👤</span>
        <div>
            <div style="color:#1e293b !important;font-size:13px;font-weight:700;">{st.session_state.login_user}</div>
            <div style="color:#64748b !important;font-size:11px;">Login aktif</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:8px;padding-left:4px;'>MENU</div>", unsafe_allow_html=True)

    nav_options = ["📊  Dashboard", "⚙️  Run Pipeline", "🔍  Hoax Checker", "📁  Dataset Preview", "📤  Upload Dataset"]
    nav_labels  = ["Dashboard", "Run Pipeline", "Hoax Checker", "Dataset Preview", "Upload Dataset"]
    display_to_key = dict(zip(nav_options, nav_labels))
    current_idx = nav_labels.index(st.session_state.page)

    selected_display = st.radio("nav", nav_options, index=current_idx, label_visibility="collapsed")
    st.session_state.page = display_to_key[selected_display]

    st.markdown("<hr style='border:none;border-top:1.5px solid #e2e8f0;margin:20px 0;'>", unsafe_allow_html=True)

    # Model status
    model_ok = os.path.exists(MODEL_PATH)
    vec_ok   = os.path.exists(VECTORIZER_PATH)

    st.markdown("<div style='color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:10px;padding-left:4px;'>STATUS SISTEM</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:13px;margin-bottom:6px;">
        <span style="color:{'#22c55e' if model_ok else '#ef4444'}">●</span>
        <span style="color:{'#16a34a' if model_ok else '#dc2626'};margin-left:6px;font-weight:600;">
            {'Model loaded' if model_ok else 'Model not found'}
        </span>
    </div>
    <div style="font-size:13px;">
        <span style="color:{'#22c55e' if vec_ok else '#ef4444'}">●</span>
        <span style="color:{'#16a34a' if vec_ok else '#dc2626'};margin-left:6px;font-weight:600;">
            {'Vectorizer loaded' if vec_ok else 'Vectorizer not found'}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.login_user = ""
        st.rerun()

# ==================================
# PAGE ROUTER
# ==================================

import streamlit as st

def render_navbar():
    st.markdown("""
    <style>

    .topnav {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;

        height: 60px;
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(12px);

        border-bottom: 1px solid #e2e8f0;
        z-index: 999999;

        display: flex;
        justify-content: space-around;
        align-items: center;

        padding: 0 10px;
    }

    .navspacer {
        height: 70px;
    }

    .navtitle {
        font-size: 12px;
        font-weight: 700;
        color: #64748b;
    }

    </style>
    """, unsafe_allow_html=True)

    # supaya konten tidak ketutup navbar
    st.markdown('<div class="navspacer"></div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    pages = [
        ("📊", "Dashboard"),
        ("⚙️", "Run Pipeline"),
        ("🔍", "Hoax Checker"),
        ("📁", "Dataset Preview"),
        ("📤", "Upload Dataset"),
    ]

    for i, (icon, label) in enumerate(pages):
        with [col1, col2, col3, col4, col5][i]:
            if st.button(f"{icon}\n{label}", use_container_width=True):
                st.session_state.page = label
                st.rerun()


page = st.session_state.page

# ==================================
# PAGE: DASHBOARD
# ==================================

if page == "Dashboard":

    # Hero banner
    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#1e40af 0%,#2563eb 50%,#7c3aed 100%);
        border-radius:24px;padding:40px 48px;margin-bottom:28px;
        position:relative;overflow:hidden;
        box-shadow:0 8px 32px rgba(37,99,235,0.25);
    ">
        <div style="position:absolute;width:260px;height:260px;background:rgba(255,255,255,0.07);border-radius:50%;right:-60px;top:-80px;"></div>
        <div style="position:absolute;width:160px;height:160px;background:rgba(255,255,255,0.05);border-radius:50%;left:-50px;bottom:-50px;"></div>
        <div style="position:relative;z-index:2;">
            <div style="
                display:inline-block;background:rgba(255,255,255,0.2);
                color:white;font-size:11px;font-weight:700;
                padding:5px 14px;border-radius:20px;letter-spacing:1px;margin-bottom:14px;
            ">MACHINE LEARNING · NLP · KLASIFIKASI</div>
            <div style="color:white;font-size:34px;font-weight:800;font-family:'Outfit',sans-serif;line-height:1.2;margin-bottom:10px;">
                Hoax Detection Intelligence 🛡️
            </div>
            <div style="color:rgba(255,255,255,0.8);font-size:14px;max-width:540px;line-height:1.7;">
                Sistem deteksi berita hoaks berbasis AI dengan Naive Bayes & SVM,
                dilengkapi analytics dataset dan real-time checker.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data (uploaded atau default) ──
    df = None
    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        st.info("📤 Menggunakan dataset yang di-upload.")
    elif os.path.exists(DATASET_PATH):
        df = load_dataset(DATASET_PATH)

    total   = len(df) if df is not None else 0
    n_hoax  = int((df["label"] == 1).sum()) if df is not None and "label" in df.columns else 0
    n_valid = int((df["label"] == 0).sum()) if df is not None and "label" in df.columns else 0
    rate    = round(n_hoax / total * 100, 1) if total else 0

    # ── KPI ──
    st.markdown("### 📊 Ringkasan Dataset")
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("📁 Total Data", f"{total:,}")
    with k2: st.metric("⚠️ Hoaks", f"{n_hoax:,}")
    with k3: st.metric("✅ Valid", f"{n_valid:,}")
    with k4: st.metric("📈 Hoax Rate", f"{rate}%")

    # ── EVALUASI KLASIFIKASI dari hasil pipeline ──
    hasil = st.session_state.pipeline_result
    if hasil:
        st.markdown("---")
        st.markdown("### 🎯 Hasil Evaluasi Model Klasifikasi")

        acc_nb  = hasil.get("acc_nb",  0)
        prec_nb = hasil.get("prec_nb", 0)
        rec_nb  = hasil.get("rec_nb",  0)
        f1_nb   = hasil.get("f1_nb",   0)

        acc_svm  = hasil.get("acc_svm",  0)
        prec_svm = hasil.get("prec_svm", 0)
        rec_svm  = hasil.get("rec_svm",  0)
        f1_svm   = hasil.get("f1_svm",   0)

        tab_nb, tab_svm = st.tabs(["🤖 Naive Bayes", "⚡ SVM"])

        def show_eval_metrics(acc, prec, rec, f1, cm=None):
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.markdown(f"""
                <div style="background:#eff6ff;border:1.5px solid #bfdbfe;border-radius:16px;padding:20px;text-align:center;">
                    <div style="font-size:11px;font-weight:700;color:#3b82f6;letter-spacing:1px;margin-bottom:8px;">ACCURACY</div>
                    <div style="font-size:2.2rem;font-weight:800;color:#1e40af;">{round(acc*100,2)}%</div>
                </div>""", unsafe_allow_html=True)
            with e2:
                st.markdown(f"""
                <div style="background:#f0fdf4;border:1.5px solid #bbf7d0;border-radius:16px;padding:20px;text-align:center;">
                    <div style="font-size:11px;font-weight:700;color:#16a34a;letter-spacing:1px;margin-bottom:8px;">PRECISION</div>
                    <div style="font-size:2.2rem;font-weight:800;color:#166534;">{round(prec*100,2)}%</div>
                </div>""", unsafe_allow_html=True)
            with e3:
                st.markdown(f"""
                <div style="background:#fdf4ff;border:1.5px solid #e9d5ff;border-radius:16px;padding:20px;text-align:center;">
                    <div style="font-size:11px;font-weight:700;color:#9333ea;letter-spacing:1px;margin-bottom:8px;">RECALL</div>
                    <div style="font-size:2.2rem;font-weight:800;color:#6b21a8;">{round(rec*100,2)}%</div>
                </div>""", unsafe_allow_html=True)
            with e4:
                st.markdown(f"""
                <div style="background:#fff7ed;border:1.5px solid #fed7aa;border-radius:16px;padding:20px;text-align:center;">
                    <div style="font-size:11px;font-weight:700;color:#ea580c;letter-spacing:1px;margin-bottom:8px;">F1-SCORE</div>
                    <div style="font-size:2.2rem;font-weight:800;color:#9a3412;">{round(f1*100,2)}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Confusion Matrix
            if cm is not None:
                st.markdown("#### Confusion Matrix")
                labels = ["Valid (0)", "Hoaks (1)"]
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale=[[0,"#eff6ff"],[1,"#1d4ed8"]],
                    showscale=False,
                    text=[[str(v) for v in row] for row in cm],
                    texttemplate="%{text}",
                    textfont={"size":22,"color":"#1e293b"},
                ))
                fig_cm.update_layout(
                    xaxis=dict(title="Prediksi", tickfont=dict(color="#475569")),
                    yaxis=dict(title="Aktual",  tickfont=dict(color="#475569"), autorange="reversed"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Nunito", color="#475569"),
                    height=300,
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        with tab_nb:
            show_eval_metrics(
                acc_nb, prec_nb, rec_nb, f1_nb,
                cm=hasil.get("cm_nb")
            )
        with tab_svm:
            show_eval_metrics(
                acc_svm, prec_svm, rec_svm, f1_svm,
                cm=hasil.get("cm_svm")
            )
    else:
        st.markdown("""
        <div style="
            background:#fffbeb;border:1.5px solid #fde68a;
            border-radius:14px;padding:16px 20px;margin:16px 0;
            color:#92400e;font-size:14px;
        ">
            ⚠️ Belum ada hasil evaluasi. Jalankan <strong>Run Pipeline</strong> terlebih dahulu untuk melihat metrik Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix.
        </div>
        """, unsafe_allow_html=True)

    # ── Charts dataset ──
    if df is not None and "label" in df.columns:
        st.markdown("---")
        st.markdown("### 📈 Analitik Dataset")
        c1, c2 = st.columns(2)

        with c1:
            fig1 = px.pie(
                pd.DataFrame({"K": ["Valid", "Hoaks"], "J": [n_valid, n_hoax]}),
                values="J", names="K", hole=0.60,
                color="K",
                color_discrete_map={"Valid": "#22c55e", "Hoaks": "#ef4444"},
            )
            fig1.update_traces(textfont_color="#1e293b")
            plotly_light_layout(fig1, "Distribusi Label Dataset")
            fig1.update_layout(legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            if "text" in df.columns:
                df2 = df.copy()
                df2["wc"] = df2["text"].astype(str).apply(lambda x: len(x.split()))
                fig2 = px.histogram(
                    df2, x="wc",
                    color="label_name" if "label_name" in df2.columns else None,
                    nbins=40,
                    color_discrete_map={"✅ Valid": "#22c55e", "⚠️ Hoaks": "#ef4444"},
                    labels={"wc": "Jumlah Kata", "count": "Frekuensi"},
                )
                plotly_light_layout(fig2, "Distribusi Panjang Teks")
                fig2.update_layout(legend=dict(title=""))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                fig2 = go.Figure(go.Bar(
                    x=["Valid", "Hoaks"], y=[n_valid, n_hoax],
                    marker_color=["#22c55e", "#ef4444"],
                    text=[n_valid, n_hoax], textposition="outside",
                ))
                plotly_light_layout(fig2, "Jumlah per Kategori")
                st.plotly_chart(fig2, use_container_width=True)

        # ── Preview dataset (10 baris pertama) ──
        st.markdown("---")
        st.markdown("### 👁️ Preview Dataset (10 Baris Pertama)")
        preview_cols = [c for c in df.columns if c not in ("label",)]
        st.dataframe(df[preview_cols].head(10), use_container_width=True, height=320)

    else:
        st.warning("⚠️ Dataset tidak ditemukan. Upload dataset via menu **Upload Dataset** atau pastikan file ada di `dataset/DataHoaxIndonesia_Cleaned.csv`.")

    # ── Quick actions ──
    st.markdown("---")
    st.markdown("### 🚀 Aksi Cepat")
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button("⚙️ Run Pipeline"):
            st.session_state.page = "Run Pipeline"; st.rerun()
    with qa2:
        if st.button("🔍 Hoax Checker"):
            st.session_state.page = "Hoax Checker"; st.rerun()
    with qa3:
        if st.button("📁 Dataset Preview"):
            st.session_state.page = "Dataset Preview"; st.rerun()
    with qa4:
        if st.button("📤 Upload Dataset"):
            st.session_state.page = "Upload Dataset"; st.rerun()

# ==================================
# PAGE: UPLOAD DATASET
# ==================================

elif page == "Upload Dataset":

    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#0f4c2a,#166534,#15803d);
        border-radius:24px;padding:40px 48px;margin-bottom:28px;
        position:relative;overflow:hidden;
        box-shadow:0 8px 24px rgba(21,128,61,0.2);
    ">
        <div style="position:relative;z-index:2;">
            <div style="color:white;font-size:34px;font-weight:800;font-family:'Outfit',sans-serif;margin-bottom:10px;">
                📤 Upload Dataset
            </div>
            <div style="color:rgba(255,255,255,0.8);font-size:14px;max-width:560px;line-height:1.7;">
                Upload file CSV kamu untuk dianalisis. Dataset harus memiliki kolom <code>label</code>
                (0 = Valid, 1 = Hoaks) dan kolom teks (Headline, Body, atau text).
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Pilih File CSV")
    uploaded_file = st.file_uploader(
        "Drag & drop atau klik untuk memilih file",
        type=["csv"],
        help="Format: CSV dengan kolom label (0/1) dan kolom teks."
    )

    if uploaded_file is not None:
        try:
            df_up = pd.read_csv(uploaded_file)

            # Normalisasi kolom
            if "Label" in df_up.columns:
                df_up.rename(columns={"Label": "label"}, inplace=True)
            if "Headline" in df_up.columns and "Body" in df_up.columns:
                df_up["text"] = df_up["Headline"].fillna("") + " " + df_up["Body"].fillna("")
            if "label" in df_up.columns:
                df_up["label_name"] = df_up["label"].map({0: "✅ Valid", 1: "⚠️ Hoaks"})

            st.success(f"✅ File berhasil dibaca! {len(df_up):,} baris, {len(df_up.columns)} kolom.")

            # Info kolom
            u1, u2, u3 = st.columns(3)
            with u1: st.metric("📁 Total Baris", f"{len(df_up):,}")
            with u2: st.metric("🗂️ Jumlah Kolom", f"{len(df_up.columns)}")
            if "label" in df_up.columns:
                with u3: st.metric("⚠️ Hoaks", f"{int((df_up['label']==1).sum()):,}")

            st.markdown("### 👁️ Preview (10 baris pertama)")
            st.dataframe(df_up.head(10), use_container_width=True)

            col_btn, _ = st.columns([1, 2])
            with col_btn:
                if st.button("✅ Gunakan Dataset Ini"):
                    st.session_state.uploaded_df = df_up
                    st.success("Dataset berhasil disimpan! Lihat di Dashboard dan Dataset Preview.")

        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")

    if st.session_state.uploaded_df is not None:
        st.markdown("---")
        st.markdown(f"""
        <div style="background:#f0fdf4;border:1.5px solid #bbf7d0;border-radius:14px;padding:14px 18px;color:#166534;font-size:14px;">
            ✅ Dataset aktif: <strong>{len(st.session_state.uploaded_df):,} baris</strong> sudah tersimpan di sesi ini.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Hapus Dataset Upload (kembali ke default)"):
            st.session_state.uploaded_df = None
            st.rerun()

# ==================================
# PAGE: RUN PIPELINE
# ==================================

elif page == "Run Pipeline":

    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#0f2a1a,#166534,#15803d);
        border-radius:24px;padding:40px 48px;margin-bottom:28px;
        position:relative;overflow:hidden;
        box-shadow:0 8px 24px rgba(21,128,61,0.2);
    ">
        <div style="position:absolute;width:180px;height:180px;background:rgba(255,255,255,0.05);border-radius:50%;right:-40px;top:-40px;"></div>
        <div style="position:relative;z-index:2;">
            <div style="color:white;font-size:34px;font-weight:800;font-family:'Outfit',sans-serif;margin-bottom:10px;">⚙️ Run Pipeline</div>
            <div style="color:rgba(255,255,255,0.8);font-size:14px;max-width:560px;line-height:1.7;">
                Jalankan pipeline lengkap: load data, preprocessing, training Naive Bayes & SVM,
                evaluasi, dan simpan model ke disk.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background:#eff6ff;border:1.5px solid #bfdbfe;
        border-radius:14px;padding:14px 18px;margin-bottom:24px;
        color:#1d4ed8;font-size:14px;line-height:1.7;
    ">
        ℹ️ Model disimpan ke <code>model/model.pkl</code> dan <code>model/vectorizer.pkl</code>.
        Hasil evaluasi langsung tampil di <strong>Dashboard</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Tahapan Pipeline")
    steps = [
        ("📂", "Load Dataset",         "Membaca DataHoaxIndonesia_Cleaned.csv atau dataset upload."),
        ("🌐", "Fetch Realtime News",   "Mengambil berita terbaru dari News API (opsional)."),
        ("🧹", "Text Cleaning",         "Lowercase, hapus URL, karakter spesial, whitespace."),
        ("🔢", "TF-IDF Vectorization",  "Konversi teks ke fitur numerik (max 5000 features)."),
        ("🤖", "Training Naive Bayes",  "Melatih MultinomialNB pada data training (80%)."),
        ("⚡", "Training SVM",          "Melatih LinearSVC sebagai model pembanding."),
        ("📊", "Evaluasi Model",        "Hitung Accuracy, Precision, Recall, F1, Confusion Matrix."),
        ("💾", "Simpan Model",          "Model & vectorizer disimpan ke folder model/."),
    ]
    col_a, col_b = st.columns(2)
    for i, (icon, title, desc) in enumerate(steps):
        with (col_a if i % 2 == 0 else col_b):
            st.markdown(f"""
            <div style="
                background:#ffffff;border:1.5px solid #e2e8f0;
                border-radius:14px;padding:14px 16px;margin-bottom:12px;
                display:flex;gap:12px;align-items:flex-start;
                box-shadow:0 2px 6px rgba(0,0,0,0.05);
            ">
                <div style="
                    min-width:36px;height:36px;
                    background:linear-gradient(135deg,#2563eb,#7c3aed);
                    border-radius:10px;display:flex;align-items:center;
                    justify-content:center;font-size:16px;flex-shrink:0;
                ">{icon}</div>
                <div>
                    <div style="color:#1e293b;font-weight:700;font-size:13px;margin-bottom:2px;">{title}</div>
                    <div style="color:#64748b;font-size:12px;line-height:1.5;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    btn_col, _ = st.columns([1, 2])
    with btn_col:
        run_clicked = st.button("🚀 Jalankan Pipeline Sekarang")

    if run_clicked:
        try:
            from pipeline import run_pipeline
        except ImportError as e:
            st.error(f"❌ Tidak bisa import pipeline.py: {e}")
            st.stop()

        with st.spinner("Pipeline sedang berjalan... mohon tunggu."):
            try:
                hasil = run_pipeline()
                st.session_state.pipeline_result = hasil
                load_model.clear()
            except Exception as e:
                st.error(f"❌ Pipeline gagal: {e}")
                st.stop()

        st.success("✅ Pipeline selesai! Lihat hasil evaluasi di Dashboard.")

    # Tampilkan hasil jika ada
    hasil = st.session_state.pipeline_result
    if hasil:
        st.markdown("### 📊 Hasil Training")

        acc_nb  = hasil.get("acc_nb",  0)
        acc_svm = hasil.get("acc_svm", 0)
        n_data  = len(hasil.get("data", []))
        n_rt    = len(hasil.get("realtime_news", []))

        r1, r2, r3, r4 = st.columns(4)
        with r1: st.metric("🤖 Naive Bayes Acc", f"{round(acc_nb  * 100, 2)}%")
        with r2: st.metric("⚡ SVM Acc",          f"{round(acc_svm * 100, 2)}%")
        with r3: st.metric("📁 Dataset",          f"{n_data:,}")
        with r4: st.metric("🌐 Realtime News",    f"{n_rt:,}")

        label_counts = hasil.get("label_counts", {})
        if label_counts:
            lc1, lc2 = st.columns(2)
            lc_df = pd.DataFrame({
                "Label": list(label_counts.keys()),
                "Jumlah": list(label_counts.values())
            })
            with lc1:
                fig_lc = px.pie(
                    lc_df, values="Jumlah", names="Label", hole=0.6,
                    color="Label",
                    color_discrete_map={"Valid": "#22c55e", "Hoaks": "#ef4444"},
                )
                plotly_light_layout(fig_lc, "Distribusi Label Training")
                fig_lc.update_layout(legend=dict(orientation="h", y=-0.1))
                st.plotly_chart(fig_lc, use_container_width=True)

            with lc2:
                fig_acc = px.bar(
                    x=["Naive Bayes", "SVM"],
                    y=[round(acc_nb*100, 2), round(acc_svm*100, 2)],
                    color=["Naive Bayes", "SVM"],
                    color_discrete_sequence=["#2563eb", "#16a34a"],
                    text=[f"{round(acc_nb*100,2)}%", f"{round(acc_svm*100,2)}%"],
                )
                fig_acc.update_traces(textposition="outside", textfont_color="#1e293b")
                plotly_light_layout(fig_acc, "Perbandingan Akurasi Model")
                fig_acc.update_layout(yaxis=dict(range=[0, 115]), showlegend=False)
                st.plotly_chart(fig_acc, use_container_width=True)

        rt = hasil.get("realtime_news", [])
        if rt:
            st.markdown("### 🌐 Sample Realtime News")
            rt_df = pd.DataFrame(rt)
            if "label" in rt_df.columns:
                rt_df["label"] = rt_df["label"].map({0: "✅ Valid", 1: "⚠️ Hoaks"})
            st.dataframe(rt_df, use_container_width=True, height=280)

        col_go, _ = st.columns([1, 2])
        with col_go:
            if st.button("📊 Lihat Dashboard"):
                st.session_state.page = "Dashboard"; st.rerun()

# ==================================
# PAGE: HOAX CHECKER
# ==================================

elif page == "Hoax Checker":

    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#2e1065,#4c1d95,#6d28d9);
        border-radius:24px;padding:40px 48px;margin-bottom:28px;
        position:relative;overflow:hidden;
        box-shadow:0 8px 24px rgba(109,40,217,0.25);
    ">
        <div style="position:absolute;width:180px;height:180px;background:rgba(255,255,255,0.05);border-radius:50%;right:-40px;top:-40px;"></div>
        <div style="position:relative;z-index:2;">
            <div style="color:white;font-size:34px;font-weight:800;font-family:'Outfit',sans-serif;margin-bottom:10px;">🔍 Hoax Checker</div>
            <div style="color:rgba(255,255,255,0.8);font-size:14px;max-width:560px;line-height:1.7;">
                Deteksi apakah sebuah berita termasuk hoaks atau valid menggunakan model ML yang sudah dilatih.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    model, vectorizer = load_model()

    if model is None or vectorizer is None:
        st.error("❌ Model belum tersedia. Jalankan pipeline terlebih dahulu.")
        btn_p, _ = st.columns([1, 3])
        with btn_p:
            if st.button("⚙️ Ke Run Pipeline"):
                st.session_state.page = "Run Pipeline"; st.rerun()
        st.stop()

    st.markdown("### 📝 Input Berita")
    tab_text, tab_url = st.tabs(["✍️ Input Teks Manual", "🔗 Input dari URL"])

    final_text = ""

    with tab_text:
        st.markdown("<br>", unsafe_allow_html=True)
        text_input = st.text_area(
            "Teks berita:",
            height=200,
            placeholder="Masukkan isi berita di sini...",
            key="text_input"
        )
        if st.button("🔍 Deteksi", key="btn_text"):
            if text_input.strip():
                final_text = text_input.strip()
            else:
                st.warning("⚠️ Teks tidak boleh kosong.")

    with tab_url:
        st.markdown("<br>", unsafe_allow_html=True)
        url_input = st.text_input("URL Berita:", placeholder="https://...", key="url_input")
        if st.button("🌐 Ambil & Deteksi", key="btn_url"):
            if url_input.strip():
                with st.spinner("Mengambil konten dari URL..."):
                    scraped = scrape_url(url_input.strip())
                if scraped.strip():
                    final_text = scraped.strip()
                    st.success(f"✅ Berhasil mengambil {len(scraped.split()):,} kata.")
                else:
                    st.error("❌ Gagal mengambil konten. Coba input teks manual.")
            else:
                st.warning("⚠️ URL tidak boleh kosong.")

    if final_text:
        clean = clean_text(final_text)
        try:
            vec     = vectorizer.transform([clean])
            pred    = model.predict(vec)[0]
            is_hoax = (pred == 1)

            confidence = None
            if hasattr(model, "predict_proba"):
                prob       = model.predict_proba(vec)[0]
                confidence = round(max(prob) * 100, 2)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🎯 Hasil Deteksi")

            if is_hoax:
                bg = "#fef2f2"; brd = "#fecaca"; clr = "#dc2626"
                ico = "⚠️"; lbl = "HOAKS"
                dsc = "Berita ini terindikasi mengandung informasi yang tidak valid."
            else:
                bg = "#f0fdf4"; brd = "#bbf7d0"; clr = "#16a34a"
                ico = "✅"; lbl = "VALID"
                dsc = "Berita ini terklasifikasi sebagai informasi valid."

            conf_str = f"{confidence}%" if confidence is not None else "—"

            st.markdown(f"""
            <div style="
                background:{bg};border:2px solid {brd};
                border-radius:22px;padding:36px 40px;text-align:center;margin-bottom:20px;
                box-shadow:0 4px 16px rgba(0,0,0,0.07);
            ">
                <div style="font-size:52px;margin-bottom:12px;">{ico}</div>
                <div style="color:{clr};font-size:32px;font-weight:800;font-family:'Outfit',sans-serif;letter-spacing:1px;margin-bottom:10px;">
                    {lbl}
                </div>
                <div style="color:#475569;font-size:14px;margin-bottom:12px;">{dsc}</div>
                <div style="color:#94a3b8;font-size:13px;">
                    Confidence: <strong style="color:#1e293b;">{conf_str}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            d1, d2, d3 = st.columns(3)
            with d1: st.metric("📝 Kata (original)", f"{len(final_text.split()):,}")
            with d2: st.metric("🧹 Kata (clean)",    f"{len(clean.split()):,}")
            with d3: st.metric("🎯 Confidence",      conf_str)

            with st.expander("📄 Preview teks (maks. 1500 karakter)"):
                st.markdown(f"""
                <div style="
                    background:#f8fafc;border:1.5px solid #e2e8f0;
                    border-radius:12px;padding:16px;color:#475569;
                    font-size:13px;line-height:1.8;
                ">{final_text[:1500]}{'...' if len(final_text) > 1500 else ''}</div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div style="
                background:#fffbeb;border:1.5px solid #fde68a;
                border-radius:12px;padding:12px 16px;margin-top:14px;
                color:#92400e;font-size:12px;line-height:1.6;
            ">
                ⚠️ <strong>Disclaimer:</strong> Hasil ini dihasilkan model ML dan bukan keputusan final.
                Selalu verifikasi dari sumber terpercaya.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error prediksi: {e}")

# ==================================
# PAGE: DATASET PREVIEW
# ==================================

elif page == "Dataset Preview":

    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#0f1f3a,#1e3a8a,#2563eb);
        border-radius:24px;padding:40px 48px;margin-bottom:28px;
        position:relative;overflow:hidden;
        box-shadow:0 8px 24px rgba(37,99,235,0.2);
    ">
        <div style="position:absolute;width:180px;height:180px;background:rgba(255,255,255,0.05);border-radius:50%;right:-40px;top:-40px;"></div>
        <div style="position:relative;z-index:2;">
            <div style="color:white;font-size:34px;font-weight:800;font-family:'Outfit',sans-serif;margin-bottom:10px;">📁 Dataset Preview</div>
            <div style="color:rgba(255,255,255,0.8);font-size:14px;max-width:560px;line-height:1.7;">
                Jelajahi dataset hoaks Indonesia. Filter berdasarkan label, cari kata kunci, dan download hasil filter.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Gunakan uploaded atau default
    df = None
    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        st.info("📤 Menampilkan dataset yang di-upload.")
    elif os.path.exists(DATASET_PATH):
        df = load_dataset(DATASET_PATH)
    else:
        st.error(f"❌ Dataset tidak ditemukan di `{DATASET_PATH}`. Upload dataset terlebih dahulu.")
        if st.button("📤 Upload Dataset"):
            st.session_state.page = "Upload Dataset"; st.rerun()
        st.stop()

    total   = len(df)
    n_hoax  = int((df["label"] == 1).sum()) if "label" in df.columns else 0
    n_valid = int((df["label"] == 0).sum()) if "label" in df.columns else 0
    n_cols  = len(df.columns)

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("📁 Total Baris", f"{total:,}")
    with m2: st.metric("⚠️ Hoaks",       f"{n_hoax:,}")
    with m3: st.metric("✅ Valid",        f"{n_valid:,}")
    with m4: st.metric("🗂️ Kolom",       f"{n_cols}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔧 Filter & Pencarian")

    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        search_q = st.text_input("🔎 Cari teks:", placeholder="ketik kata kunci...")
    with fc2:
        f_label = st.selectbox("Label:", ["Semua", "✅ Valid", "⚠️ Hoaks"])
    with fc3:
        show_n = st.selectbox("Tampilkan:", [50, 100, 200, 500, "Semua"], index=1)

    df_f = df.copy()
    if f_label == "✅ Valid"  and "label" in df_f.columns: df_f = df_f[df_f["label"] == 0]
    if f_label == "⚠️ Hoaks" and "label" in df_f.columns: df_f = df_f[df_f["label"] == 1]
    if search_q.strip():
        tcol = "text" if "text" in df_f.columns else df_f.columns[0]
        df_f = df_f[df_f[tcol].astype(str).str.contains(search_q.strip(), case=False, na=False)]
    if show_n != "Semua":
        df_f = df_f.head(int(show_n))

    st.markdown(f"<div style='color:#64748b;font-size:13px;margin-bottom:10px;'>Menampilkan <strong style='color:#1e293b'>{len(df_f):,}</strong> baris</div>", unsafe_allow_html=True)

    cols_show = [c for c in df_f.columns if c not in ("label",)]
    st.dataframe(df_f[cols_show], use_container_width=True, height=460)

    dl_col, _ = st.columns([1, 2])
    with dl_col:
        st.download_button(
            "⬇️ Download Hasil Filter (CSV)",
            df_f.to_csv(index=False).encode("utf-8"),
            "dataset_filtered.csv",
            "text/csv"
        )

    with st.expander("🔬 Info Kolom Dataset"):
        ci = pd.DataFrame({
            "Kolom":    df.columns.tolist(),
            "Tipe":     [str(df[c].dtype) for c in df.columns],
            "Non-Null": [df[c].notna().sum() for c in df.columns],
            "Null":     [df[c].isna().sum()  for c in df.columns],
            "Unique":   [df[c].nunique()     for c in df.columns],
        })
        st.dataframe(ci, use_container_width=True)

    if "label" in df.columns:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Distribusi Dataset")
        dc1, dc2 = st.columns(2)

        with dc1:
            fig_p = px.pie(
                pd.DataFrame({"K": ["Valid", "Hoaks"], "J": [n_valid, n_hoax]}),
                values="J", names="K", hole=0.6,
                color="K",
                color_discrete_map={"Valid": "#22c55e", "Hoaks": "#ef4444"}
            )
            plotly_light_layout(fig_p, "Proporsi Label")
            fig_p.update_layout(legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig_p, use_container_width=True)

        with dc2:
            if "text" in df.columns:
                df3 = df.copy()
                df3["wc2"] = df3["text"].astype(str).apply(lambda x: len(x.split()))
                fig_h = px.histogram(
                    df3, x="wc2",
                    color="label_name" if "label_name" in df3.columns else None,
                    nbins=40,
                    color_discrete_map={"✅ Valid": "#22c55e", "⚠️ Hoaks": "#ef4444"},
                    labels={"wc2": "Jumlah Kata", "count": "Frekuensi"},
                )
                plotly_light_layout(fig_h, "Distribusi Panjang Teks")
                fig_h.update_layout(legend=dict(title=""))
                st.plotly_chart(fig_h, use_container_width=True)