import streamlit as st
import requests
import random
from pathlib import Path
import time

API_BASE = "http://127.0.0.1:8000"

# -------------------- Helpers --------------------
def inject_css(css_path: str):
    p = Path(css_path)
    if p.exists():
        st.markdown(f"<style>{p.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

def show_image_if_exists(img_path: str, width: int = 72):
    p = Path(img_path)
    if p.exists():
        st.image(str(p), width=width)

@st.cache_data(show_spinner=False)
def fetch_feature_names():
    r = requests.get(f"{API_BASE}/feature_names", timeout=10)
    r.raise_for_status()
    return r.json()["features"]

@st.cache_data(show_spinner=False)
def fetch_feature_stats():
    r = requests.get(f"{API_BASE}/feature_stats", timeout=10)
    r.raise_for_status()
    return r.json().get("stats", {})

def random_value_for_feature(stats: dict, mode: str) -> float:
    # stats: {"min","q1","median","q3","max","mean","std"}
    if mode == "Realistic (Q1–Q3)":
        return random.uniform(stats["q1"], stats["q3"])
    if mode == "Extreme (min–max)":
        return random.uniform(stats["min"], stats["max"])
    # Mixed
    return random.uniform(stats["q1"], stats["q3"]) if random.random() < 0.85 else random.uniform(stats["min"], stats["max"])


# -------------------- Page --------------------
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")
inject_css("frontend/assets/style.css")

# HERO
left, right = st.columns([1, 6])
with left:
    st.markdown("<br/>", unsafe_allow_html=True)  # spacing
    st.markdown("<br/>", unsafe_allow_html=True)  # spacing
    show_image_if_exists("frontend/assets/ribbon.png", width=72)

with right:
    st.markdown(
        f"""
        <div class="hero pop">
          <div class="hero-title">Breast Cancer Prediction</div>
          <div class="hero-sub">A prevention-oriented ML demo UI with dataset-based randomization (FastAPI + Scaler + Softmax).</div>
          <div class="chips">
            <span class="chip">Prevention Focus</span>
            <span class="chip">Dataset-based Randomize</span>
            <span class="chip">Not medical advice</span>
            <span class="chip">API: {API_BASE}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<hr class='hr-pink'/>", unsafe_allow_html=True)

# Load API data
try:
    feature_names = fetch_feature_names()
    stats = fetch_feature_stats()
except Exception as e:
    st.error(f"Impossible de récupérer les infos depuis l’API. Détails: {e}")
    st.stop()

# Init session state values (IMPORTANT for randomize/reset)
for f in feature_names:
    st.session_state.setdefault(f, 0.0)

# Info card
st.markdown(
    f"""
    <div class="card pop">
      <div class="card-title">Input features</div>
      <div class="card-desc">
        {len(feature_names)} features attendues par le modèle. Randomize génère des valeurs réalistes à partir des stats du dataset (Q1/Q3, min/max).
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")

# -------------------- Controls --------------------
st.markdown("<div class='card pop shimmer'>", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns([1.5, 1.5, 1, 1, 1])

with c1:
    mode = st.selectbox("Randomize mode", ["Realistic (Q1–Q3)", "Extreme (min–max)", "Mixed"], index=0)

with c2:
    model = st.selectbox("Model", ["Softmax", "SVM", "MLP", "XGBoost"], index=0)

with c3:
    compact = st.toggle("Compact view", value=False, help="3 colonnes (normal) / 4 colonnes (compact)")

with c4:
    reset_clicked = st.button("Reset", use_container_width=True)

with c5:
    randomize_clicked = st.button("Randomize", use_container_width=True)

# Actions (THIS is what was missing in your version)
if reset_clicked:
    for f in feature_names:
        st.session_state[f] = 0.0
    st.rerun()

if randomize_clicked:
    for f in feature_names:
        s = stats.get(f)
        if not s:
            st.session_state[f] = 0.0
        else:
            st.session_state[f] = round(random_value_for_feature(s, mode), 6)
    st.rerun()

st.markdown(
    "<div class='footer-note'>Prévention: démonstration ML. Ne remplace pas un avis médical (dépistage, consultation, mammographie).</div>",
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# -------------------- Form --------------------
st.markdown("<div class='card pop'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>Prediction form</div>", unsafe_allow_html=True)
st.markdown("<div class='card-desc'>Remplis les valeurs (raw). Le backend applique scaler → modèle → résultat.</div>", unsafe_allow_html=True)
st.write("")

num_cols = 4 if compact else 2
cols = st.columns(num_cols)

features_payload = {}
with st.form("predict_form"):
    for i, f in enumerate(feature_names):
        with cols[i % num_cols]:
            features_payload[f] = st.number_input(
                label=f,
                value=float(st.session_state[f]),
                format="%.6f",
                key=f
            )

    submitted = st.form_submit_button("Predict")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Predict --------------------
if submitted:
    st.markdown("<br/>", unsafe_allow_html=True)  # spacing
    placeholder = st.empty()
    placeholder.markdown("""
    <section class="loader">
      <div class="slider" style="--i:0"></div>
      <div class="slider" style="--i:1"></div>
      <div class="slider" style="--i:2"></div>
      <div class="slider" style="--i:3"></div>
      <div class="slider" style="--i:4"></div>
    </section>
    """, unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)  # spacing
    try:
        payload = {"features": features_payload, "model": model}
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=20)
        if r.status_code != 200:
            st.error(f"Erreur API ({r.status_code}): {r.text}")
            st.stop()

        result = r.json()
        pred = result.get("prediction")
        proba = result.get("proba")
        proba_txt = f"{float(proba):.4f}" if proba is not None else "N/A"
        pct = int(float(proba) * 100) if proba is not None else 0
        model_used = result.get("model_used", model)

        time.sleep(3)

        placeholder.empty()

        st.markdown("<div class='result pop' id='result-section'>", unsafe_allow_html=True)
        st.markdown("### Result", unsafe_allow_html=True)

        st.markdown(
            f"<div class='badge'>Model: <span style='color:#e91e63'>{model_used}</span></div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='badge'>Prediction: <span style='color:#e91e63'>{'Malade de cancer du sein' if pred == 1 else 'Pas malade'}</span> "
            f"• Proba: <span style='color:#e91e63'>{proba_txt}</span></div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="proba-wrap">
              <div class="proba-label">Confidence (visual)</div>
              <div class="proba-bar">
                <div class="proba-fill" style="width:{pct}%"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        proba_by_class = result.get("proba_by_class")
        if proba_by_class:
            with st.expander("Probabilities by class"):
                st.json(proba_by_class)

        with st.expander("SHAP Explanation"):
            shap_exp = result.get("shap_explanation")
            if shap_exp:
                st.write("SHAP values indicate the contribution of each feature to the prediction (for the predicted class). Higher absolute values mean more influence.")
                # Sort by absolute SHAP value
                sorted_shap = sorted(shap_exp.items(), key=lambda x: abs(x[1]), reverse=True)
                for feat, val in sorted_shap[:10]:  # Show top 10
                    st.write(f"**{feat}**: {val:.4f}")
                if len(sorted_shap) > 10:
                    st.write(f"... and {len(sorted_shap) - 10} more features.")
            else:
                st.write("SHAP explanation not available.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<script>document.getElementById('result-section').scrollIntoView({behavior: 'smooth'});</script>""", unsafe_allow_html=True)

    except Exception as e:
        placeholder.empty()
        st.error(f"Erreur lors de l’appel API: {e}")
