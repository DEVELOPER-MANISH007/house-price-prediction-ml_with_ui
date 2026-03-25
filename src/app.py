import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")
PIPELINE_PATH = os.path.join(BASE_DIR, "..", "models", "pipeline.pkl")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: "Plus Jakarta Sans", "Segoe UI", system-ui, sans-serif;
    }

    .stApp {
        background: #f4f6fa;
    }

    .main .block-container {
        padding-top: 0.85rem;
        padding-bottom: 2.5rem;
        max-width: 880px;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e8f0;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] td {
        color: #3d4556 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1c2230 !important;
    }

    .hero-wrap {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 50%, #1e40af 100%);
        padding: 1.65rem 1.85rem;
        border-radius: 18px;
        color: #f8fafc;
        margin: 0 0 1.1rem 0;
        box-shadow: 0 16px 36px -12px rgba(37, 99, 235, 0.42);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .hero-wrap h1 {
        margin: 0;
        font-size: clamp(1.8rem, 4.2vw, 2.4rem);
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.12;
    }
    .hero-wrap p {
        margin: 0.55rem 0 0 0;
        font-size: 0.98rem;
        opacity: 0.95;
        max-width: 52ch;
        line-height: 1.55;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.95rem;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.22);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
    }

    .section-label {
        font-size: 0.66rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #64748b;
        margin: 0 0 0.45rem 0;
    }

    div[data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #e5e8f0;
        border-radius: 16px;
        padding: 1.35rem 1.45rem 1.5rem;
        box-shadow: 0 1px 8px rgba(28, 34, 48, 0.05);
    }

    .prediction-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #c7d2fe;
        border-radius: 16px;
        padding: 1.25rem 1.45rem;
        margin: 0 0 1.25rem 0;
        box-shadow: 0 14px 32px -14px rgba(79, 70, 229, 0.25);
    }
    .prediction-card .label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #4f46e5;
        margin: 0;
    }
    .prediction-card .amount {
        font-size: clamp(1.6rem, 4.2vw, 2.15rem);
        font-weight: 700;
        color: #1e3a8a;
        letter-spacing: -0.02em;
        margin: 0.3rem 0 0 0;
    }
    .prediction-card .caption {
        font-size: 0.86rem;
        color: #64748b;
        margin: 0.35rem 0 0 0;
    }

    .hint-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.95rem 1.1rem;
        margin: 0 0 1.25rem 0;
        color: #64748b;
        font-size: 0.9rem;
    }

    hr.sep {
        border: none;
        border-top: 1px solid #eef1f6;
        margin: 0.95rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not os.path.exists(MODEL_PATH):
    st.error("Model missing. Train first: `python src/main.py train`")
    st.stop()

model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

with st.sidebar:
    st.markdown("### Help")
    st.markdown("Use **☰** to open this panel. Fill the form, then **Predict price**.")
    st.markdown("---")
    st.markdown("**Example row**")
    with st.expander("Sample values", expanded=False):
        st.markdown(
            """
| Field | Value |
|------|--------|
| Longitude | -122.23 |
| Latitude | 37.88 |
| Median age | 20 |
| Rooms | 1000 |
| Bedrooms | 200 |
| Population | 500 |
| Households | 150 |
| Income | 3.5 |
| Ocean | NEAR BAY |
"""
        )

# 1) Title on top
st.markdown(
    """
    <div class="hero-wrap">
        <h1>🏠 House price prediction</h1>
        <p>Median house value from your Random Forest model. <strong>Result</strong> is shown right below; scroll down to enter inputs and run predict.</p>
        <div class="hero-badges">
            <span class="hero-badge">Regression</span>
            <span class="hero-badge">Random Forest</span>
            <span class="hero-badge">scikit-learn</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2) Prediction directly under title (session from last submit)
st.markdown('<p class="section-label">Latest prediction</p>', unsafe_allow_html=True)
if st.session_state.last_prediction is not None:
    pv = st.session_state.last_prediction
    st.markdown(
        f"""
        <div class="prediction-card">
            <p class="label">Estimated median house value</p>
            <p class="amount">${pv:,.2f}</p>
            <p class="caption">Same unit as training data (typically USD). Submit again to update.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.metric("Rounded", f"${pv:,.0f}")
else:
    st.markdown(
        """
        <div class="hint-box">
            No prediction yet — fill the form below and click <strong>Predict price</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

# 3) Form below
st.markdown('<p class="section-label">Property details</p>', unsafe_allow_html=True)

with st.form("predict_form", border=False):
    st.markdown("**Location**")
    c1, c2 = st.columns(2)
    with c1:
        longitude = st.number_input("Longitude", format="%.5f")
    with c2:
        latitude = st.number_input("Latitude", format="%.5f")

    st.markdown('<hr class="sep">', unsafe_allow_html=True)
    st.markdown("**Building**")
    c3, c4, c5 = st.columns(3)
    with c3:
        housing_median_age = st.number_input("Housing median age", format="%.2f")
    with c4:
        total_rooms = st.number_input("Total rooms", format="%.2f")
    with c5:
        total_bedrooms = st.number_input("Total bedrooms", format="%.2f")

    st.markdown('<hr class="sep">', unsafe_allow_html=True)
    st.markdown("**Population & income**")
    c6, c7, c8 = st.columns(3)
    with c6:
        population = st.number_input("Population", format="%.2f")
    with c7:
        households = st.number_input("Households", format="%.2f")
    with c8:
        median_income = st.number_input("Median income", format="%.4f")

    st.markdown('<hr class="sep">', unsafe_allow_html=True)
    ocean_proximity = st.selectbox(
        "Ocean proximity",
        ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
        index=3,
    )

    submitted = st.form_submit_button("Predict price", type="primary", use_container_width=True)

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "longitude": longitude,
                "latitude": latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity,
            }
        ]
    )
    transformed = pipeline.transform(input_data)
    st.session_state.last_prediction = float(model.predict(transformed)[0])
    st.rerun()
