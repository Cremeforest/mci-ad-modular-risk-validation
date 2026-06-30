from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="MCI-to-AD Risk Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CUSTOM_CSS = """
<style>
[data-testid="stAppViewContainer"] {
    background: #f7f8fb;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2.5rem;
    max-width: 1120px;
}
.hero {
    padding: 1.65rem 1.8rem;
    border-radius: 22px;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    margin-bottom: 1.15rem;
}
.hero-title {
    font-size: 2.15rem;
    font-weight: 760;
    color: #111827;
    line-height: 1.15;
    margin-bottom: 0.55rem;
}
.disclaimer {
    display: inline-block;
    padding: 0.45rem 0.72rem;
    border-radius: 999px;
    background: #eef2ff;
    color: #3730a3;
    font-size: 0.88rem;
    font-weight: 600;
}
.section-card {
    padding: 1.15rem 1.25rem;
    border-radius: 18px;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.045);
    margin: 0.85rem 0 1rem 0;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 720;
    color: #111827;
    margin-bottom: 0.25rem;
}
.subtle {
    color: #6b7280;
    font-size: 0.9rem;
}
.result-card {
    padding: 1.1rem 0.9rem;
    border-radius: 18px;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.055);
    text-align: center;
    min-height: 142px;
}
.result-horizon {
    font-size: 0.92rem;
    font-weight: 700;
    color: #64748b;
    margin-bottom: 0.2rem;
}
.result-risk {
    font-size: 1.9rem;
    font-weight: 800;
    color: #111827;
    margin-bottom: 0.2rem;
}
.result-level {
    display: inline-block;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
    font-size: 0.80rem;
    font-weight: 700;
}
.level-lower {
    background: #dcfce7;
    color: #166534;
}
.level-intermediate {
    background: #fef3c7;
    color: #92400e;
}
.level-higher {
    background: #fee2e2;
    color: #991b1b;
}
.stButton>button {
    border-radius: 12px !important;
    height: 3rem;
    font-weight: 760 !important;
    font-size: 1rem !important;
    background: #2563eb !important;
    border: 0 !important;
    color: white !important;
    box-shadow: 0 8px 20px rgba(37,99,235,0.18);
}
.stButton>button:hover {
    background: #1d4ed8 !important;
}
[data-testid="stTabs"] button {
    font-weight: 650;
}
hr {
    margin: 0.8rem 0 1rem 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))


def parse_optional_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


@dataclass
class VisitSummary:
    age: float
    education: float
    sex_male: float
    mmse_last: float
    mmse_change: float
    cdr_global_last: float
    cdrsb_last: float
    cdrsb_change: float
    faq_last: float
    faq_change: float
    adas_available: bool
    adas_last: float
    adas_change: float
    n_visits: int


def first_last_change(values: list[float | None], default: float) -> tuple[float, float]:
    clean = [float(x) for x in values if x is not None]
    if not clean:
        return default, 0.0
    return clean[-1], clean[-1] - clean[0]


def summarize(visits: list[dict], sex: str, education: float) -> VisitSummary:
    # Count a visit if at least one visit-level field is filled.
    active = []
    for v in visits:
        keys = ["age", "mmse", "cdr_global", "cdrsb", "faq", "adas13"]
        if any(v.get(k) is not None for k in keys):
            active.append(v)
    if not active:
        active = visits[:1]

    age_last, _ = first_last_change([v["age"] for v in active], 74.0)
    mmse_last, mmse_change = first_last_change([v["mmse"] for v in active], 28.0)
    cdr_last, _ = first_last_change([v["cdr_global"] for v in active], 0.5)
    cdrsb_last, cdrsb_change = first_last_change([v["cdrsb"] for v in active], 1.0)
    faq_last, faq_change = first_last_change([v["faq"] for v in active], 2.0)
    adas_values = [v["adas13"] for v in active if v["adas13"] is not None]
    adas_last, adas_change = first_last_change(adas_values, 13.0)

    return VisitSummary(
        age=age_last,
        education=education,
        sex_male=1.0 if sex == "Male" else 0.0,
        mmse_last=mmse_last,
        mmse_change=mmse_change,
        cdr_global_last=cdr_last,
        cdrsb_last=cdrsb_last,
        cdrsb_change=cdrsb_change,
        faq_last=faq_last,
        faq_change=faq_change,
        adas_available=len(adas_values) > 0,
        adas_last=adas_last,
        adas_change=adas_change,
        n_visits=max(len(active), 1),
    )


def demo_predict(summary: VisitSummary) -> list[dict]:
    cdr_map = {0.0: -0.55, 0.5: 0.00, 1.0: 0.65, 2.0: 1.15, 3.0: 1.65}

    severity = 0.0
    severity += 0.24 * ((summary.age - 74.0) / 8.0)
    severity += 0.06 * summary.sex_male
    severity -= 0.12 * ((summary.education - 16.0) / 6.0)

    severity += 0.55 * ((28.0 - summary.mmse_last) / 5.0)
    severity += 0.20 * ((-summary.mmse_change) / 3.0)

    severity += cdr_map.get(float(summary.cdr_global_last), 0.0)
    severity += 0.52 * ((summary.cdrsb_last - 1.0) / 4.0)
    severity += 0.18 * (summary.cdrsb_change / 3.0)

    severity += 0.40 * ((summary.faq_last - 2.0) / 8.0)
    severity += 0.14 * (summary.faq_change / 5.0)

    if summary.adas_available:
        severity += 0.28 * ((summary.adas_last - 13.0) / 10.0)
        severity += 0.10 * (summary.adas_change / 6.0)

    if summary.n_visits >= 2:
        severity += 0.04 * min(summary.n_visits - 1, 4)

    severity = max(min(severity, 2.40), -1.70)

    base = {"1 year": 0.10, "2 years": 0.24, "3 years": 0.40, "5 years": 0.60}
    risks = {h: sigmoid(logit(p) + severity) for h, p in base.items()}

    ordered = ["1 year", "2 years", "3 years", "5 years"]
    running = 0.0
    rows = []
    for h in ordered:
        running = max(running, risks[h])
        risk = running
        if risk < 0.20:
            level = "Lower"
            css = "level-lower"
        elif risk < 0.50:
            level = "Intermediate"
            css = "level-intermediate"
        else:
            level = "Higher"
            css = "level-higher"
        rows.append({"horizon": h, "risk": risk, "risk_text": f"{100 * risk:.1f}%", "level": level, "css": css})
    return rows


def render_result_card(item: dict):
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-horizon">{item["horizon"]}</div>
            <div class="result-risk">{item["risk_text"]}</div>
            <span class="result-level {item["css"]}">{item["level"]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(float(item["risk"]))


st.markdown(
    """
    <div class="hero">
        <div class="hero-title">MCI-to-AD Risk Demo</div>
        <div class="disclaimer">
            Research demonstration only; not intended for diagnosis, treatment decisions, or medical advice.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Patient information</div>', unsafe_allow_html=True)
p1, p2, p3 = st.columns([1, 1, 1])
with p1:
    sex = st.selectbox("Sex", ["Female", "Male"])
with p2:
    n_visits = st.slider("Number of visits", min_value=1, max_value=8, value=1)
with p3:
    education_text = st.text_input("Education years", value="16", placeholder="Optional")
education = parse_optional_float(education_text)
if education is None:
    education = 16.0
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Visit records</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Fill visits in chronological order. All visit fields can be left blank if unavailable.</div>', unsafe_allow_html=True)

tabs = st.tabs([f"Visit {i}" for i in range(1, n_visits + 1)])
visits = []

for i, tab in enumerate(tabs, start=1):
    with tab:
        c1, c2, c3 = st.columns(3)
        with c1:
            age = parse_optional_float(st.text_input("Age", value="74" if i == 1 else "", placeholder="Optional", key=f"age_{i}"))
            mmse = parse_optional_float(st.text_input("MMSE", value="28" if i == 1 else "", placeholder="Optional", key=f"mmse_{i}"))
        with c2:
            cdr_global = parse_optional_float(st.text_input("CDR global", value="0.5" if i == 1 else "", placeholder="0, 0.5, 1, 2, 3", key=f"cdrg_{i}"))
            cdrsb = parse_optional_float(st.text_input("CDRSB", value="1" if i == 1 else "", placeholder="Optional", key=f"cdrsb_{i}"))
        with c3:
            faq = parse_optional_float(st.text_input("FAQ total", value="2" if i == 1 else "", placeholder="Optional", key=f"faq_{i}"))
            adas13 = parse_optional_float(st.text_input("ADAS13", value="", placeholder="Optional", key=f"adas_{i}"))

        # Keep CDR global in allowed range if typed oddly.
        if cdr_global is not None:
            allowed = [0.0, 0.5, 1.0, 2.0, 3.0]
            cdr_global = min(allowed, key=lambda x: abs(x - cdr_global))

        visits.append({"age": age, "mmse": mmse, "cdr_global": cdr_global, "cdrsb": cdrsb, "faq": faq, "adas13": adas13})

st.markdown("</div>", unsafe_allow_html=True)

predict_col, _ = st.columns([1, 2])
with predict_col:
    clicked = st.button("Predict", type="primary", use_container_width=True)

if clicked:
    summary = summarize(visits, sex=sex, education=education)
    pred = demo_predict(summary)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Predicted MCI-to-AD conversion risk</div>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    for col, item in zip([r1, r2, r3, r4], pred):
        with col:
            render_result_card(item)

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Input summary"):
        summary_df = pd.DataFrame(
            [
                {
                    "visits used": summary.n_visits,
                    "latest age": summary.age,
                    "latest MMSE": summary.mmse_last,
                    "MMSE change": summary.mmse_change,
                    "latest CDR global": summary.cdr_global_last,
                    "latest CDRSB": summary.cdrsb_last,
                    "CDRSB change": summary.cdrsb_change,
                    "latest FAQ": summary.faq_last,
                    "FAQ change": summary.faq_change,
                    "ADAS13 available": summary.adas_available,
                }
            ]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
