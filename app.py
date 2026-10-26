from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUTPUT = BASE / "outputs"
MODEL = BASE / "models" / "delay_prediction_model.joblib"

st.set_page_config(
    page_title="TransitIQ | Command Centre",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLUE = "#3478F6"
PURPLE = "#895BFF"
GREEN = "#14D6A0"
ORANGE = "#FF9D52"
RED = "#FF5369"
CYAN = "#36C8FF"
NAVY = "#071329"
PANEL = "#0B1B35"
PALETTE = [BLUE, PURPLE, GREEN, ORANGE, RED, CYAN]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg:#061126;
    --panel:#0B1B35;
    --panel2:#102441;
    --border:#1B365C;
    --text:#ECF4FF;
    --muted:#8DA5C8;
    --blue:#3478F6;
    --green:#14D6A0;
}

.stApp {
    background:
        radial-gradient(ellipse at 82% 0%, #172E59 0%, transparent 34%),
        radial-gradient(ellipse at 0% 65%, #0C1D39 0%, transparent 30%),
        #061126;
    color:var(--text);
    font-family:'DM Sans',sans-serif;
}

[data-testid="stHeader"] {
    background:rgba(6,17,38,.65);
    backdrop-filter:blur(16px);
}

[data-testid="stToolbar"] {right:1rem}

.block-container {
    max-width:1800px;
    padding:1.5rem 1.8rem 3.5rem;
}

h1,h2,h3,h4,p,label {
    font-family:'DM Sans',sans-serif;
}

h1,h2,h3,h4 {
    color:#ECF4FF!important;
}

p,label {
    color:#C3D2EA;
}

[data-testid="stSidebar"] {
    background:
        radial-gradient(ellipse at 20% 0%,#14284B 0%,transparent 40%),
        linear-gradient(180deg,#091832 0%,#071329 100%);
    border-right:1px solid #20395D;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top:1.1rem;
}

[data-testid="stSidebar"] * {
    color:#C6D7F0;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color:#91A9CA;
}

.brand {
    display:flex;
    align-items:center;
    gap:12px;
    font-family:'Manrope',sans-serif;
    font-size:25px;
    font-weight:800;
    color:white;
    letter-spacing:-1px;
    padding:5px 3px 0;
}

.brand-icon {
    display:flex;
    align-items:center;
    justify-content:center;
    width:44px;
    height:44px;
    border-radius:14px;
    color:white;
    background:linear-gradient(135deg,#3478F6,#764DFF);
    box-shadow:0 0 24px #3478F655;
    font-size:24px;
    transition:transform .3s ease,box-shadow .3s ease;
}

.brand-icon:hover {
    transform:rotate(-8deg) scale(1.1);
    box-shadow:0 0 32px #3478F699;
}

.brand span {
    color:#58A4FF;
}

.brand-sub {
    font-size:10px;
    color:#8199BE;
    letter-spacing:1.3px;
    margin:8px 0 22px;
}

.eyebrow {
    color:#69A4FF;
    font-size:10px;
    font-weight:800;
    letter-spacing:2px;
}

.hero-title {
    color:#F3F7FF;
    font-family:'Manrope',sans-serif;
    font-size:clamp(27px,3vw,39px);
    font-weight:800;
    letter-spacing:-1.4px;
    line-height:1.2;
    margin:7px 0;
    animation:heroIn .65s ease both;
}

.hero-title span {
    background:linear-gradient(90deg,#58A2FF,#8C75FF,#36C8FF);
    background-size:200% auto;
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent;
    animation:gradientMove 5s linear infinite;
}

.hero-sub {
    color:#8FA9CD;
    font-size:13px;
    margin-bottom:23px;
    line-height:1.7;
}

.topline {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
    flex-wrap:wrap;
    margin-bottom:25px;
}

.status-pill {
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding:9px 13px;
    border-radius:30px;
    background:#102D37;
    border:1px solid #1A594E;
    color:#5DE8C1;
    font-size:11px;
    font-weight:700;
    transition:all .25s ease;
}

.status-pill:hover {
    border-color:#14D6A0;
    box-shadow:0 0 22px #14D6A022;
    transform:translateY(-2px);
}

.status-dot {
    height:7px;
    width:7px;
    background:#14D6A0;
    border-radius:50%;
    box-shadow:0 0 12px #14D6A0;
    animation:pulse 1.8s infinite;
}

@keyframes pulse {
    0%,100% {opacity:1;box-shadow:0 0 10px #14D6A0}
    50% {opacity:.45;box-shadow:0 0 2px #14D6A0}
}

@keyframes rise {
    from {opacity:0;transform:translateY(12px)}
    to {opacity:1;transform:translateY(0)}
}

@keyframes heroIn {
    from {opacity:0;transform:translateY(14px)}
    to {opacity:1;transform:translateY(0)}
}

@keyframes gradientMove {
    0% {background-position:0% center}
    100% {background-position:200% center}
}

@keyframes glow {
    0%,100% {box-shadow:0 0 0 rgba(52,120,246,0)}
    50% {box-shadow:0 0 22px rgba(52,120,246,.12)}
}

@keyframes borderGlow {
    0%,100% {border-color:#1B365C}
    50% {border-color:#315B91}
}

.metric-card {
    position:relative;
    overflow:hidden;
    background:linear-gradient(145deg,#102443,#09182F);
    border:1px solid #1B365C;
    border-radius:17px;
    padding:18px 18px 16px;
    min-height:139px;
    transition:transform .28s ease,border-color .28s ease,
        box-shadow .28s ease,background .28s ease;
    animation:rise .45s ease both;
}

.metric-card:hover {
    transform:translateY(-6px) scale(1.015);
    border-color:#438BFF;
    box-shadow:0 12px 32px #0005,0 0 18px #3478F622;
    background:linear-gradient(145deg,#142E53,#0C1C36);
}

.metric-card:after {
    content:"";
    position:absolute;
    right:-30px;
    top:-35px;
    width:110px;
    height:110px;
    border-radius:50%;
    background:var(--accent);
    opacity:.07;
    filter:blur(5px);
    transition:transform .35s ease,opacity .35s ease;
}

.metric-card:hover:after {
    transform:scale(1.8);
    opacity:.18;
}

.metric-top {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:8px;
}

.metric-label {
    font-size:11px;
    font-weight:600;
    color:#9BB1D1;
}

.metric-icon {
    display:flex;
    align-items:center;
    justify-content:center;
    width:36px;
    height:36px;
    border-radius:11px;
    background:color-mix(in srgb,var(--accent) 19%,transparent);
    color:var(--accent);
    font-size:19px;
    transition:transform .3s ease;
}

.metric-card:hover .metric-icon {
    transform:rotate(-8deg) scale(1.12);
}

.metric-value {
    font-family:'Manrope',sans-serif;
    font-size:28px;
    line-height:1.3;
    font-weight:800;
    color:#F2F7FF;
    margin-top:12px;
    overflow-wrap:anywhere;
}

.metric-foot {
    font-size:10px;
    color:#92A8C8;
    margin-top:5px;
}

.panel {
    background:linear-gradient(145deg,#0D203D,#09172E);
    border:1px solid #1B365C;
    border-radius:17px;
    padding:17px;
    transition:border-color .25s,box-shadow .25s,transform .25s;
}

.panel:hover {
    border-color:#3B6597;
    box-shadow:0 8px 24px #0003,0 0 16px #3478F610;
    transform:translateY(-2px);
}

.section-title {
    font-family:'Manrope',sans-serif;
    font-size:17px;
    font-weight:800;
    color:#EDF4FF;
    margin:9px 0 3px;
}

.section-sub {
    font-size:11px;
    color:#8DA5C8;
    margin-bottom:13px;
    line-height:1.6;
}

.section-divider {
    height:1px;
    background:linear-gradient(90deg,#3478F6,#895BFF,transparent);
    margin:13px 0 18px;
    opacity:.7;
}

.small-muted {
    font-size:11px;
    color:#8DA5C8;
}

.nav-caption {
    font-size:10px;
    font-weight:800;
    letter-spacing:1.7px;
    color:#6685AF;
    padding:10px 4px;
}

.info-strip {
    display:flex;
    align-items:center;
    gap:12px;
    background:linear-gradient(100deg,#10284A,#101F3C);
    border:1px solid #25466E;
    border-radius:13px;
    padding:13px 15px;
    margin:8px 0 18px;
    transition:all .25s ease;
}

.info-strip:hover {
    border-color:#438BFF;
    transform:translateY(-2px);
    box-shadow:0 8px 22px #0002;
}

.info-strip-icon {
    display:flex;
    align-items:center;
    justify-content:center;
    min-width:36px;
    height:36px;
    border-radius:11px;
    background:#3478F622;
    color:#70AAFF;
    font-size:18px;
}

.info-strip-title {
    color:#E7F0FF;
    font-size:12px;
    font-weight:800;
    margin-bottom:3px;
}

.info-strip-text {
    color:#91A9CA;
    font-size:10px;
    line-height:1.6;
}

.mini-stat {
    background:#0B1B35;
    border:1px solid #1B365C;
    border-radius:12px;
    padding:12px;
    transition:all .25s ease;
}

.mini-stat:hover {
    transform:translateY(-3px);
    border-color:#3478F6;
    box-shadow:0 7px 20px #0002;
}

.mini-stat-label {
    color:#8DA5C8;
    font-size:10px;
    letter-spacing:.5px;
}

.mini-stat-value {
    color:#ECF4FF;
    font-size:19px;
    font-weight:800;
    margin-top:5px;
}

div[data-testid="stPlotlyChart"] {
    background:linear-gradient(145deg,#0D203D,#09172E);
    border:1px solid #1B365C;
    border-radius:15px;
    padding:5px;
    transition:border-color .25s,transform .25s,box-shadow .25s;
    animation:rise .4s ease both;
}

div[data-testid="stPlotlyChart"]:hover {
    border-color:#315F95;
    transform:translateY(-3px);
    box-shadow:0 10px 25px #0002;
}

div[data-testid="stMetric"] {
    background:linear-gradient(145deg,#0D203D,#09172E);
    border:1px solid #1B365C;
    border-radius:13px;
    padding:14px;
    transition:all .25s ease;
}

div[data-testid="stMetric"]:hover {
    border-color:#3478F6;
    transform:translateY(-2px);
    box-shadow:0 7px 20px #0002;
}

div[data-testid="stMetric"] label {
    color:#91A9CC!important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color:#F0F6FF;
    font-family:'Manrope',sans-serif;
    font-weight:800;
}

.stButton>button,
.stDownloadButton>button {
    background:#10284A;
    color:#DCEAFF;
    border:1px solid #284A79;
    border-radius:10px;
    min-height:42px;
    font-weight:700;
    transition:all .22s ease;
}

.stButton>button:hover,
.stDownloadButton>button:hover {
    background:#1B3E70;
    color:white;
    border-color:#4A91FF;
    transform:translateY(-3px);
    box-shadow:0 7px 20px #3478F626;
}

.stButton>button:active,
.stDownloadButton>button:active {
    transform:translateY(0) scale(.98);
}

.stButton>button[kind="primary"],
.stDownloadButton>button[kind="primary"] {
    background:linear-gradient(100deg,#3478F6,#7758FA);
    border:0;
    color:white;
}

[data-testid="stDataFrame"] {
    border:1px solid #1B365C;
    border-radius:12px;
    overflow:hidden;
}

[data-baseweb="select"]>div {
    background:#0C1D37;
    border-color:#29486F;
    border-radius:9px;
    transition:border-color .2s,box-shadow .2s;
}

[data-baseweb="select"]>div:focus-within {
    border-color:#438BFF;
    box-shadow:0 0 0 2px #3478F622;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input {
    background:#0C1D37;
    color:#EAF3FF;
    border-color:#29486F;
    border-radius:9px;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color:#438BFF;
    box-shadow:0 0 0 2px #3478F622;
}

[data-testid="stSlider"] [data-testid="stTickBar"] {
    background:#29486F;
}

[data-testid="stExpander"] {
    background:#0C1D37;
    border:1px solid #1B365C;
    border-radius:12px;
    transition:border-color .2s;
}

[data-testid="stExpander"]:hover {
    border-color:#3478F6;
}

div[data-testid="stAlert"] {
    border-radius:12px;
}

hr {
    border-color:#1B365C;
}

footer,#MainMenu {
    visibility:hidden;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap:7px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background:#0B1B35;
    border:1px solid transparent;
    border-radius:10px;
    padding:9px 11px;
    transition:all .22s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background:#142A4B;
    border-color:#2A4D7D;
    transform:translateX(4px);
    box-shadow:0 5px 15px #0002;
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background:linear-gradient(100deg,#244DB8,#3E4AD1);
    border-color:#548DFF;
    box-shadow:0 5px 18px #3478F630;
}

[data-testid="stSidebar"] [role="radiogroup"] label p {
    color:#DDEAFF;
    font-size:12px;
    font-weight:600;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    font-size:13px!important;
}

[data-testid="stSidebar"] [data-testid="stMultiSelect"] span {
    border-radius:7px;
}

div[data-testid="stVerticalBlock"] > div {
    animation:rise .25s ease both;
}

[data-testid="stTabs"] button {
    transition:all .2s ease;
}

[data-testid="stTabs"] button:hover {
    color:#70AAFF;
}

@media(max-width:850px) {
    .block-container {padding:1rem}
    .hero-title {font-size:27px}
    .metric-value {font-size:23px}
}

@media(prefers-reduced-motion:reduce) {
    *,*:before,*:after {
        animation:none!important;
        transition:none!important;
        scroll-behavior:auto!important;
    }
}
</style>
""", unsafe_allow_html=True)


def header(kicker, title, subtitle):
    st.markdown(
        f'<div class="eyebrow">{kicker}</div>'
        f'<div class="hero-title">{title}</div>'
        f'<div class="hero-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def section(title, subtitle=None):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f'<div class="section-sub">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def section_line():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def info_strip(title, description, icon="✦"):
    st.markdown(f"""
    <div class="info-strip">
        <div class="info-strip-icon">{icon}</div>
        <div>
            <div class="info-strip-title">{title}</div>
            <div class="info-strip-text">{description}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def mini_stat(col, label, value):
    with col:
        st.markdown(f"""
        <div class="mini-stat">
            <div class="mini-stat-label">{label}</div>
            <div class="mini-stat-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)


def metric(col, label, value, foot, color, icon="●"):
    with col:
        st.markdown(f"""
        <div class="metric-card" style="--accent:{color}">
            <div class="metric-top">
                <div class="metric-label">{label}</div>
                <div class="metric-icon">{icon}</div>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """, unsafe_allow_html=True)


def chart_style(fig, height=330):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#B7CAE5", size=11),
        margin=dict(l=12, r=12, t=25, b=12),
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#B7CAE5", size=10),
        ),
        hoverlabel=dict(
            bgcolor="#122746",
            bordercolor="#3478F6",
            font=dict(color="#FFFFFF"),
        ),
    )
    fig.update_xaxes(
        gridcolor="#1B3150",
        zeroline=False,
        linecolor="#294363",
    )
    fig.update_yaxes(
        gridcolor="#1B3150",
        zeroline=False,
        linecolor="#294363",
    )
    return fig


def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def parse_time(series):
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()

    if not valid.empty and len(valid) >= 0.9 * max(series.notna().sum(), 1):
        magnitude = valid.abs().median()
        unit = (
            "ns" if magnitude >= 1e17 else
            "us" if magnitude >= 1e14 else
            "ms" if magnitude >= 1e11 else
            "s"
        )
        return pd.to_datetime(numeric, unit=unit, errors="coerce")

    return pd.to_datetime(series, errors="coerce")


def load_data():
    data = read_csv(DATA / "cleaned_transport.csv")

    if data.empty:
        data = read_csv(DATA / "raw_transport.csv")

    if data.empty or "timestamp" not in data.columns:
        return pd.DataFrame()

    data["timestamp"] = parse_time(data["timestamp"])

    for column in [
        "latitude", "longitude", "delay_minutes",
        "passengers", "speed_kmh"
    ]:
        if column not in data.columns:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in ["route_id", "route_name"]:
        if column not in data.columns:
            data[column] = "Unknown"

    data = data.dropna(
        subset=["timestamp", "delay_minutes", "passengers", "speed_kmh"]
    ).copy()

    data["route_id"] = data["route_id"].astype(str)
    data["hour"] = data["timestamp"].dt.hour
    data["date"] = data["timestamp"].dt.date
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["month"] = data["timestamp"].dt.month

    return data


df = load_data()

st.sidebar.markdown("""
<div class="brand">
    <div class="brand-icon">🚌</div>
    <div>Transit<span>IQ</span></div>
</div>
<div class="brand-sub">SMARTER TRANSIT. BETTER CITIES.</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="panel" style="margin-bottom:18px;padding:12px">
    <div style="font-size:10px;color:#819FC7;letter-spacing:1.2px">
        COMMAND CENTRE
    </div>
    <div style="font-size:12px;color:#E5EEFF;font-weight:700;margin-top:6px">
        Urban Mobility Workspace
    </div>
    <div style="font-size:10px;color:#8199BB;margin-top:4px">
        Analyze · Monitor · Predict
    </div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.error(
        "Dataset missing or invalid. Check data/cleaned_transport.csv "
        "or data/raw_transport.csv."
    )
    st.stop()

st.sidebar.markdown(
    '<div class="nav-caption">WORKSPACE</div>',
    unsafe_allow_html=True,
)

pages = [
    "Overview",
    "Route Intelligence",
    "Congestion Analysis",
    "Delay Prediction",
    "Model Insights",
    "Data Explorer",
]

page_icons = {
    "Overview": "◈",
    "Route Intelligence": "⌁",
    "Congestion Analysis": "▦",
    "Delay Prediction": "◷",
    "Model Insights": "✳",
    "Data Explorer": "▤",
}

if "navigation_page" not in st.session_state:
    st.session_state["navigation_page"] = "Overview"

page = st.sidebar.radio(
    "Navigation",
    pages,
    key="navigation_page",
    format_func=lambda p: f"{page_icons[p]}    {p}",
    label_visibility="collapsed",
)

st.sidebar.divider()

st.sidebar.markdown(
    '<div class="nav-caption">DATA FILTERS</div>',
    unsafe_allow_html=True,
)

routes = sorted(df["route_id"].dropna().unique().tolist())

selected_routes = st.sidebar.multiselect(
    "Transport routes",
    routes,
    default=routes,
)

min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Analysis period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, (tuple, list)):
    start_date = date_range[0] if len(date_range) else min_date
    end_date = date_range[-1] if len(date_range) > 1 else start_date
else:
    start_date = end_date = date_range

filtered = df[
    df["route_id"].isin(selected_routes)
    & df["timestamp"].dt.date.between(start_date, end_date)
].copy()

st.sidebar.divider()

st.sidebar.markdown(f"""
<div class="panel">
    <div style="font-size:10px;color:#819FC7;letter-spacing:1px">
        SYSTEM STATUS
    </div>
    <div style="color:#14D6A0;font-size:13px;font-weight:700;margin-top:9px">
        <span class="status-dot" style="display:inline-block;margin-right:7px"></span>
        Data engine available
    </div>
    <div style="font-size:10px;color:#8DA5C8;margin-top:7px">
        Local dataset connection
    </div>
    <div style="height:1px;background:#1B365C;margin:12px 0"></div>
    <div style="display:flex;justify-content:space-between;gap:8px">
        <span style="font-size:10px;color:#8DA5C8">Source records</span>
        <span style="font-size:11px;color:#E5EEFF;font-weight:700">{len(df):,}</span>
    </div>
    <div style="display:flex;justify-content:space-between;gap:8px;margin-top:7px">
        <span style="font-size:10px;color:#8DA5C8">Selected records</span>
        <span style="font-size:11px;color:#58A2FF;font-weight:700">{len(filtered):,}</span>
    </div>
</div>
""", unsafe_allow_html=True)

if filtered.empty:
    st.warning("No records match your current filters.")
    st.stop()

filtered_hourly = (
    filtered.assign(hour_start=filtered["timestamp"].dt.floor("h"))
    .groupby(["route_id", "route_name", "hour_start"], as_index=False)
    .agg(
        avg_delay=("delay_minutes", "mean"),
        passenger_demand=("passengers", "sum"),
        avg_speed=("speed_kmh", "mean"),
        trip_records=("route_id", "size"),
    )
)

filtered_hourly["hour"] = filtered_hourly["hour_start"].dt.hour

route_summary = (
    filtered.groupby(["route_id", "route_name"], as_index=False)
    .agg(
        avg_delay=("delay_minutes", "mean"),
        total_passengers=("passengers", "sum"),
        avg_speed=("speed_kmh", "mean"),
        records=("route_id", "size"),
    )
)

top_slots = filtered_hourly.copy()

if not top_slots.empty:
    top_slots["delay_rank"] = top_slots["avg_delay"].rank(pct=True)
    top_slots["demand_rank"] = top_slots["passenger_demand"].rank(pct=True)
    top_slots["slow_rank"] = 1 - top_slots["avg_speed"].rank(pct=True)
    top_slots["congestion_score"] = 100 * (
        0.50 * top_slots["delay_rank"]
        + 0.30 * top_slots["demand_rank"]
        + 0.20 * top_slots["slow_rank"]
    )
    top_slots = top_slots.nlargest(3, "congestion_score")


if page == "Overview":

    header(
        "URBAN MOBILITY INTELLIGENCE",
        'Welcome to <span>TransitIQ</span>',
        "Transport performance, route intelligence and predictive insights in one place.",
    )

    st.markdown("""
    <div class="topline">
        <div class="small-muted">YOUR CITY · MOBILITY OVERVIEW</div>
        <div class="status-pill">
            <span class="status-dot"></span> DATA CONNECTED
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)

    metric(
        cols[0], "TRIP RECORDS", f"{len(filtered):,}",
        "Records in selected period", BLUE, "▣"
    )
    metric(
        cols[1], "TOTAL PASSENGERS",
        f"{filtered['passengers'].sum():,.0f}",
        "Recorded passenger volume", PURPLE, "♟"
    )
    metric(
        cols[2], "AVERAGE DELAY",
        f"{filtered['delay_minutes'].mean():.1f} min",
        "Mean observed delay", GREEN, "◷"
    )
    metric(
        cols[3], "ACTIVE ROUTES",
        f"{filtered['route_id'].nunique()}",
        "Routes in current selection", ORANGE, "⌁"
    )

    st.write("")

    info_strip(
        "Network intelligence is active",
        f"Analyzing {len(filtered):,} transport records across "
        f"{filtered['route_id'].nunique()} routes for the selected period.",
        "⌖",
    )

    map_col, side_col = st.columns([1.85, 1], gap="medium")

    with map_col:
        section(
            "⌖  Transport network",
            "Recorded GPS points and route connections"
        )

        gps = filtered.dropna(subset=["latitude", "longitude"]).copy()

        if gps.empty:
            st.info("GPS coordinates are unavailable in this dataset.")
        else:
            gps = gps[
                gps["latitude"].between(-90, 90)
                & gps["longitude"].between(-180, 180)
            ]

            if gps.empty:
                st.info("No valid GPS coordinates are available.")
            else:
                m = folium.Map(
                    location=[
                        gps["latitude"].median(),
                        gps["longitude"].median()
                    ],
                    zoom_start=12,
                    tiles="CartoDB dark_matter",
                    control_scale=True,
                )

                for i, (route, group) in enumerate(gps.groupby("route_id")):
                    group = group.sort_values("timestamp")
                    color = PALETTE[i % len(PALETTE)]
                    points = group[["latitude", "longitude"]].values.tolist()

                    if len(points) > 1:
                        folium.PolyLine(
                            points,
                            color=color,
                            weight=3,
                            opacity=0.85,
                            tooltip=f"Route {route}",
                        ).add_to(m)

                    step = max(1, len(group) // 20)

                    for _, row in group.iloc[::step].iterrows():
                        folium.CircleMarker(
                            [row.latitude, row.longitude],
                            radius=4,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.9,
                            tooltip=(
                                f"Route: {route} | "
                                f"Delay: {row.delay_minutes:.1f} min"
                            ),
                        ).add_to(m)

                st_folium(
                    m,
                    use_container_width=True,
                    height=450,
                    key="transitiq_dark_map",
                )

                gps_a, gps_b, gps_c = st.columns(3)
                mini_stat(gps_a, "GPS OBSERVATIONS", f"{len(gps):,}")
                mini_stat(gps_b, "MAPPED ROUTES", f"{gps['route_id'].nunique():,}")
                mini_stat(gps_c, "UNIQUE GPS POINTS", f"{gps[['latitude', 'longitude']].drop_duplicates().shape[0]:,}")

    with side_col:
        section("Route performance", "Ranked by average delay")

        ranking = route_summary.sort_values(
            "avg_delay", ascending=False
        ).head(7)

        if ranking.empty:
            st.info("No route performance data available.")
        else:
            for _, row in ranking.iterrows():
                delay = max(0, float(row["avg_delay"]))
                level = min(100, delay * 4)
                color = RED if delay >= 15 else ORANGE if delay >= 5 else GREEN

                st.markdown(f"""
                <div style="padding:12px 0;border-bottom:1px solid #1B365C">
                    <div style="display:flex;justify-content:space-between;gap:8px">
                        <span style="font-size:12px;font-weight:700;color:#DCE9FF">
                            ◈ Route {row['route_id']}
                        </span>
                        <span style="font-size:12px;color:{color};font-weight:700">
                            {delay:.1f} min
                        </span>
                    </div>
                    <div style="height:6px;background:#182D4C;border-radius:8px;margin-top:9px;overflow:hidden">
                        <div style="height:100%;width:{level}%;background:{color};border-radius:8px;transition:width .5s ease"></div>
                    </div>
                    <div style="font-size:10px;color:#8199BB;margin-top:5px">
                        {row['avg_speed']:.1f} km/h average speed
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.write("")
        section("Recent observations", "Highest recorded delays")

        alerts = filtered.nlargest(4, "delay_minutes")

        for _, row in alerts.iterrows():
            delay = float(row["delay_minutes"])
            color = RED if delay >= 15 else ORANGE if delay >= 5 else GREEN

            st.markdown(f"""
            <div style="display:flex;gap:10px;padding:10px 0;border-bottom:1px solid #1B365C">
                <div style="font-size:18px;color:{color}">●</div>
                <div style="flex:1">
                    <div style="font-size:11px;font-weight:700;color:#E5EEFF">
                        Route {row['route_id']} · {delay:.1f} min delay
                    </div>
                    <div style="font-size:10px;color:#8199BB;margin-top:4px">
                        {row['timestamp']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.3, 1], gap="medium")

    with left:
        section(
            "Passenger flow",
            "Recorded passenger count by hour"
        )

        demand = filtered.groupby("hour", as_index=False).agg(
            passengers=("passengers", "sum")
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=demand["hour"],
            y=demand["passengers"],
            mode="lines+markers",
            name="Passengers",
            line=dict(color=GREEN, width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(20,214,160,.10)",
            marker=dict(size=5),
        ))

        st.plotly_chart(
            chart_style(fig, 300),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        section(
            "Top congested routes",
            "Composite congestion score"
        )

        congestion = (
            filtered_hourly.groupby("route_id", as_index=False)
            .agg(
                avg_delay=("avg_delay", "mean"),
                demand=("passenger_demand", "sum"),
                speed=("avg_speed", "mean"),
            )
        )

        congestion["score"] = (
            50 * congestion["avg_delay"].rank(pct=True)
            + 30 * congestion["demand"].rank(pct=True)
            + 20 * (1 - congestion["speed"].rank(pct=True))
        )

        congestion = congestion.nlargest(5, "score")

        fig = px.bar(
            congestion.sort_values("score"),
            x="score",
            y="route_id",
            orientation="h",
            color="score",
            color_continuous_scale=["#14D6A0", "#3478F6", "#FF5369"],
            labels={"score": "Score", "route_id": "Route"},
        )
        fig.update_layout(coloraxis_showscale=False)

        st.plotly_chart(
            chart_style(fig, 300),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    
    st.write("")
    section("Quick actions", "Jump directly into a workspace")

    quick_cols = st.columns(4)

    quick_actions = [
        ("⌁  Route intelligence", "Route Intelligence"),
        ("▦  Congestion analysis", "Congestion Analysis"),
        ("◷  Predict delays", "Delay Prediction"),
        ("▤  Explore data", "Data Explorer"),
    ]

    def navigate_to(target):
        st.session_state["navigation_page"] = target

    for col, (label, target) in zip(quick_cols, quick_actions):
        with col:
            st.button(
                label,
                use_container_width=True,
                on_click=navigate_to,
                args=(target,),
            )






elif page == "Route Intelligence":

    header(
        "ROUTE OPERATIONS / 02",
        "Route intelligence",
        "Investigate individual routes and compare their operating performance.",
    )

    info_strip(
        "Route-level diagnostics",
        "Select a route to inspect its delay trend, operating speed, "
        "passenger activity and individual observations.",
        "⌁",
    )

    route = st.selectbox(
        "Select transport route",
        sorted(filtered["route_id"].unique()),
    )

    rd = filtered[
        filtered["route_id"] == route
    ].sort_values("timestamp")

    if not rd.empty:
        cols = st.columns(4)

        metric(
            cols[0], "AVERAGE DELAY",
            f"{rd.delay_minutes.mean():.1f} min",
            "Recorded delay", ORANGE, "◷"
        )
        metric(
            cols[1], "AVERAGE SPEED",
            f"{rd.speed_kmh.mean():.1f}",
            "km/h", CYAN, "↗"
        )
        metric(
            cols[2], "PASSENGERS",
            f"{rd.passengers.sum():,.0f}",
            "Recorded total", GREEN, "♟"
        )
        metric(
            cols[3], "OBSERVATIONS",
            f"{len(rd):,}",
            "Selected route records", PURPLE, "▣"
        )

        st.write("")

        stat_a, stat_b, stat_c = st.columns(3)
        mini_stat(stat_a, "MAXIMUM OBSERVED DELAY", f"{rd.delay_minutes.max():.1f} min")
        mini_stat(stat_b, "MAXIMUM RECORDED SPEED", f"{rd.speed_kmh.max():.1f} km/h")
        mini_stat(stat_c, "DATE RANGE", f"{rd.timestamp.min().date()} → {rd.timestamp.max().date()}")

        st.write("")
        section(
            "Route performance timeline",
            "Hourly delay and speed trends"
        )

        rt = (
            rd.assign(time=rd["timestamp"].dt.floor("h"))
            .groupby("time", as_index=False)
            .agg(
                delay=("delay_minutes", "mean"),
                speed=("speed_kmh", "mean"),
            )
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=rt.time,
            y=rt.delay,
            name="Delay",
            mode="lines+markers",
            line=dict(color=ORANGE, width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(255,157,82,.08)",
        ))

        fig.add_trace(go.Scatter(
            x=rt.time,
            y=rt.speed,
            name="Speed",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color=CYAN, width=3, shape="spline"),
        ))

        fig.update_layout(
            yaxis=dict(title="Delay (min)"),
            yaxis2=dict(
                title="Speed (km/h)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )

        st.plotly_chart(
            chart_style(fig, 380),
            use_container_width=True,
        )

        left, right = st.columns(2)

        with left:
            section("Delay distribution", "Frequency of recorded delay values")

            fig = px.histogram(
                rd,
                x="delay_minutes",
                nbins=25,
                labels={"delay_minutes": "Delay (minutes)"},
            )
            fig.update_traces(
                marker_color=PURPLE,
                marker_line_color="#B7A4FF",
                marker_line_width=.5,
            )

            st.plotly_chart(
                chart_style(fig, 300),
                use_container_width=True,
            )

        with right:
            section("Passenger activity", "Total recorded passengers by hour")

            activity = rd.groupby("hour", as_index=False).agg(
                passengers=("passengers", "sum")
            )

            fig = px.bar(
                activity,
                x="hour",
                y="passengers",
                labels={
                    "hour": "Hour",
                    "passengers": "Passengers",
                },
            )
            fig.update_traces(
                marker_color=GREEN,
                marker_line_color="#70F5D1",
                marker_line_width=.5,
            )

            st.plotly_chart(
                chart_style(fig, 300),
                use_container_width=True,
            )

        section("Route observations", "Inspect individual records for this route")

        st.dataframe(
            rd.sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download route CSV",
            rd.to_csv(index=False).encode("utf-8"),
            file_name=f"{route}_records.csv",
            mime="text/csv",
        )


elif page == "Congestion Analysis":

    header(
        "TRAFFIC OPERATIONS / 03",
        "Congestion analysis",
        "Find pressure points using recorded delays, passenger demand and speed.",
    )

    info_strip(
        "How congestion is ranked",
        "The composite score combines delay (50%), passenger demand (30%) "
        "and lower average speed (20%). It is a relative indicator, "
        "not an official traffic measurement.",
        "▦",
    )

    busiest = int(
        filtered.groupby("hour")["passengers"].sum().idxmax()
    )

    cols = st.columns(4)

    metric(
        cols[0], "AVERAGE DELAY",
        f"{filtered.delay_minutes.mean():.1f} min",
        "All selected records", ORANGE, "◷"
    )
    metric(
        cols[1], "AVERAGE SPEED",
        f"{filtered.speed_kmh.mean():.1f} km/h",
        "Recorded operating speed", CYAN, "↗"
    )
    metric(
        cols[2], "BUSIEST HOUR",
        f"{busiest:02d}:00",
        "Highest passenger total", PURPLE, "◉"
    )
    metric(
        cols[3], "PRIORITY SLOTS",
        str(len(top_slots)),
        "Highest composite scores", RED, "⚑"
    )

    congestion = (
        filtered_hourly.groupby("route_id", as_index=False)
        .agg(
            avg_delay=("avg_delay", "mean"),
            demand=("passenger_demand", "sum"),
            speed=("avg_speed", "mean"),
        )
    )

    congestion["delay_rank"] = congestion.avg_delay.rank(pct=True)
    congestion["demand_rank"] = congestion.demand.rank(pct=True)
    congestion["slow_rank"] = 1 - congestion.speed.rank(pct=True)

    congestion["score"] = 100 * (
        .5 * congestion.delay_rank
        + .3 * congestion.demand_rank
        + .2 * congestion.slow_rank
    )

    left, right = st.columns(2)

    with left:
        section("Congestion score by route", "Higher scores indicate higher relative priority")

        fig = px.bar(
            congestion.sort_values("score"),
            x="score",
            y="route_id",
            orientation="h",
            color="score",
            color_continuous_scale=["#14D6A0", "#3478F6", "#FF5369"],
            labels={
                "score": "Score",
                "route_id": "Route",
            },
        )
        fig.update_layout(coloraxis_showscale=False)

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True,
        )

    with right:
        section("Demand versus delay", "Each point represents a route-hour group")

        fig = px.scatter(
            filtered_hourly,
            x="passenger_demand",
            y="avg_delay",
            color="route_id",
            size="trip_records",
            hover_data=["hour_start", "avg_speed"],
            labels={
                "passenger_demand": "Passengers",
                "avg_delay": "Delay (min)",
                "route_id": "Route",
            },
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True,
        )

    section("Congestion by hour", "Average delay and speed across the day")

    hourly_stats = filtered.groupby("hour", as_index=False).agg(
        delay=("delay_minutes", "mean"),
        speed=("speed_kmh", "mean"),
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=hourly_stats.hour,
        y=hourly_stats.delay,
        name="Delay",
        marker_color=ORANGE,
    ))

    fig.add_trace(go.Scatter(
        x=hourly_stats.hour,
        y=hourly_stats.speed,
        name="Speed",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color=CYAN, width=3),
    ))

    fig.update_layout(
        yaxis=dict(title="Delay (min)"),
        yaxis2=dict(
            title="Speed (km/h)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
    )

    st.plotly_chart(
        chart_style(fig, 360),
        use_container_width=True,
    )

    section("Highest congestion windows", "Top route-hour periods based on the composite score")

    if not top_slots.empty:
        st.dataframe(
            top_slots.round(2),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No congestion windows are available for this selection.")

    st.download_button(
        "Download congestion report",
        top_slots.to_csv(index=False).encode("utf-8"),
        file_name="congestion_analysis.csv",
        mime="text/csv",
    )


elif page == "Delay Prediction":

    header(
        "PREDICTIVE INTELLIGENCE / 04",
        "Delay forecast studio",
        "Estimate average route delay with your trained machine-learning model.",
    )

    info_strip(
        "Scenario-based forecasting",
        "Choose a route, hour, peak-hour condition, day and month. "
        "The existing trained model estimates delay from these inputs.",
        "◷",
    )

    if not MODEL.exists():
        st.error(f"Model file not found: {MODEL}")
        st.info("Run python train_model.py first.")
        st.stop()

    try:
        artifact = joblib.load(MODEL)

        if not isinstance(artifact, dict) or "model" not in artifact:
            st.error("Unexpected model format. Run train_model.py again.")
            st.stop()

        model = artifact["model"]
        features = artifact.get("features", [])

        expected = [
            "route_id",
            "hour",
            "is_peak_hour",
            "day_of_week",
            "month",
        ]

        if list(features) != expected:
            st.error(f"Model features do not match: {features}")
            st.stop()

        left, right = st.columns([1.15, .85], gap="large")

        with left:
            section("01 / Configure scenario", "Set the conditions for your forecast")

            with st.container(border=True):
                route = st.selectbox(
                    "Transport route",
                    sorted(df.route_id.astype(str).unique()),
                    key="predict_route",
                )

                hour = st.slider(
                    "Hour of day",
                    0,
                    23,
                    8,
                )

                peak = st.checkbox(
                    "Peak-hour conditions",
                    value=hour in [7, 8, 9, 16, 17, 18, 19],
                )

                day_names = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]

                day = st.selectbox(
                    "Day of week",
                    range(7),
                    format_func=lambda x: day_names[x],
                )

                month_names = [
                    "January", "February", "March", "April",
                    "May", "June", "July", "August",
                    "September", "October", "November", "December",
                ]

                month = st.selectbox(
                    "Month",
                    range(1, 13),
                    format_func=lambda x: month_names[x - 1],
                    index=5,
                )

                run_prediction = st.button(
                    "Generate delay forecast",
                    type="primary",
                    use_container_width=True,
                )

        with right:
            section("Scenario preview", "Review the selected conditions")

            st.markdown(f"""
            <div class="panel" style="min-height:245px">
                <div class="eyebrow">SELECTED SCENARIO</div>
                <div style="font-size:29px;font-weight:800;color:#EDF4FF;margin-top:15px">
                    Route {route}
                </div>
                <div style="color:#8DA5C8;margin-top:8px">
                    {day_names[day]} · {month_names[month-1]}
                </div>
                <div style="font-size:35px;font-weight:800;color:#58A2FF;margin-top:17px">
                    {hour:02d}:00
                </div>
                <div style="margin-top:12px;color:{ORANGE if peak else GREEN}">
                    ● {"Peak period" if peak else "Non-peak period"}
                </div>
                <div style="height:1px;background:#1B365C;margin:16px 0"></div>
                <div style="font-size:10px;color:#8DA5C8">
                    MODEL INPUTS · 5 FEATURES
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.caption(
                "Forecasts are model estimates, not live traffic data."
            )

        if run_prediction:
            input_data = pd.DataFrame([{
                "route_id": str(route),
                "hour": int(hour),
                "is_peak_hour": int(peak),
                "day_of_week": int(day),
                "month": int(month),
            }], columns=features)

            predicted = max(
                0.0,
                float(np.asarray(model.predict(input_data)).reshape(-1)[0]),
            )

            if predicted >= 15:
                category, color = "High", RED
                recommendation = (
                    "Review congestion conditions and consider "
                    "additional service capacity."
                )
            elif predicted >= 5:
                category, color = "Moderate", ORANGE
                recommendation = (
                    "Monitor this time slot and compare nearby periods."
                )
            else:
                category, color = "Low", GREEN
                recommendation = (
                    "The model estimates a relatively low average delay."
                )

            st.write("")
            section("02 / Forecast result", "Model output for the selected scenario")

            cols = st.columns(3)

            cols[0].metric(
                "Predicted delay",
                f"{predicted:.1f} min",
            )
            cols[1].metric(
                "Delay classification",
                category,
            )
            cols[2].metric(
                "Peak conditions",
                "Enabled" if peak else "Disabled",
            )

            st.markdown(f"""
            <div class="panel" style="border-left:4px solid {color};margin-top:15px">
                <div class="eyebrow">FORECAST INTERPRETATION</div>
                <h2 style="color:{color}!important">{category} delay level</h2>
                <p>{recommendation}</p>
                <div style="background:#172B48;height:8px;border-radius:10px;overflow:hidden;margin-top:15px">
                    <div style="height:100%;width:{min(100, predicted / 30 * 100):.1f}%;background:{color};border-radius:10px"></div>
                </div>
                <div style="font-size:10px;color:#8DA5C8;margin-top:7px">
                    Visual scale capped at 30 minutes for display
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.session_state["last_prediction"] = {
                "route_id": route,
                "hour": hour,
                "peak": peak,
                "day": day_names[day],
                "month": month_names[month - 1],
                "predicted_delay": predicted,
                "classification": category,
            }

        st.write("")
        section("Model evaluation metrics", "Evaluation values stored in the model artifact")

        cols = st.columns(3)

        cols[0].metric(
            "Test MAE",
            f"{artifact['mae']:.2f} min"
            if artifact.get("mae") is not None else "N/A",
        )
        cols[1].metric(
            "Test RMSE",
            f"{artifact['rmse']:.2f} min"
            if artifact.get("rmse") is not None else "N/A",
        )
        cols[2].metric(
            "Test R²",
            f"{artifact['r2']:.3f}"
            if artifact.get("r2") is not None else "N/A",
        )

    except Exception as error:
        st.error(f"Could not run prediction: {error}")


elif page == "Model Insights":

    header(
        "MODEL INTELLIGENCE / 05",
        "Model insights",
        "Evaluate model quality, feature importance and prediction errors.",
    )

    info_strip(
        "Understand before you interpret",
        "MAE and RMSE describe prediction error in the target's units. "
        "R² describes fit relative to a baseline; none alone guarantees "
        "reliable real-world forecasts.",
        "✳",
    )

    if not MODEL.exists():
        st.error("Model file not found. Run train_model.py first.")
        st.stop()

    try:
        artifact = joblib.load(MODEL)
        model = artifact["model"]

        cols = st.columns(3)

        cols[0].metric(
            "Mean Absolute Error",
            f"{artifact['mae']:.2f} min"
            if artifact.get("mae") is not None else "N/A",
        )
        cols[1].metric(
            "Root Mean Squared Error",
            f"{artifact['rmse']:.2f} min"
            if artifact.get("rmse") is not None else "N/A",
        )
        cols[2].metric(
            "R² Score",
            f"{artifact['r2']:.3f}"
            if artifact.get("r2") is not None else "N/A",
        )

        st.caption(
            "Lower MAE and RMSE generally indicate smaller prediction errors. "
            "R² should be interpreted alongside the dataset and test setup."
        )

        section("Feature importance", "Inspect which model inputs contribute most")

        if (
            hasattr(model, "named_steps")
            and "preprocessor" in model.named_steps
            and "regressor" in model.named_steps
        ):
            try:
                names = model.named_steps["preprocessor"].get_feature_names_out()
                values = model.named_steps["regressor"].feature_importances_

                importance = pd.DataFrame({
                    "Feature": names,
                    "Importance": values,
                }).sort_values("Importance")

                fig = px.bar(
                    importance,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    labels={
                        "Feature": "Input feature",
                        "Importance": "Importance",
                    },
                )
                fig.update_traces(
                    marker_color=PURPLE,
                    marker_line_color="#B7A4FF",
                    marker_line_width=.5,
                )

                st.plotly_chart(
                    chart_style(fig, 400),
                    use_container_width=True,
                )

                st.dataframe(
                    importance.sort_values(
                        "Importance", ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as error:
                st.info(f"Feature importance unavailable: {error}")
        else:
            st.info(
                "This model format does not expose feature importance "
                "through the expected pipeline."
            )

        section("Actual versus predicted", "Compare model estimates against stored test records")

        predictions_path = BASE / "models" / "test_predictions.csv"

        if predictions_path.exists():
            results = pd.read_csv(predictions_path)

            required = {
                "timestamp",
                "Actual Delay",
                "Predicted Delay",
            }

            if required.issubset(results.columns):
                results["timestamp"] = pd.to_datetime(
                    results["timestamp"],
                    errors="coerce",
                )

                for column in ["Actual Delay", "Predicted Delay"]:
                    results[column] = pd.to_numeric(
                        results[column],
                        errors="coerce",
                    )

                results = results.dropna(
                    subset=[
                        "timestamp",
                        "Actual Delay",
                        "Predicted Delay",
                    ]
                ).sort_values("timestamp")

                results["Absolute Error"] = (
                    results["Actual Delay"] - results["Predicted Delay"]
                ).abs()

                cols = st.columns(3)

                cols[0].metric(
                    "Test records",
                    f"{len(results):,}",
                )
                cols[1].metric(
                    "Mean actual delay",
                    f"{results['Actual Delay'].mean():.1f} min",
                )
                cols[2].metric(
                    "Mean absolute error",
                    f"{results['Absolute Error'].mean():.2f} min",
                )

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=results.timestamp,
                    y=results["Actual Delay"],
                    name="Actual",
                    line=dict(color=CYAN, width=2),
                ))

                fig.add_trace(go.Scatter(
                    x=results.timestamp,
                    y=results["Predicted Delay"],
                    name="Predicted",
                    line=dict(color=ORANGE, width=2),
                ))

                st.plotly_chart(
                    chart_style(fig, 380),
                    use_container_width=True,
                )

                error_values = (
                    results["Predicted Delay"] - results["Actual Delay"]
                )

                fig = px.histogram(
                    x=error_values,
                    nbins=25,
                    labels={"x": "Prediction error (minutes)"},
                )

                fig.update_traces(marker_color=PURPLE)

                st.plotly_chart(
                    chart_style(fig, 300),
                    use_container_width=True,
                )

                with st.expander("View prediction records"):
                    st.dataframe(
                        results,
                        use_container_width=True,
                        hide_index=True,
                    )

                st.download_button(
                    "Download prediction analysis",
                    results.to_csv(index=False).encode("utf-8"),
                    file_name="transitiq_prediction_analysis.csv",
                    mime="text/csv",
                )

            else:
                st.warning(
                    "Prediction CSV is missing required columns."
                )
        else:
            st.info(
                "No test_predictions.csv found. Run train_model.py "
                "if your training script generates this file."
            )

        st.write("")
        left, right = st.columns(2)

        with left:
            st.markdown(f"""
            <div class="panel" style="min-height:110px">
                <div class="eyebrow">TRAINED UNTIL</div>
                <h3>{artifact.get("trained_until", "Not available")}</h3>
                <div class="small-muted">Training data cutoff stored in artifact</div>
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown(f"""
            <div class="panel" style="min-height:110px">
                <div class="eyebrow">TEST PERIOD START</div>
                <h3>{artifact.get("tested_from", "Not available")}</h3>
                <div class="small-muted">Test period metadata stored in artifact</div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as error:
        st.error(f"Could not load model insights: {error}")


elif page == "Data Explorer":

    header(
        "DATA WORKSPACE / 06",
        "Transport data explorer",
        "Search, inspect, validate and export the underlying transport records.",
    )

    info_strip(
        "Explore your dataset",
        "Filter the records, choose columns for the table and CSV export, "
        "and inspect missing values and column types.",
        "▤",
    )

    explorer = filtered.copy()

    cols = st.columns(4)

    metric(
        cols[0], "TOTAL RECORDS",
        f"{len(explorer):,}",
        "Filtered dataset", BLUE, "▤"
    )
    metric(
        cols[1], "UNIQUE ROUTES",
        f"{explorer.route_id.nunique():,}",
        "Route identifiers", PURPLE, "⌁"
    )
    metric(
        cols[2], "MISSING VALUES",
        f"{explorer.isna().sum().sum():,}",
        "Across selected columns", ORANGE, "!"
    )
    metric(
        cols[3], "DUPLICATE ROWS",
        f"{explorer.duplicated().sum():,}",
        "Exact duplicate records", RED, "▦"
    )

    st.write("")
    section("Search and filter", "Narrow the displayed records")

    left, right = st.columns([1.2, 1])

    with left:
        search = st.text_input(
            "Search route ID or route name",
            placeholder="Enter a route...",
        )

    delay_values = pd.to_numeric(
        explorer["delay_minutes"],
        errors="coerce",
    ).dropna()

    delay_range = None

    with right:
        if not delay_values.empty:
            low = float(delay_values.min())
            high = float(delay_values.max())

            if low < high:
                delay_range = st.slider(
                    "Delay range (minutes)",
                    low,
                    high,
                    (low, high),
                )
            else:
                delay_range = (low, high)

    if search:
        mask = (
            explorer["route_id"].astype(str).str.contains(
                search,
                case=False,
                na=False,
                regex=False,
            )
            | explorer["route_name"].astype(str).str.contains(
                search,
                case=False,
                na=False,
                regex=False,
            )
        )

        explorer = explorer[mask]

    if delay_range:
        explorer = explorer[
            pd.to_numeric(
                explorer["delay_minutes"],
                errors="coerce",
            ).between(delay_range[0], delay_range[1])
        ]

    section(
        "Dataset preview",
        f"{len(explorer):,} matching records",
    )

    columns = explorer.columns.tolist()

    default_columns = [
        col for col in [
            "timestamp",
            "route_id",
            "route_name",
            "delay_minutes",
            "passengers",
            "speed_kmh",
            "latitude",
            "longitude",
        ] if col in columns
    ]

    selected_columns = st.multiselect(
        "Columns to display and export",
        columns,
        default=default_columns or columns[:8],
    )

    if not explorer.empty and selected_columns:
        display = explorer[selected_columns].copy()

        if "timestamp" in display.columns:
            display = display.sort_values(
                "timestamp",
                ascending=False,
                na_position="last",
            )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=420,
        )

        st.download_button(
            "Download selected data as CSV",
            display.to_csv(index=False).encode("utf-8"),
            file_name="transitiq_transport_export.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

    elif explorer.empty:
        st.info("No records match your current filters.")

    else:
        st.warning("Select at least one column.")

    st.write("")
    left, right = st.columns(2)

    with left:
        section(
            "Missing values by column",
            "Columns with incomplete values in the filtered data",
        )

        missing = (
            explorer.isna().sum()
            .rename("Missing Values")
            .reset_index()
        )
        missing.columns = ["Column", "Missing Values"]
        missing = missing[missing["Missing Values"] > 0]

        if missing.empty:
            st.success("No missing values in the filtered data.")
        else:
            st.dataframe(
                missing,
                use_container_width=True,
                hide_index=True,
            )

    with right:
        section(
            "Column information",
            "Data types and distinct non-null values",
        )

        types = pd.DataFrame({
            "Column": explorer.columns,
            "Data Type": [
                str(explorer[col].dtype)
                for col in explorer.columns
            ],
            "Unique Values": [
                explorer[col].nunique(dropna=True)
                for col in explorer.columns
            ],
        })

        st.dataframe(
            types,
            use_container_width=True,
            hide_index=True,
        )

    st.write("")
    section("Data quality summary")

    quality_cols = st.columns(3)

    total_cells = explorer.shape[0] * explorer.shape[1]
    missing_cells = int(explorer.isna().sum().sum())
    completeness = (
        100 * (1 - missing_cells / total_cells)
        if total_cells else 0
    )

    mini_stat(
        quality_cols[0],
        "DATA COMPLETENESS",
        f"{completeness:.1f}%",
    )
    mini_stat(
        quality_cols[1],
        "TOTAL COLUMNS",
        f"{explorer.shape[1]:,}",
    )
    mini_stat(
        quality_cols[2],
        "DUPLICATE ROWS",
        f"{explorer.duplicated().sum():,}",
    )

    st.caption(
        "Exports do not modify the source dataset. Statistics reflect "
        "the currently selected filters."
    )