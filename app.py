
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = BASE_DIR / "models" / "delay_prediction_model.joblib"

CLEAN_FILE = DATA_DIR / "cleaned_transport.csv"
RAW_FILE = DATA_DIR / "raw_transport.csv"
HOURLY_FILE = OUTPUT_DIR / "hourly_transport_summary.csv"
TOP_FILE = OUTPUT_DIR / "top_3_congestion.csv"

st.set_page_config(
    page_title="TransitIQ | Smart Mobility",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)


NAVY = "#080E19"
PANEL = "#101B2B"
TEAL = "#43E6C1"
AMBER = "#FFBF69"
MUTED = "#91A2B7"
WHITE = "#F1F5F9"
RED = "#FB7185"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {{
        color-scheme: dark;
        --bg: {NAVY};
        --panel: {PANEL};
        --accent: {TEAL};
        --muted: {MUTED};
        --border: #26374A;
    }}

    .stApp {{
        background:
            radial-gradient(ellipse at 85% 0%, rgba(67,230,193,.07), transparent 35%),
            {NAVY};
        color: {WHITE};
        font-family: 'DM Sans', sans-serif;
    }}

    [data-testid="stHeader"] {{
        background: rgba(8,14,25,.86);
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0D1726, #09111D);
        border-right: 1px solid var(--border);
    }}

    [data-testid="stSidebar"] > div {{
        padding-top: 1.5rem;
    }}

    [data-testid="stSidebar"] * {{
        color: {WHITE};
    }}

    .block-container {{
        max-width: 1550px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    h1, h2, h3, h4 {{
        color: {WHITE} !important;
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -.4px;
    }}

    p, label, li {{
        color: #D5DFEA;
    }}

    .brand {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 29px;
        line-height: 1.25;
        font-weight: 700;
        letter-spacing: -1.3px;
        color: {WHITE};
    }}

    .brand span {{
        color: {TEAL};
    }}

    .eyebrow {{
        color: {TEAL};
        font-size: 10px;
        letter-spacing: 2.2px;
        font-weight: 700;
        text-transform: uppercase;
    }}

    .subtle, [data-testid="stCaptionContainer"] {{
        color: {MUTED} !important;
        font-size: 13px;
    }}

    .section-label {{
        color: {WHITE};
        font-family: 'Space Grotesk', sans-serif;
        font-size: 20px;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 5px;
    }}

    .metric-card {{
        background: linear-gradient(145deg, #17263A, #101A29);
        border: 1px solid #293B50;
        border-radius: 17px;
        padding: 21px;
        min-height: 137px;
        box-shadow: 0 8px 24px rgba(0,0,0,.12);
        transition: border-color .2s ease, transform .2s ease;
    }}

    .metric-card:hover {{
        border-color: {TEAL};
        transform: translateY(-2px);
    }}

    .metric-label {{
        color: {MUTED};
        font-size: 11px;
        letter-spacing: 1.1px;
        font-weight: 600;
        margin-bottom: 13px;
    }}

    .metric-value {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(23px, 2vw, 31px);
        font-weight: 700;
        color: {WHITE};
        line-height: 1.25;
        overflow-wrap: anywhere;
    }}

    .metric-foot {{
        color: {TEAL};
        font-size: 11px;
        margin-top: 11px;
    }}

    .panel {{
        background: {PANEL};
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 20px;
    }}

    .insight {{
        background: linear-gradient(135deg, #142B2D, #102029);
        border: 1px solid #28524E;
        border-radius: 15px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 6px 20px rgba(0,0,0,.10);
    }}

    div[data-testid="stMetric"] {{
        background: linear-gradient(145deg, #142337, #101B2B);
        border: 1px solid #293B50;
        border-radius: 15px;
        padding: 17px 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,.10);
    }}

    div[data-testid="stMetric"] label {{
        color: {MUTED} !important;
    }}

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {WHITE};
        font-family: 'Space Grotesk', sans-serif;
    }}

    .stButton > button,
    .stDownloadButton > button {{
        border: 1px solid #315065;
        border-radius: 11px;
        background: #142639;
        color: {WHITE};
        font-weight: 600;
        min-height: 43px;
        transition: all .2s ease;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        border-color: {TEAL};
        color: {TEAL};
        background: #19313C;
    }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(100deg, #1B9E89, #43E6C1);
        border: none;
        color: #07151A;
        font-weight: 700;
    }}

    .stButton > button[kind="primary"]:hover {{
        background: {TEAL};
        color: #07151A;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid var(--border);
        border-radius: 13px;
        overflow: hidden;
    }}

    [data-testid="stPlotlyChart"] {{
        background: rgba(16,27,43,.58);
        border: 1px solid #223247;
        border-radius: 15px;
        padding: 8px 8px 0;
    }}

    [data-testid="stSelectbox"] > div > div,
    [data-testid="stMultiSelect"] > div > div,
    [data-testid="stDateInput"] > div > div,
    [data-testid="stTextInput"] > div > div {{
        background: #111E2E;
        border-color: #304257;
        border-radius: 10px;
    }}

    [data-testid="stSlider"] [data-baseweb="slider"] div {{
        border-radius: 10px;
    }}

    [data-testid="stAlert"] {{
        border-radius: 12px;
    }}

    hr {{
        border-color: #26374A;
    }}

    #MainMenu, footer {{
        visibility: hidden;
    }}

    @media (max-width: 768px) {{
        .block-container {{
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }}

        .brand {{
            font-size: 24px;
        }}

        .metric-card {{
            padding: 15px;
            min-height: 115px;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)



def parse_timestamp(series):
    values = series.copy()
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()

    if not valid.empty and len(valid) >= 0.9 * max(values.notna().sum(), 1):
        magnitude = valid.abs().median()

        if magnitude >= 1e17:
            unit = "ns"
        elif magnitude >= 1e14:
            unit = "us"
        elif magnitude >= 1e11:
            unit = "ms"
        else:
            unit = "s"

        return pd.to_datetime(numeric, unit=unit, errors="coerce")

    return pd.to_datetime(values, errors="coerce")


def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_datasets():
    clean = load_csv(CLEAN_FILE)

    if clean.empty:
        clean = load_csv(RAW_FILE)

    hourly = load_csv(HOURLY_FILE)
    top = load_csv(TOP_FILE)

    if not clean.empty:
        if "timestamp" not in clean.columns:
            return pd.DataFrame(), hourly, top

        clean["timestamp"] = parse_timestamp(clean["timestamp"])

        for col in ["latitude", "longitude"]:
            if col not in clean.columns:
                clean[col] = np.nan
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

        for col in ["route_id", "route_name"]:
            if col not in clean.columns:
                clean[col] = "Unknown"

        for col in ["delay_minutes", "passengers", "speed_kmh"]:
            if col not in clean.columns:
                clean[col] = np.nan
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

        clean = clean.dropna(
            subset=[
                "timestamp",
                "delay_minutes",
                "passengers",
                "speed_kmh",
            ]
        )

        clean["hour"] = clean["timestamp"].dt.hour
        clean["date"] = clean["timestamp"].dt.date
        clean["day_name"] = clean["timestamp"].dt.day_name()

    if not hourly.empty and "timestamp" in hourly.columns:
        hourly["timestamp"] = parse_timestamp(hourly["timestamp"])

        for col in ["avg_delay", "passenger_demand", "avg_speed"]:
            if col in hourly.columns:
                hourly[col] = pd.to_numeric(
                    hourly[col], errors="coerce"
                )

    if not top.empty and "timestamp" in top.columns:
        top["timestamp"] = parse_timestamp(top["timestamp"])

    return clean, hourly, top


df, hourly, top3 = load_datasets()

if df.empty:
    st.error(
        "Dataset not found or required data is invalid. "
        "Run python generate_data.py and python data_processing.py first."
    )
    st.stop()

st.sidebar.markdown(
    '<div class="brand">Transit<span>IQ</span></div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("SMART MOBILITY INTELLIGENCE")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Route Intelligence",
        "Congestion Analysis",
        "Delay Prediction",
        "Model Insights",
        "Data Explorer",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("### Filters")

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
    if len(date_range) == 2:
        start_date, end_date = date_range
    elif len(date_range) == 1:
        start_date = end_date = date_range[0]
    else:
        start_date, end_date = min_date, max_date
else:
    start_date = end_date = date_range

filtered = df[
    df["route_id"].isin(selected_routes)
    & df["timestamp"].dt.date.between(start_date, end_date)
].copy()

if filtered.empty:
    st.warning("No records match these filters.")
    st.stop()

filtered_hourly = (
    filtered.assign(
        hour_start=filtered["timestamp"].dt.floor("h")
    )
    .groupby(["route_id", "route_name", "hour_start"])
    .agg(
        avg_delay=("delay_minutes", "mean"),
        passenger_demand=("passengers", "sum"),
        avg_speed=("speed_kmh", "mean"),
        trip_records=("route_id", "size"),
    )
    .reset_index()
)

filtered_hourly["hour"] = filtered_hourly["hour_start"].dt.hour

route_summary = (
    filtered.groupby(["route_id", "route_name"])
    .agg(
        avg_delay=("delay_minutes", "mean"),
        total_passengers=("passengers", "sum"),
        avg_speed=("speed_kmh", "mean"),
        records=("route_id", "size"),
    )
    .reset_index()
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

    top_slots = top_slots.nlargest(3, "congestion_score").copy()

st.markdown(
    '<div class="eyebrow">URBAN MOBILITY ANALYTICS / 01</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="brand">City movement, <span>decoded.</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtle">A unified intelligence platform for route performance, '
    'passenger demand and congestion monitoring.</div>',
    unsafe_allow_html=True,
)

st.write("")

c1, c2, c3, c4 = st.columns(4)

metrics = [
    (c1, "TOTAL TRIP RECORDS", f"{len(filtered):,}", "Records in selected period"),
    (c2, "AVERAGE DELAY", f"{filtered['delay_minutes'].mean():.1f} min", "Mean recorded delay"),
    (c3, "PASSENGER OBSERVATIONS", f"{filtered['passengers'].sum():,.0f}", "Sum of recorded passenger counts"),
    (c4, "ACTIVE ROUTES", str(filtered["route_id"].nunique()), "Routes in current selection"),
]

for col, label, value, foot in metrics:
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-foot">{foot}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")


if page == "Overview":
    st.markdown(
        """
        <div class="eyebrow">CITY OPERATIONS / LIVE OVERVIEW</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:27px;
        font-weight:700;color:#F1F5F9;margin-top:5px;">
            Transport Performance
        </div>
        <div style="color:#91A2B7;font-size:13px;margin-bottom:18px;">
            Monitor route delays, passenger demand and congestion hotspots.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.6, 1])

    with left:
        st.markdown(
            '<div class="section-label">Delay performance</div>',
            unsafe_allow_html=True,
        )
        st.caption("Hourly average delay across the selected records")

        trend = (
            filtered.assign(
                time_bucket=filtered["timestamp"].dt.floor("h")
            )
            .groupby("time_bucket", as_index=False)
            .agg(
                avg_delay=("delay_minutes", "mean"),
                avg_speed=("speed_kmh", "mean"),
            )
        )

        fig = px.area(
            trend,
            x="time_bucket",
            y="avg_delay",
            template="plotly_dark",
        )
        fig.update_traces(
            line=dict(color=TEAL, width=2.5),
            fillcolor="rgba(67,230,193,0.12)",
            hovertemplate="%{x}<br>Average delay: %{y:.1f} min<extra></extra>",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=15, t=15, b=10),
            xaxis_title=None,
            yaxis_title="Delay (minutes)",
            height=340,
            hovermode="x unified",
            font=dict(family="DM Sans, sans-serif", color="#D5DFEA"),
        )
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(gridcolor="rgba(145,162,183,0.13)", zeroline=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown(
            '<div class="section-label">Route performance</div>',
            unsafe_allow_html=True,
        )
        st.caption("Routes ranked by average delay")

        route_chart = route_summary.sort_values(
            "avg_delay", ascending=True
        )

        fig = px.bar(
            route_chart,
            x="avg_delay",
            y="route_id",
            orientation="h",
            color="avg_delay",
            color_continuous_scale=[
                [0, "#43E6C1"],
                [0.55, "#FFBF69"],
                [1, "#FB7185"],
            ],
            template="plotly_dark",
            hover_data=["route_name", "records"],
            labels={
                "avg_delay": "Average delay (min)",
                "route_id": "Route",
                "route_name": "Route name",
                "records": "Records",
            },
        )
        fig.update_traces(
            marker_line_width=0,
            hovertemplate=(
                "<b>%{y}</b><br>Average delay: %{x:.1f} min"
                "<extra></extra>"
            ),
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            margin=dict(l=5, r=12, t=15, b=10),
            xaxis_title="Average delay (min)",
            yaxis_title=None,
            height=340,
            font=dict(family="DM Sans, sans-serif", color="#D5DFEA"),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(145,162,183,0.13)")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown(
            '<div class="section-label">City route network</div>',
            unsafe_allow_html=True,
        )
        st.caption("GPS traces coloured by route")

        map_data = filtered.dropna(
            subset=["latitude", "longitude"]
        )

        if map_data.empty:
            st.info("GPS coordinates are unavailable for the selected records.")
        else:
            center = [
                map_data["latitude"].median(),
                map_data["longitude"].median(),
            ]

            route_map = folium.Map(
                location=center,
                zoom_start=13,
                tiles="CartoDB dark_matter",
                control_scale=True,
            )

            colors = [
                "#43E6C1",
                "#FFBF69",
                "#60A5FA",
                "#FB7185",
                "#C4B5FD",
                "#F9A8D4",
            ]

            for i, (route_id, group) in enumerate(
                map_data.groupby("route_id")
            ):
                group = group.sort_values("timestamp")
                color = colors[i % len(colors)]
                points = group[
                    ["latitude", "longitude"]
                ].to_numpy().tolist()

                if len(points) > 1:
                    folium.PolyLine(
                        points,
                        color=color,
                        weight=3,
                        opacity=0.85,
                        tooltip=f"{route_id} route trace",
                    ).add_to(route_map)

                sample = group.iloc[
                    ::max(1, len(group) // 20)
                ]

                for _, row in sample.iterrows():
                    folium.CircleMarker(
                        location=[
                            row["latitude"],
                            row["longitude"],
                        ],
                        radius=4,
                        color=color,
                        weight=1,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.85,
                        tooltip=(
                            f"{route_id} | Delay: "
                            f"{row['delay_minutes']:.1f} min"
                        ),
                    ).add_to(route_map)

            st_folium(
                route_map,
                use_container_width=True,
                height=440,
                key="overview_route_map",
            )

    with right:
        st.markdown(
            '<div class="section-label">Passenger demand</div>',
            unsafe_allow_html=True,
        )
        st.caption("Total recorded passengers by hour of day")

        demand = (
            filtered.groupby("hour", as_index=False)
            .agg(passengers=("passengers", "sum"))
            .sort_values("hour")
        )

        fig = px.area(
            demand,
            x="hour",
            y="passengers",
            template="plotly_dark",
            labels={
                "hour": "Hour of day",
                "passengers": "Passenger count",
            },
        )
        fig.update_traces(
            line=dict(color=TEAL, width=2.5),
            fillcolor="rgba(67,230,193,0.14)",
            hovertemplate="Hour %{x}<br>Passengers: %{y:,.0f}<extra></extra>",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=15, b=10),
            xaxis_title="Hour of day",
            yaxis_title="Passenger count",
            height=440,
            font=dict(family="DM Sans, sans-serif", color="#D5DFEA"),
        )
        fig.update_xaxes(
            tickmode="linear",
            dtick=2,
            showgrid=False,
        )
        fig.update_yaxes(
            gridcolor="rgba(145,162,183,0.13)",
            rangemode="tozero",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(
        '<div class="section-label">Priority congestion slots</div>',
        unsafe_allow_html=True,
    )
    st.caption("Time slots that may need operational attention")

    if top_slots.empty:
        st.info("No congestion records available for the selected filters.")
    else:
        display = top_slots[
            [
                "route_id",
                "hour_start",
                "avg_delay",
                "passenger_demand",
                "avg_speed",
                "congestion_score",
            ]
        ].rename(
            columns={
                "route_id": "Route",
                "hour_start": "Time slot",
                "avg_delay": "Avg delay (min)",
                "passenger_demand": "Passenger count",
                "avg_speed": "Avg speed (km/h)",
                "congestion_score": "Congestion score",
            }
        )

        st.dataframe(
            display.round(2),
            use_container_width=True,
            hide_index=True,
            height=min(400, 40 + 36 * len(display)),
            column_config={
                "Avg delay (min)": st.column_config.NumberColumn(
                    format="%.1f min"
                ),
                "Avg speed (km/h)": st.column_config.NumberColumn(
                    format="%.1f km/h"
                ),
                "Congestion score": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Passenger count": st.column_config.NumberColumn(
                    format="%d"
                ),
            },
        )



elif page == "Route Intelligence":
    st.markdown(
        """
        <div class="eyebrow">ROUTE OPERATIONS / DEEP DIVE</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:27px;
        font-weight:700;color:#F1F5F9;margin-top:5px;">
            Route Intelligence
        </div>
        <div style="color:#91A2B7;font-size:13px;margin-bottom:18px;">
            Analyse route efficiency, travel delays and passenger activity.
        </div>
        """,
        unsafe_allow_html=True,
    )

    available_routes = sorted(filtered["route_id"].dropna().unique().tolist())

    if not available_routes:
        st.warning("No routes available for the selected filters.")
    else:
        selected_route = st.selectbox(
            "Select route to analyse",
            available_routes,
            key="route_intelligence_selector",
        )

        route_data = filtered[
            filtered["route_id"] == selected_route
        ].sort_values("timestamp")

        if route_data.empty:
            st.info("No records available for this route.")
        else:
            route_name = route_data["route_name"].iloc[0]

            st.markdown(
                f"""
                <div class="panel" style="margin:10px 0 18px 0;">
                    <div class="eyebrow">SELECTED ROUTE</div>
                    <div style="font-family:'Space Grotesk',sans-serif;
                    font-size:24px;font-weight:700;color:#F1F5F9;
                    margin-top:5px;">{selected_route}</div>
                    <div style="color:#91A2B7;font-size:13px;">
                    {route_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            avg_delay = route_data["delay_minutes"].mean()
            avg_speed = route_data["speed_kmh"].mean()
            total_passengers = route_data["passengers"].sum()
            total_records = len(route_data)

            a, b, c, d = st.columns(4)

            a.metric(
                "Average delay",
                f"{avg_delay:.1f} min",
            )
            b.metric(
                "Average speed",
                f"{avg_speed:.1f} km/h",
            )
            c.metric(
                "Recorded passengers",
                f"{total_passengers:,.0f}",
            )
            d.metric(
                "GPS records",
                f"{total_records:,}",
            )

            st.markdown(
                '<div class="section-label">Route performance timeline</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Hourly average delay and speed. Use the legend to toggle either metric."
            )

            route_trend = (
                route_data.assign(
                    hour_start=route_data["timestamp"].dt.floor("h")
                )
                .groupby("hour_start", as_index=False)
                .agg(
                    delay=("delay_minutes", "mean"),
                    speed=("speed_kmh", "mean"),
                )
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=route_trend["hour_start"],
                    y=route_trend["delay"],
                    mode="lines+markers",
                    name="Average delay",
                    line=dict(color=TEAL, width=3),
                    marker=dict(size=5),
                    hovertemplate="%{x}<br>Delay: %{y:.1f} min<extra></extra>",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=route_trend["hour_start"],
                    y=route_trend["speed"],
                    mode="lines+markers",
                    name="Average speed",
                    yaxis="y2",
                    line=dict(color=AMBER, width=2.5),
                    marker=dict(size=4),
                    hovertemplate="%{x}<br>Speed: %{y:.1f} km/h<extra></extra>",
                )
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=15, r=15, t=30, b=15),
                height=420,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
                font=dict(
                    family="DM Sans, sans-serif",
                    color="#D5DFEA",
                ),
                yaxis=dict(
                    title="Delay (minutes)",
                    gridcolor="rgba(145,162,183,0.13)",
                    zeroline=False,
                ),
                yaxis2=dict(
                    title="Speed (km/h)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    zeroline=False,
                ),
                xaxis=dict(
                    title=None,
                    showgrid=False,
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                '<div class="section-label">Recorded route activity</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Latest observations appear first. Use the table controls to explore the records."
            )

            display_route_data = route_data.sort_values(
                "timestamp",
                ascending=False,
            )

            st.dataframe(
                display_route_data,
                use_container_width=True,
                hide_index=True,
                height=420,
            )

            csv_data = display_route_data.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download route records (CSV)",
                data=csv_data,
                file_name=f"{selected_route}_route_records.csv",
                mime="text/csv",
                key="download_route_records",
            )


elif page == "Congestion Analysis":
    st.markdown(
        """
        <div class="eyebrow">TRAFFIC OPERATIONS / HOTSPOT MONITORING</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:27px;
        font-weight:700;color:#F1F5F9;margin-top:5px;">
            Congestion Analysis
        </div>
        <div style="color:#91A2B7;font-size:13px;margin-bottom:18px;">
            Identify high-delay routes, passenger pressure and peak congestion periods.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_hourly.empty:
        st.info("No congestion data is available for the selected filters.")
    else:
        avg_delay_all = filtered["delay_minutes"].mean()
        avg_speed_all = filtered["speed_kmh"].mean()
        peak_hour = (
            filtered.groupby("hour")["passengers"]
            .sum()
            .idxmax()
        )
        high_delay_count = int(
            (filtered_hourly["avg_delay"] >= avg_delay_all).sum()
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Average delay", f"{avg_delay_all:.1f} min")
        m2.metric("Average speed", f"{avg_speed_all:.1f} km/h")
        m3.metric("Busiest hour", f"{int(peak_hour):02d}:00")
        m4.metric("High-delay slots", f"{high_delay_count:,}")

        st.markdown("---")

        left, right = st.columns([1.2, 1])

        with left:
            st.markdown(
                '<div class="section-label">Congestion score by route</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Score combines relative delay, passenger demand and slower speeds."
            )

            congestion_route = (
                filtered_hourly.groupby("route_id", as_index=False)
                .agg(
                    avg_delay=("avg_delay", "mean"),
                    passenger_demand=("passenger_demand", "sum"),
                    avg_speed=("avg_speed", "mean"),
                )
            )

            congestion_route["delay_rank"] = (
                congestion_route["avg_delay"].rank(pct=True)
            )
            congestion_route["demand_rank"] = (
                congestion_route["passenger_demand"].rank(pct=True)
            )
            congestion_route["slow_rank"] = (
                1 - congestion_route["avg_speed"].rank(pct=True)
            )

            congestion_route["congestion_score"] = 100 * (
                0.50 * congestion_route["delay_rank"]
                + 0.30 * congestion_route["demand_rank"]
                + 0.20 * congestion_route["slow_rank"]
            )

            congestion_route = congestion_route.sort_values(
                "congestion_score",
                ascending=True,
            )

            fig = px.bar(
                congestion_route,
                x="congestion_score",
                y="route_id",
                orientation="h",
                color="congestion_score",
                color_continuous_scale=[
                    [0, "#43E6C1"],
                    [0.55, "#FFBF69"],
                    [1, "#FB7185"],
                ],
                template="plotly_dark",
                labels={
                    "congestion_score": "Congestion score",
                    "route_id": "Route",
                },
                hover_data=["avg_delay", "avg_speed", "passenger_demand"],
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                height=390,
                margin=dict(l=10, r=10, t=15, b=10),
                font=dict(family="DM Sans, sans-serif", color="#D5DFEA"),
            )
            fig.update_xaxes(
                gridcolor="rgba(145,162,183,0.13)",
                title="Relative score (0–100)",
            )
            fig.update_yaxes(showgrid=False, title=None)

            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown(
                '<div class="section-label">Demand vs delay</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Explore whether higher passenger demand coincides with longer delays."
            )

            fig = px.scatter(
                filtered_hourly,
                x="passenger_demand",
                y="avg_delay",
                color="route_id",
                size="trip_records",
                hover_data=["hour_start", "avg_speed"],
                template="plotly_dark",
                labels={
                    "passenger_demand": "Passenger count",
                    "avg_delay": "Average delay (min)",
                    "route_id": "Route",
                    "trip_records": "Records",
                    "hour_start": "Time slot",
                    "avg_speed": "Average speed",
                },
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=390,
                margin=dict(l=10, r=10, t=15, b=10),
                font=dict(family="DM Sans, sans-serif", color="#D5DFEA"),
                legend_title_text="Route",
            )
            fig.update_xaxes(gridcolor="rgba(145,162,183,0.13)")
            fig.update_yaxes(gridcolor="rgba(145,162,183,0.13)")

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.markdown(
            '<div class="section-label">Congestion by time of day</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Average delay for each hour, aggregated across the selected routes."
        )

        hourly_pattern = (
            filtered.groupby("hour", as_index=False)
            .agg(
                avg_delay=("delay_minutes", "mean"),
                passengers=("passengers", "sum"),
                avg_speed=("speed_kmh", "mean"),
            )
            .sort_values("hour")
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=hourly_pattern["hour"],
                y=hourly_pattern["avg_delay"],
                name="Average delay",
                marker_color=TEAL,
                opacity=0.9,
                hovertemplate="Hour %{x}:00<br>Delay: %{y:.1f} min<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=hourly_pattern["hour"],
                y=hourly_pattern["avg_speed"],
                name="Average speed",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color=AMBER, width=3),
                marker=dict(size=6),
                hovertemplate="Hour %{x}:00<br>Speed: %{y:.1f} km/h<extra></extra>",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=390,
            margin=dict(l=10, r=15, t=20, b=10),
            hovermode="x unified",
            font=dict(family="DM Sans, sans-serif", color="#D5DFEA"),
            xaxis=dict(
                title="Hour of day",
                dtick=2,
                showgrid=False,
            ),
            yaxis=dict(
                title="Average delay (min)",
                gridcolor="rgba(145,162,183,0.13)",
            ),
            yaxis2=dict(
                title="Average speed (km/h)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.markdown(
            '<div class="section-label">Top 3 priority congestion slots</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Ranked using the combined congestion score for the selected data."
        )

        if top_slots.empty:
            st.info("No priority congestion slots were identified.")
        else:
            priority = top_slots[
                [
                    "route_id",
                    "route_name",
                    "hour_start",
                    "avg_delay",
                    "passenger_demand",
                    "avg_speed",
                    "congestion_score",
                ]
            ].rename(
                columns={
                    "route_id": "Route",
                    "route_name": "Route name",
                    "hour_start": "Time slot",
                    "avg_delay": "Avg delay (min)",
                    "passenger_demand": "Passengers",
                    "avg_speed": "Avg speed (km/h)",
                    "congestion_score": "Score",
                }
            )

            st.dataframe(
                priority.round(2),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Avg delay (min)": st.column_config.NumberColumn(
                        format="%.1f min"
                    ),
                    "Avg speed (km/h)": st.column_config.NumberColumn(
                        format="%.1f km/h"
                    ),
                    "Score": st.column_config.NumberColumn(
                        format="%.1f"
                    ),
                },
            )

            st.download_button(
                "Download congestion report",
                data=priority.to_csv(index=False).encode("utf-8"),
                file_name="congestion_analysis.csv",
                mime="text/csv",
                key="download_congestion_report",
            )



elif page == "Delay Prediction":
    st.markdown(
        '<div class="section-label">PREDICTIVE INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel"><h3 style="margin:0;color:#F1F5F9;">Delay Forecast Studio</h3>'
        '<p style="margin:8px 0 0;color:#91A2B7;">Configure a route and operating conditions to estimate average transport delay.</p></div>',
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
        st.error(f"Trained model file not found: {MODEL_PATH}")
        st.info("Run python train_model.py to train and save the model.")
    else:
        try:
            artifact = joblib.load(MODEL_PATH)

            if not isinstance(artifact, dict) or "model" not in artifact:
                st.error(
                    "The saved model has an unexpected format. "
                    "Run train_model.py again."
                )
                st.stop()

            delay_model = artifact["model"]
            model_features = artifact.get("features", [])

            expected_features = [
                "route_id",
                "hour",
                "is_peak_hour",
                "day_of_week",
                "month",
            ]

            if list(model_features) != expected_features:
                st.error(
                    f"Unexpected model features: {model_features}. "
                    "Please retrain the model using train_model.py."
                )
                st.stop()

            route_options = sorted(
                df["route_id"].dropna().astype(str).unique().tolist()
            )

            if not route_options:
                st.warning("No routes are available for prediction.")
                st.stop()

            st.markdown(
                '<div class="section-label">01 / CONFIGURE SCENARIO</div>',
                unsafe_allow_html=True,
            )

            input_col, summary_col = st.columns([1.5, 1])

            with input_col:
                with st.container(border=True):
                    st.markdown("#### Operating conditions")

                    selected_route = st.selectbox(
                        "Transport route",
                        route_options,
                        key="prediction_route",
                    )

                    prediction_hour = st.slider(
                        "Hour of day",
                        min_value=0,
                        max_value=23,
                        value=8,
                        format="%d:00",
                    )

                    peak_hour = st.checkbox(
                        "Peak-hour conditions",
                        value=prediction_hour in [7, 8, 9, 16, 17, 18, 19],
                        help="Enable this if the selected time is expected to have high passenger demand.",
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

                    prediction_day = st.selectbox(
                        "Day of week",
                        options=list(range(7)),
                        format_func=lambda day: day_names[day],
                    )

                    month_names = [
                        "January",
                        "February",
                        "March",
                        "April",
                        "May",
                        "June",
                        "July",
                        "August",
                        "September",
                        "October",
                        "November",
                        "December",
                    ]

                    prediction_month = st.selectbox(
                        "Month",
                        options=list(range(1, 13)),
                        format_func=lambda month: month_names[month - 1],
                        index=5,
                    )

            with summary_col:
                st.markdown(
                    '<div class="section-label">SCENARIO SUMMARY</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="panel" style="min-height:140px;">
                        <div style="color:#91A2B7;font-size:12px;">SELECTED ROUTE</div>
                        <div style="font-size:27px;font-weight:700;color:#43E6C1;margin-top:8px;">{selected_route}</div>
                        <div style="color:#91A2B7;font-size:13px;margin-top:6px;">Route identifier</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                peak_label = "Peak period" if peak_hour else "Non-peak period"

                st.markdown(
                    f"""
                    <div class="panel" style="margin-top:12px;">
                        <div style="color:#91A2B7;font-size:12px;">TIME WINDOW</div>
                        <div style="font-size:22px;font-weight:650;color:#F1F5F9;margin-top:8px;">{prediction_hour:02d}:00</div>
                        <div style="color:#91A2B7;font-size:13px;margin-top:5px;">{day_names[prediction_day]} · {month_names[prediction_month - 1]}</div>
                        <div style="display:inline-block;margin-top:14px;padding:5px 10px;border-radius:20px;background:rgba(67,230,193,0.12);color:#43E6C1;font-size:12px;">{peak_label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    "The model uses the five configured features to estimate average delay."
                )

            st.markdown(
                '<div class="section-label">02 / GENERATE FORECAST</div>',
                unsafe_allow_html=True,
            )

            if st.button(
                "Generate delay forecast  →",
                type="primary",
                use_container_width=True,
            ):
                prediction_input = pd.DataFrame(
                    [{
                        "route_id": str(selected_route),
                        "hour": int(prediction_hour),
                        "is_peak_hour": int(peak_hour),
                        "day_of_week": int(prediction_day),
                        "month": int(prediction_month),
                    }],
                    columns=model_features,
                )

                try:
                    predicted_delay = float(
                        np.asarray(
                            delay_model.predict(prediction_input)
                        ).reshape(-1)[0]
                    )

                    predicted_delay = max(0.0, predicted_delay)

                    if predicted_delay >= 15:
                        category = "High"
                        status_color = "#FB7185"
                        status_text = "Elevated delay risk"
                        recommendation = (
                            "Review congestion conditions and consider "
                            "additional service capacity for this route and time."
                        )
                    elif predicted_delay >= 5:
                        category = "Moderate"
                        status_color = "#FFBF69"
                        status_text = "Moderate delay risk"
                        recommendation = (
                            "Monitor this time slot and compare nearby operating "
                            "periods before making service changes."
                        )
                    else:
                        category = "Low"
                        status_color = "#43E6C1"
                        status_text = "Lower predicted delay"
                        recommendation = (
                            "The model estimates a relatively low average delay "
                            "under the selected conditions."
                        )

                    st.markdown(
                        '<div class="section-label">03 / FORECAST RESULTS</div>',
                        unsafe_allow_html=True,
                    )

                    result_col1, result_col2, result_col3 = st.columns(3)

                    with result_col1:
                        st.metric(
                            "Predicted average delay",
                            f"{predicted_delay:.1f} min",
                        )

                    with result_col2:
                        st.metric(
                            "Delay classification",
                            category,
                        )

                    with result_col3:
                        st.metric(
                            "Peak-hour setting",
                            "Enabled" if peak_hour else "Disabled",
                        )

                    st.markdown(
                        f"""
                        <div class="panel" style="border-left:4px solid {status_color};margin-top:16px;">
                            <div style="font-size:12px;letter-spacing:1px;color:#91A2B7;">FORECAST STATUS</div>
                            <div style="font-size:23px;font-weight:700;color:{status_color};margin-top:8px;">{status_text}</div>
                            <div style="font-size:14px;line-height:1.7;color:#F1F5F9;margin-top:10px;">{recommendation}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.caption(
                        f"Scenario: {selected_route} · {day_names[prediction_day]} · "
                        f"{month_names[prediction_month - 1]} · {prediction_hour:02d}:00"
                    )

                except Exception as error:
                    st.error(f"Prediction failed: {error}")

            st.markdown(
                '<div class="section-label">MODEL PERFORMANCE</div>',
                unsafe_allow_html=True,
            )

            performance_col1, performance_col2, performance_col3 = st.columns(3)

            with performance_col1:
                if artifact.get("mae") is not None:
                    st.metric("Test MAE", f"{float(artifact['mae']):.2f} min")
                else:
                    st.metric("Test MAE", "Not available")

            with performance_col2:
                if artifact.get("rmse") is not None:
                    st.metric("Test RMSE", f"{float(artifact['rmse']):.2f} min")
                else:
                    st.metric("Test RMSE", "Not available")

            with performance_col3:
                if artifact.get("r2") is not None:
                    st.metric("Test R²", f"{float(artifact['r2']):.3f}")
                else:
                    st.metric("Test R²", "Not available")

            st.caption(
                "These are estimates from historical demonstration data, not live traffic. "
                "MAE and RMSE measure prediction error; R² describes model fit."
            )

        except Exception as error:
            st.error(f"Could not load the delay prediction model: {error}")



elif page == "Model Insights":
    st.markdown(
        '<div class="section-label">MODEL INTELLIGENCE / PERFORMANCE LAB</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel"><h3 style="margin:0;color:#F1F5F9;">Predictive Model Insights</h3>'
        '<p style="margin:8px 0 0;color:#91A2B7;">Evaluate prediction quality, understand feature contributions and compare model estimates with observed delays.</p></div>',
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
        st.error("Trained model file not found. Please train the model first.")
    else:
        try:
            artifact = joblib.load(MODEL_PATH)

            if not isinstance(artifact, dict) or "model" not in artifact:
                st.error(
                    "Model artifact format is incorrect. "
                    "Run train_model.py again."
                )
            else:
                mae = artifact.get("mae")
                rmse = artifact.get("rmse")
                r2 = artifact.get("r2")

                st.markdown(
                    '<div class="section-label">01 / MODEL PERFORMANCE</div>',
                    unsafe_allow_html=True,
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Mean Absolute Error",
                        f"{mae:.2f} min" if mae is not None else "N/A",
                        help="Average absolute difference between actual and predicted delay.",
                    )

                with col2:
                    st.metric(
                        "Root Mean Squared Error",
                        f"{rmse:.2f} min" if rmse is not None else "N/A",
                        help="Penalizes larger prediction errors more heavily.",
                    )

                with col3:
                    st.metric(
                        "R² Score",
                        f"{r2:.3f}" if r2 is not None else "N/A",
                        help="Indicates how well predictions explain variation in the test data.",
                    )

                st.caption(
                    "Lower MAE and RMSE generally indicate smaller errors. "
                    "An R² closer to 1 generally indicates a better fit; "
                    "these metrics should be interpreted together."
                )

                st.markdown(
                    '<div class="section-label">02 / FEATURE IMPORTANCE</div>',
                    unsafe_allow_html=True,
                )

                model = artifact["model"]

                if (
                    hasattr(model, "named_steps")
                    and "preprocessor" in model.named_steps
                    and "regressor" in model.named_steps
                ):
                    preprocessor = model.named_steps["preprocessor"]
                    regressor = model.named_steps["regressor"]

                    try:
                        encoded_names = preprocessor.get_feature_names_out()
                        importances = regressor.feature_importances_

                        importance_df = pd.DataFrame({
                            "Encoded Feature": encoded_names,
                            "Importance": importances,
                        })

                        def group_feature(name):
                            name = str(name).lower()

                            if "route_id" in name:
                                return "Route ID"
                            if "is_peak_hour" in name:
                                return "Peak Hour"
                            if "day_of_week" in name:
                                return "Day of Week"
                            if "hour" in name:
                                return "Hour of Day"
                            if "month" in name:
                                return "Month"

                            return "Other"

                        importance_df["Feature"] = (
                            importance_df["Encoded Feature"].apply(group_feature)
                        )

                        grouped_importance = (
                            importance_df.groupby(
                                "Feature",
                                as_index=False,
                            )["Importance"]
                            .sum()
                            .sort_values("Importance", ascending=True)
                        )

                        feature_col, detail_col = st.columns([1.6, 1])

                        with feature_col:
                            fig = px.bar(
                                grouped_importance,
                                x="Importance",
                                y="Feature",
                                orientation="h",
                                labels={
                                    "Importance": "Relative Importance",
                                    "Feature": "Input Feature",
                                },
                                template="plotly_dark",
                            )

                            fig.update_traces(
                                marker_color="#43E6C1",
                                marker_line_width=0,
                            )

                            fig.update_layout(
                                height=370,
                                margin=dict(l=10, r=20, t=25, b=20),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#91A2B7"),
                                xaxis=dict(
                                    showgrid=True,
                                    gridcolor="rgba(145,162,183,0.12)",
                                ),
                                yaxis=dict(showgrid=False),
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                            )

                        with detail_col:
                            st.markdown(
                                '<div class="panel"><div style="color:#91A2B7;font-size:12px;">FEATURE SUMMARY</div>'
                                '<div style="font-size:19px;font-weight:700;color:#F1F5F9;margin-top:8px;">Input contribution</div>'
                                '<div style="color:#91A2B7;font-size:13px;margin-top:8px;">Aggregated importance across encoded model inputs.</div></div>',
                                unsafe_allow_html=True,
                            )

                            display_importance = grouped_importance.sort_values(
                                "Importance",
                                ascending=False,
                            )

                            for _, row in display_importance.iterrows():
                                st.markdown(
                                    f"""
                                    <div style="margin-top:15px;">
                                        <div style="display:flex;justify-content:space-between;gap:10px;color:#F1F5F9;font-size:13px;">
                                            <span>{row['Feature']}</span>
                                            <span>{row['Importance']:.3f}</span>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                        st.info(
                            "Feature importance indicates how the trained model uses its inputs. "
                            "It does not establish that a feature causes transport delays."
                        )

                    except Exception as error:
                        st.warning(
                            f"Could not calculate feature importance for this model: {error}"
                        )
                else:
                    st.info(
                        "Feature importance is unavailable because the saved model "
                        "does not expose the expected preprocessor and regressor pipeline."
                    )

                st.markdown(
                    '<div class="section-label">03 / ACTUAL VS PREDICTED</div>',
                    unsafe_allow_html=True,
                )

                predictions_path = (
                    BASE_DIR / "models" / "test_predictions.csv"
                )

                if predictions_path.exists():
                    results = pd.read_csv(predictions_path)

                    needed = {
                        "timestamp",
                        "Actual Delay",
                        "Predicted Delay",
                    }

                    if needed.issubset(results.columns):
                        results["timestamp"] = pd.to_datetime(
                            results["timestamp"],
                            errors="coerce",
                        )
                        results["Actual Delay"] = pd.to_numeric(
                            results["Actual Delay"],
                            errors="coerce",
                        )
                        results["Predicted Delay"] = pd.to_numeric(
                            results["Predicted Delay"],
                            errors="coerce",
                        )

                        results = results.dropna(
                            subset=[
                                "timestamp",
                                "Actual Delay",
                                "Predicted Delay",
                            ]
                        ).sort_values("timestamp")

                        if not results.empty:
                            results["Absolute Error"] = (
                                results["Actual Delay"]
                                - results["Predicted Delay"]
                            ).abs()

                            results["Signed Error"] = (
                                results["Predicted Delay"]
                                - results["Actual Delay"]
                            )

                            actual_mean = results["Actual Delay"].mean()
                            prediction_mean = results["Predicted Delay"].mean()
                            mean_error = results["Absolute Error"].mean()

                            metric1, metric2, metric3 = st.columns(3)

                            with metric1:
                                st.metric(
                                    "Test Records",
                                    f"{len(results):,}",
                                )

                            with metric2:
                                st.metric(
                                    "Mean Actual Delay",
                                    f"{actual_mean:.1f} min",
                                )

                            with metric3:
                                st.metric(
                                    "Mean Absolute Error",
                                    f"{mean_error:.2f} min",
                                )

                            fig = go.Figure()

                            fig.add_trace(
                                go.Scatter(
                                    x=results["timestamp"],
                                    y=results["Actual Delay"],
                                    mode="lines",
                                    name="Actual Delay",
                                    line=dict(
                                        color="#43E6C1",
                                        width=2,
                                    ),
                                )
                            )

                            fig.add_trace(
                                go.Scatter(
                                    x=results["timestamp"],
                                    y=results["Predicted Delay"],
                                    mode="lines",
                                    name="Predicted Delay",
                                    line=dict(
                                        color="#FFBF69",
                                        width=2,
                                    ),
                                )
                            )

                            fig.update_layout(
                                xaxis_title="Timestamp",
                                yaxis_title="Delay (minutes)",
                                template="plotly_dark",
                                height=420,
                                hovermode="x unified",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#91A2B7"),
                                margin=dict(l=10, r=15, t=25, b=10),
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="left",
                                    x=0,
                                ),
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                            )

                            error_fig = px.histogram(
                                results,
                                x="Signed Error",
                                nbins=25,
                                labels={
                                    "Signed Error": "Prediction Error (minutes)"
                                },
                                template="plotly_dark",
                            )

                            error_fig.update_traces(
                                marker_color="#43E6C1",
                                marker_line_color="#101B2B",
                                marker_line_width=1,
                            )

                            error_fig.update_layout(
                                height=300,
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#91A2B7"),
                                margin=dict(l=10, r=15, t=20, b=10),
                                xaxis=dict(
                                    showgrid=True,
                                    gridcolor="rgba(145,162,183,0.12)",
                                ),
                                yaxis=dict(
                                    title="Number of Records",
                                    showgrid=True,
                                    gridcolor="rgba(145,162,183,0.12)",
                                ),
                            )

                            st.markdown("#### Prediction error distribution")
                            st.plotly_chart(
                                error_fig,
                                use_container_width=True,
                            )

                            with st.expander("View test prediction records"):
                                st.dataframe(
                                    results[
                                        [
                                            "timestamp",
                                            "Actual Delay",
                                            "Predicted Delay",
                                            "Absolute Error",
                                            "Signed Error",
                                        ]
                                    ].head(100),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                            st.download_button(
                                "Download prediction analysis CSV",
                                data=results.to_csv(index=False).encode("utf-8"),
                                file_name="transitiq_prediction_analysis.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

                        else:
                            st.warning(
                                "The test predictions file has no valid rows."
                            )
                    else:
                        st.warning(
                            "The test predictions file is missing required columns."
                        )
                else:
                    st.warning(
                        "Test predictions file not found. "
                        "Run python train_model.py first."
                    )

                st.markdown(
                    '<div class="section-label">04 / TRAINING INFORMATION</div>',
                    unsafe_allow_html=True,
                )

                info_col1, info_col2 = st.columns(2)

                with info_col1:
                    st.markdown(
                        f"""
                        <div class="panel">
                            <div style="color:#91A2B7;font-size:12px;">TRAINED UNTIL</div>
                            <div style="font-size:19px;font-weight:650;color:#F1F5F9;margin-top:9px;">{artifact.get("trained_until", "Not available")}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with info_col2:
                    st.markdown(
                        f"""
                        <div class="panel">
                            <div style="color:#91A2B7;font-size:12px;">TESTING PERIOD STARTS</div>
                            <div style="font-size:19px;font-weight:650;color:#F1F5F9;margin-top:9px;">{artifact.get("tested_from", "Not available")}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        except Exception as error:
            st.error(f"Could not load model insights: {error}")



elif page == "Data Explorer":
    st.markdown(
        '<div class="section-label">DATA MANAGEMENT / EXPLORATION</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel"><h3 style="margin:0;color:#F1F5F9;">Transport Data Explorer</h3>'
        '<p style="margin:8px 0 0;color:#91A2B7;">Search, inspect and export transport records using the active dashboard filters.</p></div>',
        unsafe_allow_html=True,
    )

    explorer = filtered.copy()

    if "timestamp" in explorer.columns:
        explorer["timestamp"] = pd.to_datetime(
            explorer["timestamp"],
            errors="coerce",
        )

    total_records = len(explorer)
    total_routes = (
        explorer["route_id"].nunique()
        if "route_id" in explorer.columns
        else 0
    )
    missing_cells = int(explorer.isna().sum().sum())
    duplicate_rows = int(explorer.duplicated().sum())

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric("Total Records", f"{total_records:,}")

    with metric2:
        st.metric("Unique Routes", f"{total_routes:,}")

    with metric3:
        st.metric("Missing Values", f"{missing_cells:,}")

    with metric4:
        st.metric("Duplicate Rows", f"{duplicate_rows:,}")

    st.markdown(
        '<div class="section-label">01 / SEARCH & FILTER</div>',
        unsafe_allow_html=True,
    )

    filter_col1, filter_col2 = st.columns([1.4, 1])

    with filter_col1:
        search = st.text_input(
            "Search route ID or route name",
            placeholder="e.g. R01 or Central Route",
            key="explorer_search",
        )

    with filter_col2:
        delay_column = (
            "delay_minutes"
            if "delay_minutes" in explorer.columns
            else None
        )

        if delay_column and not explorer.empty:
            delay_values = pd.to_numeric(
                explorer[delay_column],
                errors="coerce",
            ).dropna()

            if not delay_values.empty:
                delay_min = float(delay_values.min())
                delay_max = float(delay_values.max())

                if delay_min < delay_max:
                    delay_range = st.slider(
                        "Delay range (minutes)",
                        min_value=delay_min,
                        max_value=delay_max,
                        value=(delay_min, delay_max),
                    )
                else:
                    delay_range = (delay_min, delay_max)
                    st.caption(f"Delay value: {delay_min:.1f} minutes")
            else:
                delay_range = None
                st.caption("No valid delay values available.")
        else:
            delay_range = None
            st.caption("Delay filter unavailable.")

    if search:
        search_mask = pd.Series(False, index=explorer.index)

        if "route_id" in explorer.columns:
            search_mask |= explorer["route_id"].astype(str).str.contains(
                search,
                case=False,
                na=False,
                regex=False,
            )

        if "route_name" in explorer.columns:
            search_mask |= explorer["route_name"].astype(str).str.contains(
                search,
                case=False,
                na=False,
                regex=False,
            )

        explorer = explorer[search_mask]

    if delay_range is not None and "delay_minutes" in explorer.columns:
        numeric_delay = pd.to_numeric(
            explorer["delay_minutes"],
            errors="coerce",
        )

        explorer = explorer[
            numeric_delay.between(
                delay_range[0],
                delay_range[1],
                inclusive="both",
            )
        ]

    st.markdown(
        '<div class="section-label">02 / DATASET PREVIEW</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        f"Showing {len(explorer):,} of {total_records:,} records "
        "after applying the explorer filters."
    )

    if explorer.empty:
        st.info("No records match your search and filter settings.")
    else:
        available_columns = explorer.columns.tolist()

        default_columns = [
            column
            for column in [
                "timestamp",
                "route_id",
                "route_name",
                "delay_minutes",
                "passengers",
                "speed_kmh",
                "latitude",
                "longitude",
            ]
            if column in available_columns
        ]

        if not default_columns:
            default_columns = available_columns[: min(8, len(available_columns))]

        selected_columns = st.multiselect(
            "Choose columns to display and export",
            options=available_columns,
            default=default_columns,
            key="explorer_columns",
        )

        if not selected_columns:
            st.warning("Select at least one column to display the dataset.")
        else:
            display_data = explorer[selected_columns].copy()

            if "timestamp" in display_data.columns:
                display_data = display_data.sort_values(
                    "timestamp",
                    ascending=False,
                    na_position="last",
                )

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True,
                height=430,
            )

            csv_bytes = display_data.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download selected data as CSV",
                data=csv_bytes,
                file_name="transitiq_transport_export.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

    st.markdown(
        '<div class="section-label">03 / DATA QUALITY</div>',
        unsafe_allow_html=True,
    )

    quality_col1, quality_col2 = st.columns(2)

    with quality_col1:
        st.markdown("#### Missing values by column")

        missing_df = (
            explorer.isna()
            .sum()
            .rename("Missing Values")
            .reset_index()
        )

        missing_df.columns = ["Column", "Missing Values"]

        if not missing_df.empty:
            missing_df = missing_df.sort_values(
                "Missing Values",
                ascending=False,
            )

            missing_df = missing_df[missing_df["Missing Values"] > 0]

        if missing_df.empty:
            st.success("No missing values found in the filtered dataset.")
        else:
            st.dataframe(
                missing_df,
                use_container_width=True,
                hide_index=True,
            )

    with quality_col2:
        st.markdown("#### Column data types")

        dtype_df = pd.DataFrame({
            "Column": explorer.columns,
            "Data Type": [
                str(explorer[column].dtype)
                for column in explorer.columns
            ],
            "Unique Values": [
                explorer[column].nunique(dropna=True)
                for column in explorer.columns
            ],
        })

        st.dataframe(
            dtype_df,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Data quality statistics are calculated on the records remaining "
        "after the active dashboard filters and explorer filters. "
        "Exporting data does not modify the original dataset."
    )
