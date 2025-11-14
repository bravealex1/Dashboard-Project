# local_csv.py — City Health Dashboard (Local CSV only)
# Focus: simple & robust; defaults to Baltimore (FIPS 24510).
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="City Health Dashboard — Local CSV", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .small-note { font-size: 0.9rem; color: #666; }
      .metric-card { padding: 12px 16px; border: 1px solid #eee; border-radius: 14px; }
      .warn { background: #fff8e1; padding: 10px 12px; border-radius: 10px; border: 1px solid #fde7a1; }
    </style>
    """,
    unsafe_allow_html=True
)

NOT_APPLICABLE = -999

# ---------- Helpers ----------
def _clean_numeric(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.replace(NOT_APPLICABLE, np.nan)

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    for col in ["num", "denom", "est", "lci", "uci"]:
        if col in df.columns:
            df[col] = _clean_numeric(df[col])
    if "data_period" in df.columns:
        df["data_period"] = pd.to_numeric(df["data_period"], errors="coerce")
    for col in ["state_abbr","geo_level","geo_name","metric_name","group_name",
                "source_name","parent_level","parent_name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    for col in ["geo_fips","parent_fips","state_fips"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def _available_values(df: pd.DataFrame, col: str) -> List[str]:
    if col not in df.columns or df.empty: return []
    vals = (
        df[col].dropna().astype(str).replace({"nan": np.nan}).dropna().unique()
    )
    return sorted(vals)

def _summarize_metric(df_metric: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if "est" not in df_metric.columns or df_metric["est"].dropna().empty:
        return (None, None, None)
    return (df_metric["est"].mean(), df_metric["est"].min(), df_metric["est"].max())

def _normalize_city_name(name: str) -> str:
    return (name or "").lower().strip()

def _filter_city(df: pd.DataFrame, city_name: str, parent_fips: Optional[str]) -> pd.DataFrame:
    """
    Robust city filter for both tract-level (preferred) and city-level CSVs.
    Tracts: parent_level='city', uses parent_name/parent_fips.
    City-level: geo_level='city', uses geo_name.
    """
    if df.empty: return df
    city_norm = _normalize_city_name(city_name)
    variants = {city_norm}
    if city_norm == "baltimore":
        variants.update({"baltimore city", "city of baltimore"})

    # 1) Prefer exact parent_fips match when present (e.g., 24510)
    if parent_fips and "parent_fips" in df.columns:
        m = df["parent_fips"].astype(str).str.strip() == str(parent_fips).strip()
        hit = df.loc[m]
        if not hit.empty: return hit

    # 2) parent_name + parent_level
    if {"parent_name","parent_level"}.issubset(df.columns):
        m = df["parent_level"].str.lower().eq("city") & df["parent_name"].str.lower().isin(variants)
        hit = df.loc[m]
        if not hit.empty: return hit

    # 3) city-level name
    if {"geo_level","geo_name"}.issubset(df.columns):
        m = df["geo_level"].str.lower().eq("city") & df["geo_name"].str.lower().isin(variants)
        hit = df.loc[m]
        if not hit.empty: return hit

    # 4) loose contains match as a final fallback
    if "parent_name" in df.columns:
        m = df["parent_name"].str.lower().str.contains(city_norm, na=False)
        hit = df.loc[m]
        if not hit.empty: return hit
    if "geo_name" in df.columns:
        m = df["geo_name"].str.lower().str.contains(city_norm, na=False)
        hit = df.loc[m]
        if not hit.empty: return hit

    return df.iloc[0:0].copy()

# ---------- Data load ----------
@st.cache_data(show_spinner=False)
def load_local(cities_csv_path: Optional[str], tracts_csv_path: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cities_df, tracts_df = pd.DataFrame(), pd.DataFrame()
    if cities_csv_path and os.path.exists(cities_csv_path):
        cities_df = pd.read_csv(cities_csv_path, low_memory=False)
        cities_df = _normalize_columns(cities_df)
    if tracts_csv_path and os.path.exists(tracts_csv_path):
        tracts_df = pd.read_csv(tracts_csv_path, low_memory=False)
        tracts_df = _normalize_columns(tracts_df)
    return cities_df, tracts_df

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🧭 Controls (Local CSVs)")

    # Use your absolute paths if you like; otherwise keep filenames (same folder).
    cities_path = st.text_input("Path to city CSV (optional)",
                                value="CityHealthDash-cities.csv")
    tracts_path = st.text_input("Path to tract CSV",
                                value="CityHealthDash-tracts.csv")

    st.divider()
    st.write("**Geography**")
    city_input = st.text_input("City name", value="Baltimore")
    parent_fips_input = st.text_input("Parent FIPS (Baltimore city = 24510)", value="24510")

    st.divider()
    # group dropdown will be populated *after* filtering; this is just a placeholder label
    st.caption("Choose metric & group on the main pane.")

st.title("📊 City Health Dashboard — Local CSV")
st.caption(
    "Local CSVs only. Column rules per CHD Technical Document & Data Dictionary "
    "(e.g., **-999** = not applicable; pooled years via **period_type**)."
)

ref = st.expander("Column meanings & notes", expanded=False)
with ref:
    st.markdown(
        "- **est**: estimate (%/rate/value). **lci/uci**: 90% CI. "
        "**-999** means not applicable; **NA** can indicate censored/unavailable. "
        "Use **period_type** to interpret pooled years."
    )

# ---------- Load files ----------
cities_df, tracts_df = load_local(cities_path or None, tracts_path or None)

if (cities_path and not os.path.exists(cities_path)) and (tracts_path and not os.path.exists(tracts_path)):
    st.markdown("<div class='warn'><b>No CSVs found.</b> Check your paths.</div>", unsafe_allow_html=True)
    st.stop()

# Prefer tract-level for neighborhood detail
work_tracts = _filter_city(tracts_df, city_input, parent_fips_input.strip() or None)
work_cities = _filter_city(cities_df, city_input, parent_fips_input.strip() or None)

if work_tracts.empty and work_cities.empty:
    st.markdown("<div class='warn'><b>No rows matched your city filter.</b> Try a different city name or FIPS.</div>", unsafe_allow_html=True)
    # Show examples to help the user debug:
    cols = st.columns(2)
    with cols[0]:
        if "parent_name" in tracts_df.columns:
            st.write("Sample `parent_name` values (tracts):", pd.Series(_available_values(tracts_df, "parent_name")).head(12))
        if "geo_name" in cities_df.columns:
            st.write("Sample `geo_name` values (cities):", pd.Series(_available_values(cities_df, "geo_name")).head(12))
    with cols[1]:
        if "parent_fips" in tracts_df.columns:
            st.write("Sample `parent_fips` values:", pd.Series(_available_values(tracts_df, "parent_fips")).head(12))
        if "state_abbr" in tracts_df.columns:
            st.write("States present:", pd.Series(_available_values(tracts_df, "state_abbr")).head(12))
    st.stop()

working_df = work_tracts if not work_tracts.empty else work_cities

# ---------- Metric & Group dropdowns (robust) ----------
all_metrics = _available_values(working_df, "metric_name")
if not all_metrics:
    st.error("No `metric_name` values available in the filtered data.")
    st.stop()

# Metric dropdown (searchable)
metric = st.selectbox("Metric", options=all_metrics, index=0, help="Start typing to search.")

# Build group list for *this* metric
metric_df = working_df[working_df["metric_name"] == metric].copy()
groups_for_metric = _available_values(metric_df, "group_name")
default_group = "Total" if "Total" in groups_for_metric else (groups_for_metric[0] if groups_for_metric else None)

group = st.selectbox("Group", options=groups_for_metric or ["(no group values)"],
                     index=(groups_for_metric.index("Total") if "Total" in groups_for_metric else 0) if groups_for_metric else 0)

if groups_for_metric:
    metric_df = metric_df[metric_df["group_name"].str.lower() == group.lower()]

# ---------- Year control (safe when empty) ----------
years = sorted(pd.to_numeric(metric_df.get("data_period", pd.Series(dtype=float)), errors="coerce").dropna().unique().tolist())

# If there are no years, skip the slider instead of erroring
if years:
    year = st.select_slider("Year (data_period)", options=years, value=years[-1])
    metric_year_df = metric_df[metric_df["data_period"] == year].copy()
else:
    st.info("No `data_period` available for this selection — showing all rows for the chosen metric/group.")
    metric_year_df = metric_df.copy()

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
mean_est, min_est, max_est = _summarize_metric(metric_year_df)
with c1:
    st.markdown("**Rows**")
    st.markdown(f"<div class='metric-card'>{len(metric_year_df):,}</div>", unsafe_allow_html=True)
with c2:
    st.markdown("**Average est**")
    st.markdown(f"<div class='metric-card'>{'—' if mean_est is None else round(float(mean_est),3)}</div>", unsafe_allow_html=True)
with c3:
    st.markdown("**Min (tract/city)**")
    st.markdown(f"<div class='metric-card'>{'—' if min_est is None else round(float(min_est),3)}</div>", unsafe_allow_html=True)
with c4:
    st.markdown("**Max (tract/city)**")
    st.markdown(f"<div class='metric-card'>{'—' if max_est is None else round(float(max_est),3)}</div>", unsafe_allow_html=True)

# ---------- Map (tracts) ----------
st.subheader("Neighborhood View (Census Tracts)")
lat_col = next((c for c in ["centroid_lat","lat","latitude"] if c in metric_year_df.columns), None)
lon_col = next((c for c in ["centroid_lon","lon","longitude"] if c in metric_year_df.columns), None)

if lat_col and lon_col and not metric_year_df[[lat_col, lon_col]].dropna().empty:
    map_df = metric_year_df.dropna(subset=[lat_col, lon_col, "est"]).copy()
    map_df["hover"] = map_df.apply(
        lambda r: f"Tract: {r.get('geo_fips','')}<br>Estimate: {r.get('est','')}<br>Metric: {r.get('metric_name','')}<br>Year: {r.get('data_period','')}",
        axis=1
    )
    fig = px.scatter_mapbox(
        map_df, lat=lat_col, lon=lon_col, color="est",
        custom_data=["hover"], height=520, zoom=10, color_continuous_scale="Viridis"
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No centroid latitude/longitude in this slice. If your tract CSV includes centroids, the map will appear.")

# ---------- Trend ----------
st.subheader("Trend Over Time (Aggregate)")
if "data_period" in metric_df.columns and not metric_df["data_period"].dropna().empty:
    trend = (
        metric_df.dropna(subset=["data_period"])
                 .groupby("data_period", as_index=False)["est"]
                 .mean(numeric_only=True)
                 .sort_values("data_period")
    )
    if not trend.empty:
        fig2 = px.line(trend, x="data_period", y="est", markers=True, height=380)
        fig2.update_layout(xaxis_title="Year (data_period)", yaxis_title="Average estimate (filtered area)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No trend available for this metric/group.")
else:
    st.info("`data_period` not present for this metric/group.")

# ---------- Table ----------
st.subheader("Data Table")
show_cols = [c for c in [
    "state_abbr","parent_name","geo_name","geo_fips","parent_fips",
    "metric_name","group_name","est","lci","uci","num","denom",
    "data_period","period_type","source_name","version"
] if c in metric_year_df.columns]
if not show_cols:
    show_cols = list(metric_year_df.columns)

st.dataframe(
    metric_year_df[show_cols].sort_values(
        by=[c for c in ["parent_name","geo_name","geo_fips","data_period"] if c in show_cols]
    ),
    use_container_width=True
)

st.divider()
st.caption(
    "Per City Health Dashboard docs: -999 = not applicable; some values may be censored/unavailable; "
    "interpret pooled years with 'period_type'."
)
