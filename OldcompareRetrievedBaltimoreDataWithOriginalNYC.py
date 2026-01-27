# water_chd_dashboard.py
# Dashboard to explore correlations between local surface water parameters
# and City Health Dashboard (city-level) metrics.
#
# Files expected (adjust in Sidebar if needed):
#   /mnt/data/Surface_Water_Quality_Data_Jan_2020_to_present.csv
#   /mnt/data/CityHealthDash-cities.csv
#   /mnt/data/CityHealthDash-tracts.csv  (not required for joins here)
#
# Notes:
# - This is EXPLORATORY. Correlations ≠ causation.
# - Surface water (recreation/ecology) ≠ household tap water risk.
# - LOD handling: substitutes <x with x/2 (common exploratory practice).

import os, re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import spearmanr

st.set_page_config(page_title="Water ↔ CityHealth Explorer", page_icon="💧", layout="wide")

st.markdown("""
<style>
.small { color:#666; font-size:0.9rem; }
.metric-card { padding:12px 16px; border:1px solid #eee; border-radius:14px; }
.warn { background:#fff8e1; padding:10px 12px; border-radius:10px; border:1px solid #fde7a1; }
</style>
""", unsafe_allow_html=True)

NOT_APPLICABLE = -999

# --------------------------- Helpers --------------------------- #
def read_csv_safely(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")

def to_numeric_censored(x, strategy="LOD/2"):
    """Handle strings like '<6.7' per selected strategy; else try float."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = re.match(r"^<\s*([0-9]*\.?[0-9]+)$", s)
    if m:
        val = float(m.group(1))
        if strategy == "LOD/2":
            return val/2.0
        elif strategy == "LOD/sqrt2":
            return val/np.sqrt(2.0)
        elif strategy == "zero":
            return 0.0
        else:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def normalize_chd(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    # Standard cleaning per CHD docs: cast numerics; -999 to NaN
    for c in ["num","denom","est","lci","uci","data_period"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].replace(NOT_APPLICABLE, np.nan)
    for c in ["geo_level","geo_name","group_name","metric_name","parent_name","parent_level"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in ["geo_fips","parent_fips","state_fips"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def filter_city_frame(df: pd.DataFrame, city_name: str, parent_fips: Optional[str]) -> pd.DataFrame:
    """Robust city filter for both tract & city CSVs."""
    if df.empty: return df
    cname = (city_name or "").lower().strip()
    variants = {cname}
    if cname == "baltimore":
        variants.update({"baltimore city","city of baltimore"})
    # Prefer exact parent_fips match (tracts)
    if parent_fips and "parent_fips" in df.columns:
        hit = df[df["parent_fips"].astype(str).str.strip() == str(parent_fips).strip()]
        if not hit.empty: return hit
    # tract names
    if {"parent_name","parent_level"}.issubset(df.columns):
        m = df["parent_level"].str.lower().eq("city") & df["parent_name"].str.lower().isin(variants)
        hit = df.loc[m]
        if not hit.empty: return hit
    # city-level names
    if {"geo_level","geo_name"}.issubset(df.columns):
        m = df["geo_level"].str.lower().eq("city") & df["geo_name"].str.lower().isin(variants)
        hit = df.loc[m]
        if not hit.empty: return hit
    # loose contains
    for nm_col in ["parent_name","geo_name"]:
        if nm_col in df.columns:
            hit = df[df[nm_col].str.lower().str.contains(cname, na=False)]
            if not hit.empty: return hit
    return df.iloc[0:0]

def aggregate_water_by_year(df: pd.DataFrame, value_col="Result_num", how="median") -> pd.DataFrame:
    if df.empty or "year" not in df.columns or "Parameter" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["year","Parameter"])
    if df.empty: return pd.DataFrame()
    agg = (
        df.groupby(["year","Parameter"], as_index=False)
          .agg(median_val=(value_col,"median"),
               mean_val=(value_col,"mean"),
               n=(value_col,"count"))
    )
    med = agg.pivot(index="year", columns="Parameter", values="median_val")
    med.columns = [f"Water__{c}__median" for c in med.columns]
    mean = agg.pivot(index="year", columns="Parameter", values="mean_val")
    mean.columns = [f"Water__{c}__mean" for c in mean.columns]
    out = pd.concat([med, mean], axis=1).sort_index()
    out.index.name = "year"
    return out

def chd_city_wide(df_city: pd.DataFrame, group="Total", picked_metrics: Optional[List[str]]=None) -> pd.DataFrame:
    if df_city.empty: return pd.DataFrame()
    c = df_city.copy()
    if "group_name" in c.columns:
        c = c[c["group_name"].str.lower() == (group or "Total").lower()]
    if picked_metrics:
        c = c[c["metric_name"].isin(picked_metrics)]
    # year & estimate
    c["year"] = pd.to_numeric(c.get("data_period"), errors="coerce")
    c["est"]  = pd.to_numeric(c.get("est"), errors="coerce")
    c = c.dropna(subset=["year","metric_name","est"])
    agg = c.groupby(["year","metric_name"], as_index=False)["est"].mean()
    wide = agg.pivot(index="year", columns="metric_name", values="est").sort_index()
    wide.columns = [f"CHD__{m}" for m in wide.columns]
    wide.index.name = "year"
    return wide

def corr_table(df_join: pd.DataFrame, min_overlap=4) -> pd.DataFrame:
    if df_join.empty: return pd.DataFrame(columns=["water_series","chd_metric","n_years","pearson_r","spearman_r","p_spearman"])
    water_cols = [c for c in df_join.columns if c.startswith("Water__")]
    chd_cols   = [c for c in df_join.columns if c.startswith("CHD__")]
    out = []
    for w in water_cols:
        for c in chd_cols:
            xy = df_join[[w,c]].dropna()
            if len(xy) >= min_overlap:
                pear = xy[w].corr(xy[c], method="pearson")
                spear, pval = spearmanr(xy[w], xy[c], nan_policy="omit")
                out.append({"water_series": w, "chd_metric": c, "n_years": len(xy),
                            "pearson_r": pear, "spearman_r": spear, "p_spearman": pval})
    if not out:
        return pd.DataFrame(columns=["water_series","chd_metric","n_years","pearson_r","spearman_r","p_spearman"])
    return pd.DataFrame(out).sort_values(by="pearson_r", ascending=False)

# --------------------------- Sidebar --------------------------- #
with st.sidebar:
    st.header("💧 Data inputs (Local CSVs)")
    water_path  = st.text_input("Water CSV",  "/Users/qiuhaozhu/Desktop/Capstone_Project_Dashboard/Surface_Water_Quality_Data_Jan_2020_to_present.csv")
    cities_path = st.text_input("CityHealth cities CSV", "/Users/qiuhaozhu/Desktop/Capstone_Project_Dashboard/CityHealthDash-cities.csv")
    tracts_path = st.text_input("CityHealth tracts CSV", "/Users/qiuhaozhu/Desktop/Capstone_Project_Dashboard/CityHealthDash-tracts.csv")

    st.divider()
    st.header("📍 Geography")
    city_name   = st.text_input("City name", value="Baltimore")
    parent_fips = st.text_input("Parent FIPS (Baltimore=24510)", value="24510")

    st.divider()
    st.header("⚙️ Settings")
    lod_strategy = st.selectbox("Censored results handling", ["LOD/2","LOD/sqrt2","zero","drop"], index=0)
    min_overlap  = st.slider("Min overlapping years for correlations", 3, 10, 5, 1)

# --------------------------- Load water --------------------------- #
water_raw = read_csv_safely(water_path)
if water_raw.empty:
    st.error("Water CSV not found or empty. Check the path.")
    st.stop()

# expect columns: Parameter, Result, datetime (or date), Station, Unit, and coords (varies)
dfw = water_raw.copy()
# parse datetime/year
date_col = "datetime" if "datetime" in dfw.columns else ("Date" if "Date" in dfw.columns else None)
if date_col:
    dfw[date_col] = pd.to_datetime(dfw[date_col], errors="coerce")
    dfw["year"] = dfw[date_col].dt.year
else:
    dfw["year"] = np.nan

# numeric results with LOD handling
if lod_strategy == "drop":
    # drop censored values entirely
    dfw["Result_num"] = pd.to_numeric(dfw["Result"], errors="coerce")
else:
    dfw["Result_num"] = dfw["Result"].apply(lambda x: to_numeric_censored(x, lod_strategy))

dfw["Parameter"] = dfw["Parameter"].astype(str).str.strip()

# --------------------------- Load CHD --------------------------- #
cities = normalize_chd(read_csv_safely(cities_path))
tracts = normalize_chd(read_csv_safely(tracts_path))  # not used directly here

# Filter to chosen city
cities_city = filter_city_frame(cities, city_name, parent_fips)
if cities_city.empty:
    st.warning("No city-level rows matched your city/FIPS in the cities CSV.")
# (tracts kept for potential extensions)

# --------------------------- UI: parameter & metric pickers --------------------------- #
st.title("Water ↔ CityHealth Explorer (Local CSVs)")

left, right = st.columns([1,2])

with left:
    params_all = sorted(dfw["Parameter"].dropna().astype(str).unique().tolist())
    picked_params = st.multiselect("Water parameters", options=params_all, default=[p for p in params_all[:5]])
    metric_all = sorted(cities_city["metric_name"].dropna().astype(str).unique().tolist()) if not cities_city.empty else []
    default_metrics = [m for m in metric_all if "Lead" in m] or metric_all[:5]
    picked_metrics = st.multiselect("CityHealth metrics", options=metric_all, default=default_metrics)
    group = st.selectbox("Group", options=sorted(cities_city["group_name"].dropna().unique()) if "group_name" in cities_city.columns and not cities_city.empty else ["Total"], index=0)

with right:
    st.markdown("<div class='small'>EPA & CHD context: E. coli/enterococci predict GI illness risk in recreational waters; turbidity can shield microbes; CHD Lead Exposure Risk Index reflects housing + poverty, not river water.</div>", unsafe_allow_html=True)

# Filter water to picked params and aggregate yearly medians/means
dfw_use = dfw[dfw["Parameter"].isin(picked_params)] if picked_params else dfw.copy()
water_year = aggregate_water_by_year(dfw_use, value_col="Result_num", how="median")

# Reshape CHD to wide by year for selected metrics & group
cities_city_group = cities_city[cities_city["group_name"].str.lower() == group.lower()] if not cities_city.empty and "group_name" in cities_city.columns else cities_city
chd_wide = chd_city_wide(cities_city_group[cities_city_group["metric_name"].isin(picked_metrics)] if picked_metrics else cities_city_group,
                         group=group, picked_metrics=picked_metrics)

# --------------------------- Checks & info --------------------------- #
if water_year.empty:
    st.warning("No annual aggregates produced for the chosen water parameters (check dates/Parameter names).")
if chd_wide.empty:
    st.warning("No CityHealth data for the chosen city/group/metrics (check cities CSV and filters).")

if water_year.empty or chd_wide.empty:
    st.stop()

# Join on year
joined = water_year.join(chd_wide, how="inner")
if joined.empty:
    st.info("No overlapping years between water series and CHD metrics. Showing outer-joined view for inspection below.")
    joined_outer = water_year.join(chd_wide, how="outer").sort_index()
    st.dataframe(joined_outer, use_container_width=True)
    st.stop()

# --------------------------- KPIs --------------------------- #
c1,c2,c3 = st.columns(3)
with c1:
    st.markdown("**Years overlapped**")
    st.markdown(f"<div class='metric-card'>{joined.shape[0]}</div>", unsafe_allow_html=True)
with c2:
    st.markdown("**Water series**")
    st.markdown(f"<div class='metric-card'>{sum(col.startswith('Water__') for col in joined.columns)}</div>", unsafe_allow_html=True)
with c3:
    st.markdown("**CHD metrics**")
    st.markdown(f"<div class='metric-card'>{sum(col.startswith('CHD__') for col in joined.columns)}</div>", unsafe_allow_html=True)

# --------------------------- Correlations --------------------------- #
corr_df = []
water_cols = [c for c in joined.columns if c.startswith("Water__")]
chd_cols   = [c for c in joined.columns if c.startswith("CHD__")]
for w in water_cols:
    for c in chd_cols:
        xy = joined[[w,c]].dropna()
        if len(xy) >= min_overlap:
            pear = xy[w].corr(xy[c], method="pearson")
            spear, pval = spearmanr(xy[w], xy[c], nan_policy="omit")
            corr_df.append({"water_series": w, "chd_metric": c, "n_years": len(xy),
                            "pearson_r": pear, "spearman_r": spear, "p_spearman": pval})

corr_df = pd.DataFrame(corr_df).sort_values("pearson_r", ascending=False) if corr_df else pd.DataFrame(
    columns=["water_series","chd_metric","n_years","pearson_r","spearman_r","p_spearman"]
)

st.subheader("Correlation results")
st.caption("Pearson and Spearman shown; require min overlapping years set in the sidebar.")
st.dataframe(corr_df, use_container_width=True)

# Heatmap (Pearson)
if not corr_df.empty:
    # Build matrix
    heat = pd.DataFrame(index=water_cols, columns=chd_cols, dtype=float)
    for _, r in corr_df.iterrows():
        heat.loc[r["water_series"], r["chd_metric"]] = r["pearson_r"]
    heat = heat.sort_index(axis=0).sort_index(axis=1)
    fig = px.imshow(heat, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                    labels=dict(color="Pearson r"), aspect="auto", height=420)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No pairs met the overlapping-year threshold for correlations.")

# --------------------------- Scatter (choose a pair) --------------------------- #
st.subheader("Scatter (pick a pair)")
if not corr_df.empty:
    # build human-readable labels
    corr_df["pair_label"] = corr_df.apply(
        lambda r: f"{r['chd_metric']}  vs  {r['water_series']}  (n={int(r['n_years'])}, r={r['pearson_r']:.2f})", axis=1
    )
    choice = st.selectbox("Correlation pair", options=corr_df["pair_label"].tolist(), index=0)
    row = corr_df.loc[corr_df["pair_label"] == choice].iloc[0]
    wcol, ccol = row["water_series"], row["chd_metric"]
    xy = joined[[wcol, ccol]].dropna()
    fig2 = px.scatter(xy, x=wcol, y=ccol, trendline="ols", height=420)
    st.plotly_chart(fig2, use_container_width=True)

# --------------------------- Raw tables (optional) --------------------------- #
with st.expander("Show annual tables (joined / water / CHD)"):
    st.write("Joined (years with both):")
    st.dataframe(joined, use_container_width=True)
    st.write("Water annual aggregates:")
    st.dataframe(water_year, use_container_width=True)
    st.write("CHD annual (city level):")
    st.dataframe(chd_wide, use_container_width=True)

st.divider()
st.caption("Context: Recreational pathogen indicators (E. coli/enterococci), turbidity & disinfection, and CHD Lead Exposure Risk Index methodology per cited EPA & CHD sources. Correlations are exploratory only.")
