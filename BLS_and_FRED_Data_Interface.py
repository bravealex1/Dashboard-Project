# import os
# import json
# from datetime import datetime
# from typing import List, Dict, Optional, Tuple, Iterable

# import requests
# import pandas as pd
# import numpy as np
# import streamlit as st
# import plotly.express as px
# from dotenv import load_dotenv, find_dotenv

# # Load .env
# load_dotenv(find_dotenv(), override=False)

# APP_TITLE = "Baltimore Economic Dashboards (BLS + Federal Reserve) with Chatbot"
# st.set_page_config(page_title=APP_TITLE, layout="wide")

# # ---- Date window (7 most recent calendar years) ----
# TODAY = datetime.now()
# CURRENT_YEAR = TODAY.year
# START_YEAR = str(CURRENT_YEAR - 6)
# END_YEAR   = str(CURRENT_YEAR)

# def month_period_to_date(year: str, period: str) -> Optional[pd.Timestamp]:
#     if period and period.startswith("M") and period != "M13":
#         m = int(period[1:])
#         return pd.Timestamp(year=int(year), month=m, day=1)
#     return None

# # ---- Endpoints ----
# BLS_TIMESERIES_URL   = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
# FRED_SERIES_OBS_URL  = "https://api.stlouisfed.org/fred/series/observations"
# FRED_MAPS_SERIES_DATA= "https://api.stlouisfed.org/geofred/series/data"
# FRED_MAPS_REGIONAL   = "https://api.stlouisfed.org/geofred/regional/data"
# FRED_MAPS_GROUP      = "https://api.stlouisfed.org/geofred/series/group"
# FRED_MAPS_SHAPES     = "https://api.stlouisfed.org/geofred/shapes/file"

# # ---- Series IDs ----
# LAUS_UR_MSA     = "LAUMT241258000000003"
# LAUS_UNEMP_MSA  = "LAUMT241258000000004"
# LAUS_EMP_MSA    = "LAUMT241258000000005"
# LAUS_LF_MSA     = "LAUMT241258000000006"
# CES_TNF_MSA     = "SMU24125800000000001"
# CPIU_BALT_ALL   = "CUURS35ESA0"

# FRED_SERIES = {
#     "HPI (FHFA) Baltimore MSA (quarterly, NSA)": "ATNHPIUS12580Q",
#     "Unemployment Rate, Baltimore MSA (monthly, NSA)": "BALT524URN",
#     "Labor Force, Baltimore MSA (monthly, NSA)": "BALT524LFN",
# }

# # Any county series in the unemployment rate group works as a seed
# EXAMPLE_FRED_COUNTY_UR = "MDBALT5URN"

# # ---- Keys from .env ----
# def require_keys_from_env() -> Tuple[str, str]:
#     fred_key = os.getenv("FRED_API_KEY", "").strip()
#     bls_key  = os.getenv("BLS_API_KEY", "").strip()
#     missing = [k for k,v in (("BLS_API_KEY", bls_key), ("FRED_API_KEY", fred_key)) if not v]
#     if missing:
#         st.error(
#             "Missing required API keys in `.env`: "
#             + ", ".join(missing)
#             + "\n\nAdd:\nBLS_API_KEY=...\nFRED_API_KEY=..."
#         )
#         st.stop()
#     with st.sidebar:
#         st.success("Loaded API keys from .env")
#         st.caption("Using environment variables for configuration.")
#     return fred_key, bls_key

# # ---- BLS helpers ----
# def bls_fetch_timeseries(
#     series_ids: List[str],
#     start_year: str,
#     end_year: str,
#     registration_key: Optional[str] = None,
#     catalog: bool = True,
#     annualaverage: bool = False,
#     calculations: bool = False,
#     aspects: bool = False,
# ) -> Dict:
#     payload = {
#         "seriesid": series_ids,
#         "startyear": start_year,
#         "endyear": end_year,
#         "catalog": catalog,
#         "annualaverage": annualaverage,
#         "calculations": calculations,
#         "aspects": aspects,
#     }
#     if registration_key:
#         payload["registrationkey"] = registration_key
#     r = requests.post(BLS_TIMESERIES_URL, data=json.dumps(payload),
#                       headers={"Content-type":"application/json"}, timeout=60)
#     r.raise_for_status()
#     js = r.json()
#     if js.get("status") != "REQUEST_SUCCEEDED":
#         raise RuntimeError(f"BLS API error: {js.get('message')}")
#     return js

# def bls_to_dataframe(bls_json: Dict) -> pd.DataFrame:
#     res = bls_json.get("Results") or {}
#     series_list = res.get("series") or []
#     rows = []
#     for s in series_list:
#         sid = s.get("seriesID") or s.get("seriesId") or s.get("series_id")
#         cat = s.get("catalog") or {}
#         title = cat.get("series_title") or cat.get("seriesTitle") or sid
#         for obs in s.get("data", []):
#             year   = obs.get("year")
#             period = obs.get("period")
#             dt = month_period_to_date(year, period)
#             if dt is None:
#                 continue
#             val = obs.get("value")
#             rows.append({
#                 "series_id": sid,
#                 "series_title": title,
#                 "year": int(year),
#                 "period": period,
#                 "periodName": obs.get("periodName"),
#                 "value": float(val.replace(",", "")) if isinstance(val, str) else float(val),
#                 "date": dt
#             })
#     return pd.DataFrame(rows).sort_values("date")

# # ---- FRED helpers ----
# def fred_series_observations(series_id: str, api_key: str, start: Optional[str] = None) -> pd.DataFrame:
#     params = {
#         "series_id": series_id,
#         "api_key": api_key,
#         "file_type": "json",
#         "observation_start": start or f"{START_YEAR}-01-01",
#     }
#     r = requests.get(FRED_SERIES_OBS_URL, params=params, timeout=60)
#     r.raise_for_status()
#     js = r.json()
#     df = pd.DataFrame(js.get("observations", []))
#     if df.empty:
#         return df
#     df["value"] = pd.to_numeric(df["value"], errors="coerce")
#     df["date"]  = pd.to_datetime(df["date"])
#     return df[["date","value"]]

# def _fred_maps_group_meta(series_id: str, api_key: str) -> Dict:
#     """
#     Maps API - Series Group Info (returns region_type, series_group, season, units, frequency, min/max_date).
#     """
#     r = requests.get(FRED_MAPS_GROUP,
#                      params={"series_id": series_id, "api_key": api_key, "file_type": "json"},
#                      timeout=60)
#     r.raise_for_status()
#     return (r.json() or {}).get("series_group", {})

# # >>> FIX: Utilities to safely coerce rows to dicts (avoid ValueError seen in your traceback)
# def _as_kv_dict(rec: object) -> Optional[Dict]:
#     """
#     Return a dict if `rec` is (a) already a mapping, or (b) a sequence of (key,value) pairs.
#     Otherwise return None.
#     """
#     if isinstance(rec, dict):
#         return dict(rec)
#     if isinstance(rec, (list, tuple)):
#         try:
#             if all(isinstance(x, (list, tuple)) and len(x) == 2 for x in rec):  # list of pairs
#                 return dict(rec)
#         except Exception:
#             return None
#     return None

# # >>> FIX: Hardened flattener for regional/data JSON (skip non-dict rows)
# def _flatten_regional_json(js: Dict) -> pd.DataFrame:
#     """
#     Expected shape per docs:
#       {
#         "<Title>": {
#            "YYYY" or "YYYY-MM-DD": [ {region, code, value, series_id}, ... ]
#         }
#       }
#     We:
#       - detect the single title key
#       - iterate date keys
#       - keep only items coercible to dicts
#     """
#     if not isinstance(js, dict) or not js:
#         return pd.DataFrame()

#     # Find the first key whose value is a dict (that should be the title node)
#     title_key = None
#     for k, v in js.items():
#         if isinstance(v, dict):
#             title_key = k
#             break
#     if not title_key:
#         return pd.DataFrame()

#     by_date = js.get(title_key, {})
#     if not isinstance(by_date, dict):
#         return pd.DataFrame()

#     rows: List[Dict] = []
#     for date_key, arr in by_date.items():
#         # arr should be a list of observations; if it's a dict, take its values
#         if isinstance(arr, dict):
#             iterable: Iterable = arr.values()
#         elif isinstance(arr, list):
#             iterable = arr
#         else:
#             continue

#         dt = pd.to_datetime(date_key, errors="coerce")
#         for rec in iterable:
#             row = _as_kv_dict(rec)
#             if row is None:   # skip malformed rows instead of crashing
#                 continue
#             row["date"] = dt
#             rows.append(row)

#     return pd.DataFrame(rows)

# def _normalize_code_column(df: pd.DataFrame) -> pd.DataFrame:
#     if "code" in df.columns:
#         return df
#     for alt in ("region_code", "fips", "GEOID", "geoid", "id"):
#         if alt in df.columns:
#             return df.rename(columns={alt: "code"})
#     return df

# def _freq_to_code(freq_str: Optional[str]) -> Optional[str]:
#     if not freq_str:
#         return None
#     s = freq_str.strip().lower()
#     if s in {"d","w","bw","m","q","sa","a","wef","weth","wew","wetu","wem","wesu","wesa","bwew","bwem"}:
#         return s
#     return {
#         "daily":"d","weekly":"w","biweekly":"bw","monthly":"m",
#         "quarterly":"q","semiannual":"sa","semi-annual":"sa","annual":"a",
#     }.get(s)

# def _normalize_date_for_freq(date_str: str, freq_code: Optional[str]) -> str:
#     dt = pd.to_datetime(date_str, errors="coerce")
#     if not isinstance(dt, pd.Timestamp) or pd.isna(dt) or not freq_code:
#         return date_str
#     if freq_code == "a":
#         return f"{dt.year:04d}-01-01"
#     if freq_code == "m":
#         return f"{dt.year:04d}-{dt.month:02d}-01"
#     if freq_code == "q":
#         q_end = dt.quarter * 3
#         return f"{dt.year:04d}-{q_end:02d}-01"
#     return date_str

# def fred_maps_series_cross_section(series_id: str, api_key: str, date: Optional[str] = None
#                                    ) -> Tuple[pd.DataFrame, Optional[str]]:
#     """
#     Try Series Data first (latest cross-section if date omitted).
#     Fallback to Regional Data with all required params from Series Group Info.
#     """
#     # 1) Series Data
#     p_sd = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
#     if date: p_sd["date"] = date
#     r1 = requests.get(FRED_MAPS_SERIES_DATA, params=p_sd, timeout=60)
#     r1.raise_for_status()
#     js1 = r1.json() or {}
#     meta = js1.get("meta", {})
#     region_type = meta.get("region")
#     data = meta.get("data") or []
#     df = _normalize_code_column(pd.DataFrame(data))
#     if not df.empty and "code" in df.columns:
#         if "date" not in df.columns and isinstance(meta.get("date"), str):
#             df["date"] = pd.to_datetime(meta["date"], errors="coerce")
#         return df, region_type

#     # 2) Regional Data (with fully matched params)
#     meta2 = _fred_maps_group_meta(series_id, api_key)
#     region_type = meta2.get("region_type") or region_type
#     series_group = meta2.get("series_group")
#     season = meta2.get("season")
#     units = meta2.get("units")
#     freq_code = _freq_to_code(meta2.get("frequency"))
#     use_date = date or meta2.get("max_date") or meta2.get("min_date")
#     if use_date and freq_code:
#         use_date = _normalize_date_for_freq(use_date, freq_code)

#     if not all([series_group, region_type, season, units, use_date]):
#         return df, region_type

#     p_rd = {
#         "api_key": api_key, "file_type": "json",
#         "series_group": series_group, "region_type": region_type,
#         "season": season, "units": units, "date": use_date,
#     }
#     if freq_code:
#         p_rd["frequency"] = freq_code

#     r2 = requests.get(FRED_MAPS_REGIONAL, params=p_rd, timeout=60)
#     r2.raise_for_status()
#     js2 = r2.json() or {}
#     df2 = _normalize_code_column(_flatten_regional_json(js2))
#     if not df2.empty and "code" in df2.columns:
#         return df2, region_type
#     return df, region_type

# def fred_maps_shapes(shape: str, api_key: str) -> Dict:
#     r = requests.get(FRED_MAPS_SHAPES, params={"shape": shape, "api_key": api_key}, timeout=60)
#     r.raise_for_status()
#     return r.json()

# def _infer_featureidkey_for_county(geojson: Dict) -> str:
#     try:
#         props = geojson["features"][0]["properties"]
#         for k in ("fips","FIPS","GEOID","geoid"):
#             if k in props:
#                 return f"properties.{k}"
#     except Exception:
#         pass
#     return "properties.fips"

# # ---- Dashboards ----
# def bls_dashboard(bls_key: str):
#     st.subheader("📊 BLS — Baltimore (MSA) core indicators")
#     st.write(
#         "BLS v2 time series endpoint (`timeseries/data` POST). "
#         "Series: LAUS unemployment & levels (MSA), CES total nonfarm (MSA), CPI-U (Baltimore area)."
#     )

#     series_ids = [LAUS_UR_MSA, LAUS_UNEMP_MSA, LAUS_EMP_MSA, LAUS_LF_MSA, CES_TNF_MSA, CPIU_BALT_ALL]
#     with st.spinner("Fetching BLS series…"):
#         bls_json = bls_fetch_timeseries(series_ids, START_YEAR, END_YEAR, registration_key=bls_key, catalog=True)
#         df = bls_to_dataframe(bls_json)

#     names = {
#         LAUS_UR_MSA: "Unemployment Rate (%) — LAUS MSA",
#         LAUS_UNEMP_MSA: "Unemployed (persons) — LAUS MSA",
#         LAUS_EMP_MSA: "Employed (persons) — LAUS MSA",
#         LAUS_LF_MSA: "Labor Force (persons) — LAUS MSA",
#         CES_TNF_MSA: "Total Nonfarm Employment (CES, NSA) — MSA",
#         CPIU_BALT_ALL: "CPI-U All Items (NSA) — Baltimore area",
#     }
#     df["metric"] = df["series_id"].map(names).fillna(df["series_id"])

#     chosen = st.multiselect(
#         "Select up to 4 metrics to plot",
#         options=list(names.values()),
#         default=[
#             "Unemployment Rate (%) — LAUS MSA",
#             "Total Nonfarm Employment (CES, NSA) — MSA",
#             "CPI-U All Items (NSA) — Baltimore area",
#         ],
#         max_selections=4,
#     )

#     plot_df = df[df["metric"].isin(chosen)].copy()
#     if plot_df.empty:
#         st.warning("No data for the selected metrics."); return

#     fig = px.line(
#         plot_df, x="date", y="value", color="metric",
#         title=f"BLS Indicators — {START_YEAR}–{END_YEAR}",
#         labels={"value":"Value","date":"Date","metric":"Series"},
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     latest = plot_df.sort_values("date").groupby("metric").tail(1)[["metric","date","value"]]
#     latest["date"] = latest["date"].dt.date.astype(str)
#     st.dataframe(latest.set_index("metric"))

# def fed_dashboard(fred_key: str):
#     st.subheader("🏦 Federal Reserve — FRED & FRED Maps")
#     st.write(
#         "Choropleth uses FRED **Maps** (Series Data; fallback to Regional Data). "
#         "Time series use core **FRED** endpoints."
#     )

#     # Choropleth: Maryland county unemployment
#     with st.spinner("Building Maryland county unemployment choropleth…"):
#         cross, region_type = fred_maps_series_cross_section(EXAMPLE_FRED_COUNTY_UR, fred_key, date=None)
#         if cross.empty:
#             st.warning("No county cross-section returned by FRED Maps.")
#         else:
#             cross = _normalize_code_column(cross)
#             if "code" not in cross.columns:
#                 st.warning("FRED Maps did not return a region code column; cannot render the map.")
#                 return

#             cross["code"] = cross["code"].astype(str).str.zfill(5)
#             md = cross[cross["code"].str[:2] == "24"].copy()
#             if md.empty:
#                 st.warning("No Maryland county rows found in the cross-section."); return

#             county_geo = fred_maps_shapes("county", fred_key)
#             featureidkey = _infer_featureidkey_for_county(county_geo)

#             fig_map = px.choropleth(
#                 md,
#                 geojson=county_geo,
#                 locations="code",
#                 featureidkey=featureidkey,
#                 color="value",
#                 color_continuous_scale="Viridis",
#                 hover_name="region",
#                 title="Maryland County Unemployment Rate (latest available, FRED Maps)",
#             )
#             fig_map.update_geos(fitbounds="locations", visible=False)
#             st.plotly_chart(fig_map, use_container_width=True)

#     # FRED time series
#     col1, col2 = st.columns(2)
#     with col1:
#         series_label = "HPI (FHFA) Baltimore MSA (quarterly, NSA)"
#         sid = FRED_SERIES[series_label]
#         df_hpi = fred_series_observations(sid, fred_key, start=f"{START_YEAR}-01-01")
#         if df_hpi.empty:
#             st.warning("No HPI data returned.")
#         else:
#             st.plotly_chart(
#                 px.line(df_hpi, x="date", y="value",
#                         title=f"{series_label} — {START_YEAR}–{END_YEAR}",
#                         labels={"value":"Index (1995:Q1=100)"}),
#                 use_container_width=True
#             )
#     with col2:
#         series_label2 = "Unemployment Rate, Baltimore MSA (monthly, NSA)"
#         sid2 = FRED_SERIES[series_label2]
#         df_ur = fred_series_observations(sid2, fred_key, start=f"{START_YEAR}-01-01")
#         if df_ur.empty:
#             st.warning("No MSA unemployment (FRED) data returned.")
#         else:
#             st.plotly_chart(
#                 px.line(df_ur, x="date", y="value",
#                         title=f"{series_label2} — {START_YEAR}–{END_YEAR}",
#                         labels={"value":"Percent"}),
#                 use_container_width=True
#             )

# # ---- Chatbot (lite) ----
# SYSTEM_TIP = "You are the Baltimore Economic Assistant. Be concise and prefer the most recent 7 years."

# def init_chat_state():
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = [{"role":"system","content":SYSTEM_TIP}]

# def chatbot_tools(intent: str, fred_key: str, bls_key: str) -> str:
#     try:
#         if intent == "bls_laus":
#             data = bls_fetch_timeseries([LAUS_UR_MSA, LAUS_LF_MSA], START_YEAR, END_YEAR,
#                                         registration_key=bls_key, catalog=True)
#             df = bls_to_dataframe(data)
#             last = df.sort_values("date").groupby("series_id").tail(1)
#             d = {row["series_id"]: (row["date"].date().isoformat(), row["value"]) for _, row in last.iterrows()}
#             ur_date, ur_val = d.get(LAUS_UR_MSA, ("n/a", None))
#             lf_date, lf_val = d.get(LAUS_LF_MSA, ("n/a", None))
#             return f"Latest LAUS (MSA): unemployment rate {ur_val:.1f}% ({ur_date}); labor force {int(lf_val):,} ({lf_date})."
#         if intent == "bls_cpi":
#             data = bls_fetch_timeseries([CPIU_BALT_ALL], START_YEAR, END_YEAR, registration_key=bls_key, catalog=True)
#             df = bls_to_dataframe(data)
#             if df.empty: return "No CPI-U data found."
#             last = df.iloc[-1]
#             yoy = df.set_index("date")["value"].pct_change(12).loc[last["date"]]
#             return f"CPI-U All Items (Baltimore): {last['value']:.1f} as of {last['date'].date().isoformat()}; YoY {yoy*100:.1f}%."
#         if intent == "bls_ces":
#             data = bls_fetch_timeseries([CES_TNF_MSA], START_YEAR, END_YEAR, registration_key=bls_key, catalog=True)
#             df = bls_to_dataframe(data)
#             if df.empty: return "No CES data found."
#             last = df.iloc[-1]
#             return f"Total Nonfarm (CES, MSA): {int(last['value']):,} as of {last['date'].date().isoformat()}."
#         if intent == "fred_hpi":
#             df = fred_series_observations(FRED_SERIES["HPI (FHFA) Baltimore MSA (quarterly, NSA)"], fred_key,
#                                           start=f"{START_YEAR}-01-01")
#             if df.empty: return "No HPI data found."
#             last = df.iloc[-1]
#             return f"FHFA HPI (Baltimore MSA): {last['value']:.2f} as of {last['date'].date().isoformat()}."
#         if intent == "fred_map":
#             cross, _ = fred_maps_series_cross_section(EXAMPLE_FRED_COUNTY_UR, fred_key, date=None)
#             cross = _normalize_code_column(cross)
#             if cross.empty or "code" not in cross.columns:
#                 return "Could not find Maryland county unemployment right now."
#             md = cross[cross["code"].astype(str).str.zfill(5).str[:2] == "24"].copy()
#             if md.empty: return "No Maryland counties found in the map cross-section."
#             top = md.sort_values("value", ascending=False).head(3)[["region","value"]]
#             parts = [f"{r.region}: {r.value:.1f}%" for r in top.itertuples()]
#             return "Latest county unemployment (MD, top 3): " + "; ".join(parts) + "."
#         return "Ask about 'latest unemployment', 'CPI', 'nonfarm employment', 'HPI', or 'show county map'."
#     except Exception as e:
#         return f"Error fetching data: {e}"

# def chatbot(fred_key: str, bls_key: str):
#     st.subheader("🤖 Chatbot — Baltimore Data Assistant")
#     if "chat_history" not in st.session_state:
#         init_chat_state()
#     for m in st.session_state.chat_history:
#         if m["role"] != "system":
#             with st.chat_message(m["role"]):
#                 st.markdown(m["content"])
#     user_msg = st.chat_input("Ask about unemployment, CPI, nonfarm jobs, HPI, or county maps…")
#     if not user_msg:
#         return
#     st.session_state.chat_history.append({"role":"user","content":user_msg})
#     m = user_msg.lower()
#     if any(k in m for k in ["unemployment","laus","labor force"]): intent="bls_laus"
#     elif "cpi" in m or "inflation" in m: intent="bls_cpi"
#     elif "nonfarm" in m or ("employment" in m and "nonfarm" in m): intent="bls_ces"
#     elif "hpi" in m or "house price" in m: intent="fred_hpi"
#     elif "map" in m or "county" in m: intent="fred_map"
#     else: intent="smalltalk"
#     answer = chatbot_tools(intent, fred_key, bls_key)
#     st.session_state.chat_history.append({"role":"assistant","content":answer})
#     with st.chat_message("assistant"):
#         st.markdown(answer)

# # ---- Main ----
# def main():
#     st.title(APP_TITLE)
#     fred_key, bls_key = require_keys_from_env()

#     tabs = st.tabs(["BLS Dashboard", "Federal Reserve Dashboard", "Chatbot"])
#     with tabs[0]: bls_dashboard(bls_key)
#     with tabs[1]: fed_dashboard(fred_key)
#     with tabs[2]: chatbot(fred_key, bls_key)

#     st.markdown("---")
#     st.subheader("⬇️ Quick Exports (CSV)")
#     colA, colB = st.columns(2)
#     with colA:
#         try:
#             bls_json = bls_fetch_timeseries(
#                 [LAUS_UR_MSA, LAUS_UNEMP_MSA, LAUS_EMP_MSA, LAUS_LF_MSA, CES_TNF_MSA, CPIU_BALT_ALL],
#                 START_YEAR, END_YEAR, registration_key=bls_key, catalog=True
#             )
#             df_bls = bls_to_dataframe(bls_json)
#             st.download_button("Download BLS CSV", df_bls.to_csv(index=False), "bls_baltimore.csv", "text/csv")
#         except Exception as e:
#             st.info(f"BLS export unavailable: {e}")
#     with colB:
#         try:
#             df_hpi = fred_series_observations(FRED_SERIES["HPI (FHFA) Baltimore MSA (quarterly, NSA)"], fred_key,
#                                               start=f"{START_YEAR}-01-01")
#             st.download_button("Download FRED HPI CSV", df_hpi.to_csv(index=False), "fred_hpi_baltimore.csv", "text/csv")
#         except Exception as e:
#             st.info(f"FRED export unavailable: {e}")

# if __name__ == "__main__":
#     main()





# Baltimore Economic Dashboards (BLS + Federal Reserve) + Chatbot
# - Robust FRED Maps (Series Data first; Regional Data fallback with matched params)
# - Auto back-off to the nearest published date if a cross-section is empty
# - Auto-detect best GeoJSON feature key (fips/GEOID/etc.)
# - .env for keys (BLS_API_KEY, FRED_API_KEY)
# - Streamlit UI shows plots + underlying tables + download buttons + data status banner




# import os
# import json
# from datetime import datetime
# from typing import List, Dict, Optional, Tuple, Iterable

# import requests
# import pandas as pd
# import numpy as np
# import streamlit as st
# import plotly.express as px
# from dotenv import load_dotenv, find_dotenv
# from dateutil.relativedelta import relativedelta

# # -------------------- App config --------------------
# APP_TITLE = "Baltimore Economic Dashboards (BLS + Federal Reserve) with Chatbot"
# st.set_page_config(page_title=APP_TITLE, layout="wide")

# # -------------------- Keys & Env --------------------
# load_dotenv(find_dotenv(), override=False)

# def require_keys_from_env() -> Tuple[str, str]:
#     fred_key = os.getenv("FRED_API_KEY", "").strip()
#     bls_key  = os.getenv("BLS_API_KEY", "").strip()
#     missing = []
#     if not bls_key:
#         missing.append("BLS_API_KEY")
#     if not fred_key:
#         missing.append("FRED_API_KEY")
#     if missing:
#         st.error(
#             "Missing required API keys in `.env`: "
#             + ", ".join(missing)
#             + "\n\nAdd a .env file with:\nBLS_API_KEY=...\nFRED_API_KEY=..."
#         )
#         st.stop()
#     with st.sidebar:
#         st.success("Loaded API keys from .env")
#         st.caption("Using environment variables for configuration.")
#     return fred_key, bls_key

# # -------------------- Time window (last 7 years) --------------------
# TODAY = datetime.now()
# CURRENT_YEAR = TODAY.year
# START_YEAR = str(CURRENT_YEAR - 6)  # 7 most recent years inclusive
# END_YEAR   = str(CURRENT_YEAR)

# def month_period_to_date(year: str, period: str) -> Optional[pd.Timestamp]:
#     # BLS period codes like M01..M12 (M13 is annual avg; we skip it for monthly charts)
#     if period and period.startswith("M") and period != "M13":
#         m = int(period[1:])
#         return pd.Timestamp(year=int(year), month=m, day=1)
#     return None

# # -------------------- API endpoints --------------------
# # BLS v2: https://www.bls.gov/developers/api_signature_v2.htm
# BLS_TIMESERIES_URL   = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# # FRED docs: https://fred.stlouisfed.org/docs/api/fred/
# # Maps docs (Series Data / Regional Data / Series Group Info / Shapes) are on the same site.
# FRED_SERIES_OBS_URL  = "https://api.stlouisfed.org/fred/series/observations"
# FRED_MAPS_SERIES_DATA= "https://api.stlouisfed.org/geofred/series/data"
# FRED_MAPS_REGIONAL   = "https://api.stlouisfed.org/geofred/regional/data"
# FRED_MAPS_GROUP      = "https://api.stlouisfed.org/geofred/series/group"
# FRED_MAPS_SHAPES     = "https://api.stlouisfed.org/geofred/shapes/file"

# # -------------------- Series IDs (Baltimore) --------------------
# # BLS LAUS / CES / CPI for Baltimore-Columbia-Towson, MD MSA and CPI area
# LAUS_UR_MSA     = "LAUMT241258000000003"  # Unemployment rate (%), MSA
# LAUS_UNEMP_MSA  = "LAUMT241258000000004"  # Unemployed (persons), MSA
# LAUS_EMP_MSA    = "LAUMT241258000000005"  # Employed (persons), MSA
# LAUS_LF_MSA     = "LAUMT241258000000006"  # Labor force (persons), MSA
# CES_TNF_MSA     = "SMU24125800000000001"  # CES Total Nonfarm (NSA), MSA
# CPIU_BALT_ALL   = "CUURS35ESA0"           # CPI-U All Items (NSA), Baltimore area

# # FRED time series for the Fed dashboard
# FRED_SERIES = {
#     "HPI (FHFA) Baltimore MSA (quarterly, NSA)": "ATNHPIUS12580Q",
#     "Unemployment Rate, Baltimore MSA (monthly, NSA)": "BALT524URN",
#     "Labor Force, Baltimore MSA (monthly, NSA)": "BALT524LFN",
# }

# # A county unemployment series (any county series in the group works as a seed)
# EXAMPLE_FRED_COUNTY_UR = "MDBALT5URN"

# # -------------------- BLS helpers --------------------
# def bls_fetch_timeseries(
#     series_ids: List[str],
#     start_year: str,
#     end_year: str,
#     registration_key: Optional[str] = None,
#     catalog: bool = True,
#     annualaverage: bool = False,
#     calculations: bool = False,
#     aspects: bool = False,
# ) -> Dict:
#     payload = {
#         "seriesid": series_ids,
#         "startyear": start_year,
#         "endyear": end_year,
#         "catalog": catalog,
#         "annualaverage": annualaverage,
#         "calculations": calculations,
#         "aspects": aspects,
#     }
#     if registration_key:
#         payload["registrationkey"] = registration_key
#     r = requests.post(
#         BLS_TIMESERIES_URL,
#         data=json.dumps(payload),
#         headers={"Content-type": "application/json"},
#         timeout=60,
#     )
#     r.raise_for_status()
#     js = r.json()
#     if js.get("status") != "REQUEST_SUCCEEDED":
#         raise RuntimeError(f"BLS API error: {js.get('message')}")
#     return js

# def bls_to_dataframe(bls_json: Dict) -> pd.DataFrame:
#     res = bls_json.get("Results") or {}
#     series_list = res.get("series") or []
#     rows = []
#     for s in series_list:
#         sid = s.get("seriesID") or s.get("seriesId") or s.get("series_id")
#         cat = s.get("catalog") or {}
#         title = cat.get("series_title") or cat.get("seriesTitle") or sid
#         for obs in s.get("data", []):
#             year   = obs.get("year")
#             period = obs.get("period")
#             dt = month_period_to_date(year, period)
#             if dt is None:
#                 continue
#             val = obs.get("value")
#             rows.append({
#                 "series_id": sid,
#                 "series_title": title,
#                 "year": int(year),
#                 "period": period,
#                 "periodName": obs.get("periodName"),
#                 "value": float(val.replace(",", "")) if isinstance(val, str) else float(val),
#                 "date": dt,
#             })
#     return pd.DataFrame(rows).sort_values("date")

# # -------------------- FRED helpers --------------------
# def fred_series_observations(series_id: str, api_key: str, start: Optional[str] = None) -> pd.DataFrame:
#     """Core FRED time series (JSON): /fred/series/observations"""
#     params = {
#         "series_id": series_id,
#         "api_key": api_key,
#         "file_type": "json",
#         "observation_start": start or f"{START_YEAR}-01-01",
#     }
#     r = requests.get(FRED_SERIES_OBS_URL, params=params, timeout=60)
#     r.raise_for_status()
#     js = r.json()
#     df = pd.DataFrame(js.get("observations", []))
#     if df.empty:
#         return df
#     df["value"] = pd.to_numeric(df["value"], errors="coerce")
#     df["date"]  = pd.to_datetime(df["date"])
#     return df[["date", "value"]]

# def _fred_maps_group_meta(series_id: str, api_key: str) -> Dict:
#     """Maps API - Series Group Info (region_type, series_group, season, units, frequency, min/max_date)."""
#     params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
#     r = requests.get(FRED_MAPS_GROUP, params=params, timeout=60)
#     r.raise_for_status()
#     return (r.json() or {}).get("series_group", {})

# # ---- robust utilities for Regional Data ----
# def _as_kv_dict(rec: object) -> Optional[Dict]:
#     """Return a dict if rec is already a mapping, or a sequence of (key, value) pairs; else None."""
#     if isinstance(rec, dict):
#         return dict(rec)
#     if isinstance(rec, (list, tuple)):
#         try:
#             if all(isinstance(x, (list, tuple)) and len(x) == 2 for x in rec):
#                 return dict(rec)
#         except Exception:
#             return None
#     return None

# def _flatten_regional_json(js: Dict) -> pd.DataFrame:
#     """
#     Expected shape per docs:
#       { "<Title>": { "<date>": [ {region, code, value, series_id}, ... ] } }
#     We keep only items coercible to dicts (skip malformed rows).
#     """
#     if not isinstance(js, dict) or not js:
#         return pd.DataFrame()

#     # Identify the title node (first key with dict value)
#     title_key = None
#     for k, v in js.items():
#         if isinstance(v, dict):
#             title_key = k
#             break
#     if not title_key:
#         return pd.DataFrame()

#     by_date = js.get(title_key, {})
#     if not isinstance(by_date, dict):
#         return pd.DataFrame()

#     rows: List[Dict] = []
#     for date_key, arr in by_date.items():
#         # arr should be list of observations; handle dict (take values) or list
#         if isinstance(arr, dict):
#             iterable: Iterable = arr.values()
#         elif isinstance(arr, list):
#             iterable = arr
#         else:
#             continue

#         dt = pd.to_datetime(date_key, errors="coerce")
#         for rec in iterable:
#             row = _as_kv_dict(rec)
#             if row is None:
#                 continue  # skip malformed
#             row["date"] = dt
#             rows.append(row)

#     return pd.DataFrame(rows)

# def _normalize_code_column(df: pd.DataFrame) -> pd.DataFrame:
#     """Ensure we have 'code' column (check common alternatives)."""
#     if "code" in df.columns:
#         return df
#     for alt in ("region_code", "fips", "GEOID", "geoid", "id"):
#         if alt in df.columns:
#             return df.rename(columns={alt: "code"})
#     return df

# def _freq_to_code(freq_str: Optional[str]) -> Optional[str]:
#     """Map 'Monthly' -> 'm', 'Quarterly' -> 'q', etc., or pass through if already a code."""
#     if not freq_str:
#         return None
#     s = freq_str.strip().lower()
#     if s in {"d","w","bw","m","q","sa","a","wef","weth","wew","wetu","wem","wesu","wesa","bwew","bwem"}:
#         return s
#     return {
#         "daily":"d","weekly":"w","biweekly":"bw","monthly":"m",
#         "quarterly":"q","semiannual":"sa","semi-annual":"sa","annual":"a",
#     }.get(s)

# def _normalize_date_for_freq(date_str: str, freq_code: Optional[str]) -> str:
#     """Align date to period start for the given frequency code."""
#     dt = pd.to_datetime(date_str, errors="coerce")
#     if not isinstance(dt, pd.Timestamp) or pd.isna(dt) or not freq_code:
#         return date_str
#     if freq_code == "a":
#         return f"{dt.year:04d}-01-01"
#     if freq_code == "m":
#         return f"{dt.year:04d}-{dt.month:02d}-01"
#     if freq_code == "q":
#         q_end = dt.quarter * 3
#         return f"{dt.year:04d}-{q_end:02d}-01"
#     return date_str

# def _step_back_date(iso_date: str, freq_code: str) -> str:
#     """Step back one period in time for the given frequency code."""
#     dt = pd.to_datetime(iso_date, errors="coerce")
#     if pd.isna(dt):
#         return iso_date
#     if freq_code == "m":
#         return (dt - relativedelta(months=1)).strftime("%Y-%m-01")
#     if freq_code == "q":
#         return (dt - relativedelta(months=3)).strftime("%Y-%m-01")
#     if freq_code == "a":
#         return (dt - relativedelta(years=1)).strftime("%Y-01-01")
#     return iso_date

# def fred_maps_series_cross_section(
#     series_id: str,
#     api_key: str,
#     date: Optional[str] = None
# ) -> Tuple[pd.DataFrame, Optional[str], Optional[str], str]:
#     """
#     Return: (df, region_type, used_date, source_endpoint)
#       - Try Series Data first (latest if date omitted). If present and has 'code', return it.
#       - Else, use Regional Data with fully matched params from Series Group Info.
#         If empty, back off up to 6 periods to find a published cross-section.
#     """
#     # 1) Series Data (preferred)
#     params_sd = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
#     if date:
#         params_sd["date"] = date
#     r1 = requests.get(FRED_MAPS_SERIES_DATA, params=params_sd, timeout=60)
#     r1.raise_for_status()
#     js1 = r1.json() or {}
#     meta = js1.get("meta", {})
#     region_type = meta.get("region")
#     data = meta.get("data") or []
#     df = _normalize_code_column(pd.DataFrame(data))
#     used_date = meta.get("date") if isinstance(meta.get("date"), str) else None

#     if not df.empty and "code" in df.columns:
#         if "date" not in df.columns and used_date:
#             df["date"] = pd.to_datetime(used_date, errors="coerce")
#         return df, region_type, used_date, "series_data"

#     # 2) Regional Data fallback (requires matched params)
#     meta2 = _fred_maps_group_meta(series_id, api_key)
#     region_type = meta2.get("region_type") or region_type
#     series_group = meta2.get("series_group")
#     season      = meta2.get("season")
#     units       = meta2.get("units")
#     freq_code   = _freq_to_code(meta2.get("frequency"))
#     # pick a date: user-provided -> max_date -> min_date
#     use_date    = date or meta2.get("max_date") or meta2.get("min_date")
#     if use_date and freq_code:
#         use_date = _normalize_date_for_freq(use_date, freq_code)

#     if not all([series_group, region_type, season, units, use_date]):
#         # nothing we can do; return the (possibly empty) Series Data frame
#         return df, region_type, used_date, "series_data"

#     attempts = 0
#     while attempts < 6:
#         params_rd = {
#             "api_key": api_key,
#             "file_type": "json",
#             "series_group": series_group,
#             "region_type": region_type,
#             "season": season,
#             "units": units,
#             "date": use_date,
#         }
#         if freq_code:
#             params_rd["frequency"] = freq_code

#         r2 = requests.get(FRED_MAPS_REGIONAL, params=params_rd, timeout=60)
#         r2.raise_for_status()
#         js2 = r2.json() or {}
#         df2 = _normalize_code_column(_flatten_regional_json(js2))
#         if not df2.empty and "code" in df2.columns:
#             return df2, region_type, use_date, "regional_data"

#         # try previous period
#         prev_date = _step_back_date(use_date, freq_code or "m")
#         if prev_date == use_date:
#             break
#         use_date = prev_date
#         attempts += 1

#     return df, region_type, used_date, "regional_data"

# def fred_maps_shapes(shape: str, api_key: str) -> Dict:
#     params = {"shape": shape, "api_key": api_key}
#     r = requests.get(FRED_MAPS_SHAPES, params=params, timeout=60)
#     r.raise_for_status()
#     return r.json()

# def _best_featureidkey(geojson: Dict, sample_codes: pd.Series) -> str:
#     """
#     Choose the properties.* key that matches most of our 'code' values.
#     """
#     candidates = ["fips", "FIPS", "GEOID", "geoid", "GEO_ID", "COUNTYFP", "id"]
#     best_key, best_hits = "fips", -1
#     codes = set(sample_codes.astype(str).str.zfill(5))
#     feats = (geojson.get("features") or [])[:500]
#     for key in candidates:
#         hits = 0
#         for f in feats:
#             props = (f or {}).get("properties") or {}
#             val = str(props.get(key, "")).zfill(5)
#             if val in codes:
#                 hits += 1
#         if hits > best_hits:
#             best_hits, best_key = hits, key
#     return f"properties.{best_key}"

# # -------------------- Dashboards --------------------
# def bls_dashboard(bls_key: str):
#     st.subheader("📊 BLS — Baltimore (MSA) core indicators")
#     st.write(
#         "BLS v2 time series endpoint; series include LAUS (MSA) unemployment and levels, CES total nonfarm (MSA), and CPI-U (Baltimore area)."
#     )

#     series_ids = [LAUS_UR_MSA, LAUS_UNEMP_MSA, LAUS_EMP_MSA, LAUS_LF_MSA, CES_TNF_MSA, CPIU_BALT_ALL]
#     with st.spinner("Fetching BLS series…"):
#         bls_json = bls_fetch_timeseries(series_ids, START_YEAR, END_YEAR, registration_key=bls_key, catalog=True)
#         df = bls_to_dataframe(bls_json)

#     nice_names = {
#         LAUS_UR_MSA: "Unemployment Rate (%) — LAUS MSA",
#         LAUS_UNEMP_MSA: "Unemployed (persons) — LAUS MSA",
#         LAUS_EMP_MSA: "Employed (persons) — LAUS MSA",
#         LAUS_LF_MSA: "Labor Force (persons) — LAUS MSA",
#         CES_TNF_MSA: "Total Nonfarm Employment (CES, NSA) — MSA",
#         CPIU_BALT_ALL: "CPI-U All Items (NSA) — Baltimore area",
#     }
#     df["metric"] = df["series_id"].map(nice_names).fillna(df["series_id"])

#     chosen = st.multiselect(
#         "Select up to 4 metrics to plot",
#         options=list(nice_names.values()),
#         default=[
#             "Unemployment Rate (%) — LAUS MSA",
#             "Total Nonfarm Employment (CES, NSA) — MSA",
#             "CPI-U All Items (NSA) — Baltimore area",
#         ],
#         max_selections=4,
#     )

#     plot_df = df[df["metric"].isin(chosen)].copy()
#     if plot_df.empty:
#         st.warning("No data for the selected metrics.")
#         return

#     fig = px.line(
#         plot_df, x="date", y="value", color="metric",
#         title=f"BLS Indicators — {START_YEAR}–{END_YEAR}",
#         labels={"value": "Value", "date": "Date", "metric": "Series"},
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     st.caption("Latest values per selected metric")
#     latest = plot_df.sort_values("date").groupby("metric").tail(1)[["metric","date","value"]]
#     latest["date"] = latest["date"].dt.date.astype(str)
#     st.dataframe(latest.set_index("metric"))

#     with st.expander("Show BLS raw table (all rows)"):
#         st.dataframe(df)

#     st.download_button("Download BLS CSV", df.to_csv(index=False), "bls_baltimore.csv", "text/csv")

# def fed_dashboard(fred_key: str):
#     st.subheader("🏦 Federal Reserve — FRED & FRED Maps")
#     st.write(
#         "Choropleth uses FRED **Maps** (Series Data first; Regional Data fallback with fully matched parameters). "
#         "Time series use core **FRED** endpoints."
#     )

#     # --- Choropleth: Maryland county unemployment (latest available or nearest back-off) ---
#     with st.spinner("Building Maryland county unemployment choropleth…"):
#         cross, region_type, used_date, endpoint = fred_maps_series_cross_section(EXAMPLE_FRED_COUNTY_UR, fred_key, date=None)
#         data_status = f"Maps source: {endpoint.replace('_',' ')} | Region type: {region_type or 'unknown'} | Date used: {used_date or 'latest'} | Rows: {len(cross)}"
#         st.info(data_status)

#         if cross.empty:
#             st.warning("No county cross-section returned by FRED Maps.")
#         else:
#             cross = _normalize_code_column(cross)
#             if "code" not in cross.columns:
#                 st.warning("FRED Maps did not return a region code column; cannot render the map.")
#                 return

#             cross["code"] = cross["code"].astype(str).str.zfill(5)
#             md = cross[cross["code"].str[:2] == "24"].copy()
#             if md.empty:
#                 st.warning("No Maryland county rows found in the cross-section.")
#                 return

#             county_geo = fred_maps_shapes("county", fred_key)
#             featureidkey = _best_featureidkey(county_geo, md["code"])

#             fig_map = px.choropleth(
#                 md,
#                 geojson=county_geo,
#                 locations="code",
#                 featureidkey=featureidkey,
#                 color="value",
#                 color_continuous_scale="Viridis",
#                 hover_name="region",
#                 title=f"Maryland County Unemployment Rate — {used_date or 'latest'} (FRED Maps)",
#             )
#             fig_map.update_geos(fitbounds="locations", visible=False)
#             st.plotly_chart(fig_map, use_container_width=True)

#             with st.expander("Show FRED Maps cross-section (Maryland counties)"):
#                 st.dataframe(md.sort_values("region") )

#             st.download_button("Download FRED Maps cross-section (MD counties)",
#                                md.to_csv(index=False), "fred_maps_md_counties.csv", "text/csv")

#     # --- Time series: FHFA HPI + MSA Unemployment (FRED core) ---
#     col1, col2 = st.columns(2)
#     with col1:
#         label = "HPI (FHFA) Baltimore MSA (quarterly, NSA)"
#         sid = FRED_SERIES[label]
#         df_hpi = fred_series_observations(sid, fred_key, start=f"{START_YEAR}-01-01")
#         if df_hpi.empty:
#             st.warning("No HPI data returned.")
#         else:
#             st.plotly_chart(
#                 px.line(df_hpi, x="date", y="value",
#                         title=f"{label} — {START_YEAR}–{END_YEAR}",
#                         labels={"value": "Index (1995:Q1=100)"}),
#                 use_container_width=True
#             )
#             with st.expander("Show HPI raw series"):
#                 st.dataframe(df_hpi)
#             st.download_button("Download HPI CSV", df_hpi.to_csv(index=False), "fred_hpi_baltimore.csv", "text/csv")

#     with col2:
#         label2 = "Unemployment Rate, Baltimore MSA (monthly, NSA)"
#         sid2 = FRED_SERIES[label2]
#         df_ur = fred_series_observations(sid2, fred_key, start=f"{START_YEAR}-01-01")
#         if df_ur.empty:
#             st.warning("No MSA unemployment (FRED) data returned.")
#         else:
#             st.plotly_chart(
#                 px.line(df_ur, x="date", y="value",
#                         title=f"{label2} — {START_YEAR}–{END_YEAR}",
#                         labels={"value": "Percent"}),
#                 use_container_width=True
#             )
#             with st.expander("Show MSA unemployment raw series"):
#                 st.dataframe(df_ur)
#             st.download_button("Download MSA Unemployment CSV", df_ur.to_csv(index=False), "fred_msa_unemp.csv", "text/csv")

# # -------------------- Chatbot (light agent) --------------------
# SYSTEM_TIP = """
# You are the Baltimore Economic Assistant. Be concise and prefer the most recent 7 years. When asked, fetch BLS/FRED data live.
# """

# def init_chat_state():
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = [{"role": "system", "content": SYSTEM_TIP.strip()}]

# def chatbot_tools(intent: str, fred_key: str, bls_key: str) -> str:
#     # Acts like a tiny agent: chooses a tool & composes a concise answer
#     try:
#         if intent == "bls_laus":
#             data = bls_fetch_timeseries([LAUS_UR_MSA, LAUS_LF_MSA], START_YEAR, END_YEAR, registration_key=bls_key, catalog=True)
#             df = bls_to_dataframe(data)
#             last = df.sort_values("date").groupby("series_id").tail(1)
#             d = {row["series_id"]: (row["date"].date().isoformat(), row["value"]) for _, row in last.iterrows()}
#             ur_date, ur_val = d.get(LAUS_UR_MSA, ("n/a", None))
#             lf_date, lf_val = d.get(LAUS_LF_MSA, ("n/a", None))
#             return f"Latest LAUS (MSA): unemployment rate {ur_val:.1f}% ({ur_date}); labor force {int(lf_val):,} ({lf_date})."
#         if intent == "bls_cpi":
#             data = bls_fetch_timeseries([CPIU_BALT_ALL], START_YEAR, END_YEAR, registration_key=bls_key, catalog=True)
#             df = bls_to_dataframe(data)
#             if df.empty:
#                 return "No CPI-U data found."
#             last = df.iloc[-1]
#             yoy = df.set_index("date")["value"].pct_change(12).loc[last["date"]]
#             return f"CPI-U All Items (Baltimore): {last['value']:.1f} as of {last['date'].date().isoformat()}; YoY {yoy*100:.1f}%."
#         if intent == "bls_ces":
#             data = bls_fetch_timeseries([CES_TNF_MSA], START_YEAR, END_YEAR, registration_key=bls_key, catalog=True)
#             df = bls_to_dataframe(data)
#             if df.empty:
#                 return "No CES data found."
#             last = df.iloc[-1]
#             return f"Total Nonfarm (CES, MSA): {int(last['value']):,} as of {last['date'].date().isoformat()}."
#         if intent == "fred_hpi":
#             df = fred_series_observations(FRED_SERIES["HPI (FHFA) Baltimore MSA (quarterly, NSA)"], fred_key, start=f"{START_YEAR}-01-01")
#             if df.empty:
#                 return "No HPI data found."
#             last = df.iloc[-1]
#             return f"FHFA HPI (Baltimore MSA): {last['value']:.2f} as of {last['date'].date().isoformat()}."
#         if intent == "fred_map":
#             cross, _, used_date, endpoint = fred_maps_series_cross_section(EXAMPLE_FRED_COUNTY_UR, fred_key, date=None)
#             cross = _normalize_code_column(cross)
#             if cross.empty or "code" not in cross.columns:
#                 return "Could not find Maryland county unemployment right now."
#             md = cross[cross["code"].astype(str).str.zfill(5).str[:2] == "24"].copy()
#             if md.empty:
#                 return "No Maryland counties found in the map cross-section."
#             top = md.sort_values("value", ascending=False).head(3)[["region","value"]]
#             parts = [f"{r.region}: {r.value:.1f}%" for r in top.itertuples()]
#             return f"Latest county unemployment (MD, top 3) — {used_date or 'latest'} ({endpoint.replace('_',' ')}): " + "; ".join(parts) + "."
#         return "Try: latest unemployment, CPI, nonfarm employment, HPI, or show county map."
#     except Exception as e:
#         return f"Error fetching data: {e}"

# def chatbot(fred_key: str, bls_key: str):
#     st.subheader("🤖 Chatbot — Baltimore Data Assistant")
#     if "chat_history" not in st.session_state:
#         init_chat_state()

#     for m in st.session_state.chat_history:
#         if m["role"] != "system":
#             with st.chat_message(m["role"]):
#                 st.markdown(m["content"])

#     user_msg = st.chat_input("Ask about unemployment, CPI, nonfarm jobs, HPI, or county maps…")
#     if not user_msg:
#         return
#     st.session_state.chat_history.append({"role": "user", "content": user_msg})

#     m = user_msg.lower()
#     if any(k in m for k in ["unemployment", "laus", "labor force"]):
#         intent = "bls_laus"
#     elif "cpi" in m or "inflation" in m:
#         intent = "bls_cpi"
#     elif "nonfarm" in m or ("employment" in m and "nonfarm" in m):
#         intent = "bls_ces"
#     elif "hpi" in m or "house price" in m:
#         intent = "fred_hpi"
#     elif "map" in m or "county" in m:
#         intent = "fred_map"
#     else:
#         intent = "smalltalk"

#     answer = chatbot_tools(intent, fred_key, bls_key)
#     st.session_state.chat_history.append({"role": "assistant", "content": answer})
#     with st.chat_message("assistant"):
#         st.markdown(answer)

# # -------------------- App entry --------------------
# def main():
#     st.title(APP_TITLE)
#     fred_key, bls_key = require_keys_from_env()

#     tabs = st.tabs(["BLS Dashboard", "Federal Reserve Dashboard", "Chatbot"])
#     with tabs[0]:
#         bls_dashboard(bls_key)
#     with tabs[1]:
#         fed_dashboard(fred_key)
#     with tabs[2]:
#         chatbot(fred_key, bls_key)

# if __name__ == "__main__":
#     main()


# Baltimore Economic Dashboards (BLS + Federal Reserve) + Chatbot
# - Robust FRED Maps parsing:
#     * Flattens nested observations from /geofred/regional/data or /geofred/series/data
#     * Promotes nested columns (e.g., observation_code -> code)
#     * Fallback renames (fips/GEOID/id -> code)
#     * Automatic date back-off when a cross-section isn't yet available
#     * Auto-selects best GeoJSON feature key (fips/GEOID/etc.)
# - Uses .env for BLS_API_KEY and FRED_API_KEY
# - Streamlit UI: plots + full tables + download buttons + status banner

from __future__ import annotations  # Enable postponed evaluation of annotations

import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Iterable, TYPE_CHECKING, Any

import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv, find_dotenv
from dateutil.relativedelta import relativedelta

# Optional: CrewAI for AI chatbot (multi-agent orchestration framework)
# CrewAI enables role-based AI agents that collaborate to complete complex tasks
try:
    # Disable CrewAI telemetry and execution traces prompt BEFORE importing
    # This prevents the "Would you like to view your execution traces? [y/N]" prompt
    os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
    
    from crewai import Agent, Task, Crew
    import crewai
    CREWAI_AVAILABLE = True
    print("[info] ✅ CrewAI loaded successfully - Multi-agent orchestration mode available")
except ImportError as e:
    CREWAI_AVAILABLE = False
    # Define dummy types for when CrewAI is not available
    Agent = None
    Task = None
    Crew = None
    print(f"[info] ⚠️  CrewAI import failed: {e}")
    print("[info] Chatbot will use basic keyword matching mode")
    print("[info] Required package: pip install crewai crewai-tools")
except Exception as e:
    CREWAI_AVAILABLE = False
    Agent = None
    Task = None
    Crew = None
    print(f"[error] Unexpected error loading CrewAI: {e}")
    print("[info] Chatbot will use basic keyword matching mode")

# ================================================================================
# APP CONFIGURATION
# ================================================================================
APP_TITLE = "Baltimore Economic Intelligence Dashboard"
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# -------------------- Keys & Env --------------------
load_dotenv(find_dotenv(), override=False)

def require_keys_from_env() -> Tuple[str, str]:
    fred_key = os.getenv("FRED_API_KEY", "").strip()
    bls_key  = os.getenv("BLS_API_KEY", "").strip()
    missing = []
    if not bls_key:
        missing.append("BLS_API_KEY")
    if not fred_key:
        missing.append("FRED_API_KEY")
    if missing:
        st.error(
            "Missing required API keys in `.env`: "
            + ", ".join(missing)
            + "\n\nAdd a .env file with:\nBLS_API_KEY=...\nFRED_API_KEY=..."
        )
        st.stop()
    with st.sidebar:
        st.success("Loaded API keys from .env")
        st.caption("Using environment variables for configuration.")
    return fred_key, bls_key

# -------------------- Time window (last 7 years) --------------------
TODAY = datetime.now()
CURRENT_YEAR = TODAY.year
START_YEAR = str(CURRENT_YEAR - 6)  # 7 most recent years inclusive
END_YEAR   = str(CURRENT_YEAR)

def month_period_to_date(year: str, period: str) -> Optional[pd.Timestamp]:
    # BLS period codes like M01..M12 (M13 is annual avg; skip it for monthly charts)
    if period and period.startswith("M") and period != "M13":
        m = int(period[1:])
        return pd.Timestamp(year=int(year), month=m, day=1)
    return None

# -------------------- API endpoints --------------------
# BLS v2: https://www.bls.gov/developers/api_signature_v2.htm
BLS_TIMESERIES_URL   = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# FRED & GeoFRED docs:
#  - FRED API: https://fred.stlouisfed.org/docs/api/fred/
#  - Maps Series Data: https://fred.stlouisfed.org/docs/api/geofred/series_data.html
#  - Maps Regional Data: https://fred.stlouisfed.org/docs/api/geofred/regional_data.html
#  - Maps Series Group Info: https://fred.stlouisfed.org/docs/api/geofred/series_group.html
#  - Maps Shapes: https://fred.stlouisfed.org/docs/api/geofred/shapes.html
FRED_SERIES_OBS_URL  = "https://api.stlouisfed.org/fred/series/observations"
FRED_MAPS_SERIES_DATA= "https://api.stlouisfed.org/geofred/series/data"
FRED_MAPS_REGIONAL   = "https://api.stlouisfed.org/geofred/regional/data"
FRED_MAPS_GROUP      = "https://api.stlouisfed.org/geofred/series/group"
FRED_MAPS_SHAPES     = "https://api.stlouisfed.org/geofred/shapes/file"

# ================================================================================
# DATA SERIES DEFINITIONS - ORGANIZED BY CATEGORY
# ================================================================================
# 
# IMPORTANT DISTINCTION BETWEEN BLS and FRED DATA:
# 
# BLS (Bureau of Labor Statistics):
#   - Primary source for employment statistics (official government estimates)
#   - Data comes from surveys: Current Employment Statistics (CES), 
#     Local Area Unemployment Statistics (LAUS)
#   - CES: Based on payroll records from ~145,000 businesses and government agencies
#   - LAUS: Based on Current Population Survey (CPS) household data
#   - Generally considered the "official" numbers for policy and media
#
# FRED (Federal Reserve Economic Data):
#   - Aggregator/repository maintained by St. Louis Federal Reserve
#   - Republishes BLS data PLUS adds Fed-specific analyses and series
#   - Often includes seasonally adjusted versions and custom calculations
#   - May have slight delays compared to BLS direct releases
#   - Includes Fed-specific data (e.g., Financial Stress Index, Regional Price Parities)
#
# WHY BOTH SOURCES APPEAR SIMILAR:
#   - FRED often republishes BLS data for convenience
#   - Example: "BALT524URN" (FRED) ultimately sources from BLS LAUS program
#   - Use BLS for official source-of-truth; FRED for broader context & Fed analyses
# ================================================================================

# ============================================================
# CATEGORY 1: LABOR MARKET - UNEMPLOYMENT & EMPLOYMENT
# ============================================================
# These metrics track job market health in Baltimore MSA

# --- BLS LAUS (Local Area Unemployment Statistics) ---
# Source: BLS Current Population Survey (household survey)
# Region: Baltimore-Columbia-Towson, MD MSA (CBSA code 12580)
# Coverage: Baltimore City + Baltimore, Carroll, Anne Arundel, Howard, Harford, Queen Anne's counties
# Frequency: Monthly, Not Seasonally Adjusted (NSA)
# Note: LAUS provides official unemployment rates used by media and policymakers

LAUS_UR_MSA     = "LAUMT241258000000003"  # Unemployment rate (%) - Baltimore MSA
LAUS_UNEMP_MSA  = "LAUMT241258000000004"  # Number of unemployed persons - Baltimore MSA
LAUS_EMP_MSA    = "LAUMT241258000000005"  # Number of employed persons - Baltimore MSA
LAUS_LF_MSA     = "LAUMT241258000000006"  # Total labor force size - Baltimore MSA

# --- BLS CES (Current Employment Statistics) ---
# Source: BLS establishment survey (employer payroll records)
# Region: Baltimore-Columbia-Towson, MD MSA
# Frequency: Monthly, Not Seasonally Adjusted (NSA)
# Note: CES counts jobs, not people (one person can hold multiple jobs)

CES_TNF_MSA     = "SMU24125800000000001"  # Total Nonfarm Employment - Baltimore MSA
CES_GOODS_MSA   = "SMU24125800000000002"  # Goods-Producing Employment - Baltimore MSA
CES_SERVICE_MSA = "SMU24125800060000001"  # Service-Providing Employment - Baltimore MSA
CES_MANUF_MSA   = "SMU24125803000000001"  # Manufacturing Employment - Baltimore MSA
CES_TRADE_MSA   = "SMU24125804200000001"  # Retail Trade Employment - Baltimore MSA
CES_PROF_MSA    = "SMU24125806054000001"  # Professional & Business Services - Baltimore MSA
CES_HEALTH_MSA  = "SMU24125806562000001"  # Healthcare Employment - Baltimore MSA
CES_LEISURE_MSA = "SMU24125807000000001"  # Leisure & Hospitality Employment - Baltimore MSA
CES_GOVT_MSA    = "SMU24125809000000001"  # Government Employment - Baltimore MSA

# --- FRED Labor Market Series (republished BLS + Fed calculations) ---
# These are FRED's versions/analyses of similar data
FRED_LABOR_SERIES = {
    "Unemployment Rate (FRED)": {
        "series_id": "BALT524URN",
        "region": "Baltimore-Columbia-Towson, MD MSA",
        "frequency": "Monthly, NSA",
        "source": "BLS LAUS (via FRED)",
        "description": "FRED's republication of BLS unemployment rate"
    },
    "Labor Force (FRED)": {
        "series_id": "BALT524LFN",
        "region": "Baltimore-Columbia-Towson, MD MSA",
        "frequency": "Monthly, NSA",
        "source": "BLS LAUS (via FRED)",
        "description": "Total labor force (employed + unemployed)"
    },
    "Employment Level (FRED)": {
        "series_id": "BALT524EMPN",
        "region": "Baltimore-Columbia-Towson, MD MSA",
        "frequency": "Monthly, NSA",
        "source": "BLS LAUS (via FRED)",
        "description": "Number of employed persons"
    },
}

# ============================================================
# CATEGORY 2: PRICES & INFLATION
# ============================================================
# Track cost of living and purchasing power changes

# --- BLS CPI-U (Consumer Price Index for All Urban Consumers) ---
# Source: BLS Consumer Price Survey
# Region: Baltimore-Washington CPI Area (broader than MSA)
# Coverage: Baltimore-Columbia-Towson MSA + Washington-Arlington-Alexandria DC-VA-MD-WV MSA
# Frequency: Monthly, Not Seasonally Adjusted (NSA)
# Base Period: 1982-84 = 100
# Note: Official inflation measure for the region

CPIU_BALT_ALL   = "CUURS35ESA0"           # CPI-U All Items - Baltimore-Washington area
CPIU_BALT_FOOD  = "CUURS35ESAF"           # CPI-U Food - Baltimore-Washington area
CPIU_BALT_HOUSING = "CUURS35ESAH"         # CPI-U Housing - Baltimore-Washington area
CPIU_BALT_TRANSP = "CUURS35ESAT"          # CPI-U Transportation - Baltimore-Washington area
CPIU_BALT_MEDICAL = "CUURS35ESAM"         # CPI-U Medical Care - Baltimore-Washington area
CPIU_BALT_ENERGY = "CUURS35ESAE"          # CPI-U Energy - Baltimore-Washington area

# --- FRED Price/Inflation Series ---
FRED_PRICE_SERIES = {
    "Personal Income Per Capita": {
        "series_id": "PCPI24510",
        "region": "Baltimore City, MD",
        "frequency": "Annual",
        "source": "Bureau of Economic Analysis (via FRED)",
        "description": "Total personal income divided by population"
    },
}

# ============================================================
# CATEGORY 3: HOUSING & REAL ESTATE
# ============================================================
# Track housing market trends and affordability

FRED_HOUSING_SERIES = {
    "Home Price Index (FHFA)": {
        "series_id": "ATNHPIUS12580Q",
        "region": "Baltimore-Columbia-Towson, MD MSA",
        "frequency": "Quarterly, NSA",
        "source": "Federal Housing Finance Agency",
        "description": "House price index tracking home value changes (1995 Q1 = 100)"
    },
    "Median Home Sale Price": {
        "series_id": "MSPUS",  # National series as Baltimore-specific may not be available
        "region": "United States",
        "frequency": "Quarterly",
        "source": "Census Bureau / HUD (via FRED)",
        "description": "Median sales price of houses sold"
    },
    "Home Ownership Rate": {
        "series_id": "RHORUSQ156N",  # US series
        "region": "United States",
        "frequency": "Quarterly, SA",
        "source": "Census Bureau (via FRED)",
        "description": "Percentage of housing units owner-occupied"
    },
}

# ============================================================
# CATEGORY 4: ECONOMIC INDICATORS
# ============================================================
# Broader economic health metrics

FRED_ECONOMIC_SERIES = {
    "Real GDP (Maryland)": {
        "series_id": "MDRGSP",
        "region": "Maryland (State)",
        "frequency": "Annual",
        "source": "Bureau of Economic Analysis (via FRED)",
        "description": "Real Gross Domestic Product for Maryland (millions of 2017 dollars)"
    },
    "Per Capita Real GDP (Maryland)": {
        "series_id": "MDPCRGDP",
        "region": "Maryland (State)",
        "frequency": "Annual",
        "source": "BEA (via FRED)",
        "description": "Real GDP per person"
    },
}

# ============================================================
# CATEGORY 5: GEOGRAPHIC/COUNTY DATA (FRED MAPS)
# ============================================================
# County-level unemployment for choropleth mapping

# Example county unemployment series (Maryland - Baltimore City)
# Note: Any county series in the unemployment group can be used as a "seed"
# to fetch ALL Maryland county unemployment rates via FRED Maps API
EXAMPLE_FRED_COUNTY_UR = "MDBALT5URN"  # Baltimore City unemployment rate

# -------------------- BLS helpers --------------------
def bls_fetch_timeseries(
    series_ids: List[str],
    start_year: str,
    end_year: str,
    registration_key: Optional[str] = None,
    catalog: bool = True,
    annualaverage: bool = False,
    calculations: bool = False,
    aspects: bool = False,
) -> Dict:
    payload = {
        "seriesid": series_ids,
        "startyear": start_year,
        "endyear": end_year,
        "catalog": catalog,
        "annualaverage": annualaverage,
        "calculations": calculations,
        "aspects": aspects,
    }
    if registration_key:
        payload["registrationkey"] = registration_key
    r = requests.post(
        BLS_TIMESERIES_URL,
        data=json.dumps(payload),
        headers={"Content-type": "application/json"},
        timeout=60,
    )
    r.raise_for_status()
    js = r.json()
    if js.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API error: {js.get('message')}")
    return js

def bls_to_dataframe(bls_json: Dict) -> pd.DataFrame:
    res = bls_json.get("Results") or {}
    series_list = res.get("series") or []
    rows = []
    for s in series_list:
        sid = s.get("seriesID") or s.get("seriesId") or s.get("series_id")
        cat = s.get("catalog") or {}
        title = cat.get("series_title") or cat.get("seriesTitle") or sid
        for obs in s.get("data", []):
            year   = obs.get("year")
            period = obs.get("period")
            dt = month_period_to_date(year, period)
            if dt is None:
                continue
            val = obs.get("value")
            rows.append({
                "series_id": sid,
                "series_title": title,
                "year": int(year),
                "period": period,
                "periodName": obs.get("periodName"),
                "value": float(val.replace(",", "")) if isinstance(val, str) else float(val),
                "date": dt,
            })
    return pd.DataFrame(rows).sort_values("date")

# -------------------- FRED helpers --------------------
def fred_series_observations(series_id: str, api_key: str, start: Optional[str] = None) -> pd.DataFrame:
    """Core FRED time series (JSON): /fred/series/observations"""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start or f"{START_YEAR}-01-01",
    }
    r = requests.get(FRED_SERIES_OBS_URL, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js.get("observations", []))
    if df.empty:
        return df
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"]  = pd.to_datetime(df["date"])
    return df[["date", "value"]]

def _fred_maps_group_meta(series_id: str, api_key: str) -> Dict:
    """Maps API - Series Group Info (region_type, series_group, season, units, frequency, min/max_date)."""
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    r = requests.get(FRED_MAPS_GROUP, params=params, timeout=60)
    r.raise_for_status()
    return (r.json() or {}).get("series_group", {})

# ---- robust utilities for Regional / Series Data ----
def _as_kv_dict(rec: object) -> Optional[Dict]:
    """Return a dict if rec is already a mapping, or a sequence of (key, value) pairs; else None."""
    if isinstance(rec, dict):
        return dict(rec)
    if isinstance(rec, (list, tuple)):
        try:
            if all(isinstance(x, (list, tuple)) and len(x) == 2 for x in rec):
                return dict(rec)
        except Exception:
            return None
    return None

def _unwrap_nested_row(row: Dict) -> Dict:
    """
    If the row is like {'observation': {...}} or {'@attributes': {...}}, merge the inner dict up.
    Repeat until there are no single nested-dict keys left.
    """
    changed = True
    while changed:
        changed = False
        keys = list(row.keys())
        if len(keys) == 1 and isinstance(row[keys[0]], dict):
            inner = row[keys[0]]
            row = dict(inner)
            changed = True
        else:
            # also lift known nested containers if present
            for k in keys:
                if isinstance(row.get(k), dict) and any(t in row[k].keys() for t in ("code", "region", "value", "series_id")):
                    inner = row.pop(k)
                    row.update(inner)
                    changed = True
    return row

def _promote_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    After json_normalize, pick the best candidate columns for 'code' and 'region'
    if they exist under names like 'observation_code', 'obs.code', etc.
    """
    def pick_and_rename(target_name: str, keywords=("code", "region")):
        lower = [c for c in df.columns if any(k == c.lower().split(".")[-1] or k == c.lower().split("_")[-1] for k in keywords)]
        if target_name in df.columns:
            return
        # prioritize columns that actually have values
        best = None
        best_nonnull = -1
        for c in df.columns:
            cl = c.lower()
            if cl.endswith(target_name) or target_name in cl:
                nn = int(df[c].notna().sum())
                if nn > best_nonnull:
                    best, best_nonnull = c, nn
        if best:
            df.rename(columns={best: target_name}, inplace=True)

    pick_and_rename("code", keywords=("code",))
    pick_and_rename("region", keywords=("region",))
    return df

def _flatten_maps_payload(js: Dict) -> pd.DataFrame:
    """
    Flatten either:
      - Series Data JSON: {"meta": { ... "data": [ {region, code, value, ...}, ... ]}}
      - Regional Data JSON: { "<Title>": { "<date>": [ {...}, {...} ] } }
    Normalize any nested shapes so we end up with columns including 'code', 'region', 'value', 'date'.
    """
    # Case 1: Series Data shape
    if isinstance(js, dict) and "meta" in js and isinstance(js["meta"], dict):
        meta = js["meta"]
        data = meta.get("data") or []
        date = meta.get("date")
        rows = []
        for rec in data:
            row = _as_kv_dict(rec)
            if row is None:
                continue
            row = _unwrap_nested_row(row)
            row["date"] = pd.to_datetime(date, errors="coerce")
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        df = pd.json_normalize(rows, sep="_")
        df = _promote_nested_columns(df)
        return df

    # Case 2: Regional Data shape
    # Expected: { "<Title>": { "<date>": [ {region, code, value, series_id}, ... ] } }
    if not isinstance(js, dict) or not js:
        return pd.DataFrame()

    title_key = None
    for k, v in js.items():
        if isinstance(v, dict):
            title_key = k
            break
    if not title_key:
        return pd.DataFrame()

    by_date = js.get(title_key, {})
    if not isinstance(by_date, dict):
        return pd.DataFrame()

    rows: List[Dict] = []
    for date_key, arr in by_date.items():
        # arr may be a list of dicts or a dict
        iterable: Iterable
        if isinstance(arr, dict):
            iterable = arr.values()
        elif isinstance(arr, list):
            iterable = arr
        else:
            continue

        dt = pd.to_datetime(date_key, errors="coerce")
        for rec in iterable:
            row = _as_kv_dict(rec)
            if row is None:
                continue
            row = _unwrap_nested_row(row)
            row["date"] = dt
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.json_normalize(rows, sep="_")
    df = _promote_nested_columns(df)
    return df

def _normalize_code_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure we have a 'code' column (check common alternatives)."""
    if "code" in df.columns:
        return df
    for alt in ("region_code", "fips", "FIPS", "GEOID", "geoid", "GEO_ID", "id", "properties.fips"):
        if alt in df.columns:
            return df.rename(columns={alt: "code"})
    # If still missing but we have something like 'observation_code' from a different flattener
    cand = [c for c in df.columns if c.lower().endswith("code")]
    if cand:
        return df.rename(columns={cand[0]: "code"})
    return df

def _freq_to_code(freq_str: Optional[str]) -> Optional[str]:
    """Map 'Monthly' -> 'm', 'Quarterly' -> 'q', etc., or pass through if already a code."""
    if not freq_str:
        return None
    s = freq_str.strip().lower()
    if s in {"d","w","bw","m","q","sa","a","wef","weth","wew","wetu","wem","wesu","wesa","bwew","bwem"}:
        return s
    return {
        "daily":"d","weekly":"w","biweekly":"bw","monthly":"m",
        "quarterly":"q","semiannual":"sa","semi-annual":"sa","annual":"a",
    }.get(s)

def _normalize_date_for_freq(date_str: str, freq_code: Optional[str]) -> str:
    """Align date to period start for the given frequency code."""
    dt = pd.to_datetime(date_str, errors="coerce")
    if not isinstance(dt, pd.Timestamp) or pd.isna(dt) or not freq_code:
        return date_str
    if freq_code == "a":
        return f"{dt.year:04d}-01-01"
    if freq_code == "m":
        return f"{dt.year:04d}-{dt.month:02d}-01"
    if freq_code == "q":
        q_end = dt.quarter * 3
        return f"{dt.year:04d}-{q_end:02d}-01"
    return date_str

def _step_back_date(iso_date: str, freq_code: str) -> str:
    """Step back one period in time for the given frequency code."""
    dt = pd.to_datetime(iso_date, errors="coerce")
    if pd.isna(dt):
        return iso_date
    if freq_code == "m":
        return (dt - relativedelta(months=1)).strftime("%Y-%m-01")
    if freq_code == "q":
        return (dt - relativedelta(months=3)).strftime("%Y-%m-01")
    if freq_code == "a":
        return (dt - relativedelta(years=1)).strftime("%Y-01-01")
    return iso_date

def fred_maps_series_cross_section(
    series_id: str,
    api_key: str,
    date: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[str], Optional[str], str]:
    """
    Return: (df, region_type, used_date, source_endpoint)
      - Try Series Data first (latest if date omitted). If present and has 'code', return it.
      - Else, use Regional Data with fully matched params from Series Group Info.
        If empty, back off up to 6 periods to find a published cross-section.
    """
    # 1) Series Data (preferred)
    params_sd = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    if date:
        params_sd["date"] = date
    r1 = requests.get(FRED_MAPS_SERIES_DATA, params=params_sd, timeout=60)
    r1.raise_for_status()
    js1 = r1.json() or {}
    meta = js1.get("meta", {})
    region_type = meta.get("region")
    used_date = meta.get("date") if isinstance(meta.get("date"), str) else None

    df_sd = _flatten_maps_payload(js1)
    df_sd = _normalize_code_column(df_sd)
    if not df_sd.empty and "code" in df_sd.columns:
        # Series Data returns one cross-section
        return df_sd, region_type, used_date, "series_data"

    # 2) Regional Data fallback (needs matched params) — docs list required params. :contentReference[oaicite:1]{index=1}
    meta2 = _fred_maps_group_meta(series_id, api_key)
    region_type = meta2.get("region_type") or region_type
    series_group = meta2.get("series_group")
    season      = meta2.get("season")
    units       = meta2.get("units")
    freq_code   = _freq_to_code(meta2.get("frequency"))
    # pick a date: user-provided -> max_date -> min_date
    use_date    = date or meta2.get("max_date") or meta2.get("min_date")
    if use_date and freq_code:
        use_date = _normalize_date_for_freq(use_date, freq_code)

    if not all([series_group, region_type, season, units, use_date]):
        return df_sd, region_type, used_date, "series_data"

    attempts = 0
    df_rd: pd.DataFrame = pd.DataFrame()
    while attempts < 6:
        params_rd = {
            "api_key": api_key,
            "file_type": "json",
            "series_group": series_group,
            "region_type": region_type,
            "season": season,
            "units": units,
            "date": use_date,
        }
        if freq_code:
            params_rd["frequency"] = freq_code

        r2 = requests.get(FRED_MAPS_REGIONAL, params=params_rd, timeout=60)
        r2.raise_for_status()
        js2 = r2.json() or {}
        df_rd = _flatten_maps_payload(js2)
        df_rd = _normalize_code_column(df_rd)

        if not df_rd.empty and "code" in df_rd.columns:
            return df_rd, region_type, use_date, "regional_data"

        # try previous period
        prev_date = _step_back_date(use_date, freq_code or "m")
        if prev_date == use_date:
            break
        use_date = prev_date
        attempts += 1

    return df_sd, region_type, used_date, "regional_data"

def fred_maps_shapes(shape: str, api_key: str) -> Dict:
    params = {"shape": shape, "api_key": api_key}
    r = requests.get(FRED_MAPS_SHAPES, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def _best_featureidkey(geojson: Dict, sample_codes: pd.Series) -> str:
    """
    Choose the properties.* key that matches most of our 'code' values.
    """
    candidates = ["fips", "FIPS", "GEOID", "geoid", "GEO_ID", "COUNTYFP", "id"]
    best_key, best_hits = "fips", -1
    codes = set(sample_codes.astype(str).str.zfill(5))
    feats = (geojson.get("features") or [])[:500]
    for key in candidates:
        hits = 0
        for f in feats:
            props = (f or {}).get("properties") or {}
            val = str(props.get(key, "")).zfill(5)
            if val in codes:
                hits += 1
        if hits > best_hits:
            best_hits, best_key = hits, key
    return f"properties.{best_key}"

# ================================================================================
# DASHBOARD SECTIONS - ORGANIZED BY CATEGORY
# ================================================================================

def display_labor_market_dashboard(bls_key: str, fred_key: str):
    """
    Category 1: Labor Market Dashboard
    Shows unemployment rates, employment levels, and industry breakdowns
    """
    st.header("💼 Labor Market Indicators")
    st.markdown("""
    **Geographic Coverage:** Baltimore-Columbia-Towson, MD Metropolitan Statistical Area (MSA)  
    **MSA Code:** 12580 | **Counties:** Baltimore City + Baltimore, Carroll, Anne Arundel, Howard, Harford, Queen Anne's  
    **Data Sources:** BLS LAUS (household survey) + BLS CES (establishment survey) + FRED
    """)
    
    # Tabs for different subcategories
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🏭 By Industry", "📈 Trends"])
    
    with tab1:
        st.subheader("Labor Market Overview")
        
        # Fetch core LAUS data
        laus_series = [LAUS_UR_MSA, LAUS_UNEMP_MSA, LAUS_EMP_MSA, LAUS_LF_MSA]
        with st.spinner("Fetching BLS LAUS data..."):
            bls_json = bls_fetch_timeseries(laus_series, START_YEAR, END_YEAR, 
                                           registration_key=bls_key, catalog=True)
            df_laus = bls_to_dataframe(bls_json)
        
        # Map series IDs to readable names with source attribution
        laus_names = {
            LAUS_UR_MSA: "Unemployment Rate (%) | BLS LAUS",
            LAUS_UNEMP_MSA: "Unemployed Persons | BLS LAUS",
            LAUS_EMP_MSA: "Employed Persons | BLS LAUS",
            LAUS_LF_MSA: "Labor Force | BLS LAUS",
        }
        df_laus["metric"] = df_laus["series_id"].map(laus_names).fillna(df_laus["series_id"])
        
        # Latest values as key metrics
        col1, col2, col3, col4 = st.columns(4)
        latest_data = df_laus.sort_values("date").groupby("series_id").tail(1)
        
        for idx, (sid, row) in enumerate(latest_data.iterrows()):
            metric_name = laus_names.get(row["series_id"], row["series_id"])
            with [col1, col2, col3, col4][idx]:
                if "Rate" in metric_name:
                    st.metric(metric_name.split("|")[0].strip(), 
                             f"{row['value']:.1f}%",
                             delta=None)
                else:
                    st.metric(metric_name.split("|")[0].strip(), 
                             f"{int(row['value']):,}",
                             delta=None)
                st.caption(f"As of {row['date'].date()}")
        
        # Time series plot
        fig = px.line(df_laus, x="date", y="value", color="metric",
                     title=f"Labor Market Trends — {START_YEAR} to {END_YEAR}",
                     labels={"value": "Value", "date": "Date"})
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 View Raw Data"):
            st.dataframe(df_laus)
        
        st.download_button("⬇️ Download LAUS Data (CSV)", 
                          df_laus.to_csv(index=False), 
                          "baltimore_laus_data.csv", 
                          "text/csv")
    
    with tab2:
        st.subheader("Employment by Industry")
        st.markdown("*Source: BLS Current Employment Statistics (CES) - Establishment Survey*")
        
        # Fetch industry employment data
        ces_series = [CES_TNF_MSA, CES_MANUF_MSA, CES_TRADE_MSA, CES_PROF_MSA, 
                     CES_HEALTH_MSA, CES_LEISURE_MSA, CES_GOVT_MSA]
        with st.spinner("Fetching BLS CES industry data..."):
            ces_json = bls_fetch_timeseries(ces_series, START_YEAR, END_YEAR,
                                           registration_key=bls_key, catalog=True)
            df_ces = bls_to_dataframe(ces_json)
        
        ces_names = {
            CES_TNF_MSA: "Total Nonfarm",
            CES_MANUF_MSA: "Manufacturing",
            CES_TRADE_MSA: "Retail Trade",
            CES_PROF_MSA: "Professional & Business Services",
            CES_HEALTH_MSA: "Healthcare",
            CES_LEISURE_MSA: "Leisure & Hospitality",
            CES_GOVT_MSA: "Government",
        }
        df_ces["metric"] = df_ces["series_id"].map(ces_names).fillna(df_ces["series_id"])
        
        # Allow user to select industries to compare
        selected_industries = st.multiselect(
            "Select industries to visualize:",
            options=list(ces_names.values()),
            default=["Total Nonfarm", "Healthcare", "Professional & Business Services"]
        )
        
        plot_df = df_ces[df_ces["metric"].isin(selected_industries)]
        if not plot_df.empty:
            fig2 = px.line(plot_df, x="date", y="value", color="metric",
                          title="Employment by Industry",
                          labels={"value": "Number of Jobs", "date": "Date"})
            fig2.update_layout(hovermode='x unified')
            st.plotly_chart(fig2, use_container_width=True)
            
            # Latest values table
            latest_ind = plot_df.sort_values("date").groupby("metric").tail(1)[["metric", "date", "value"]]
            latest_ind["value"] = latest_ind["value"].apply(lambda x: f"{int(x):,}")
            latest_ind["date"] = latest_ind["date"].dt.date.astype(str)
            st.dataframe(latest_ind.set_index("metric"), use_container_width=True)
        
        st.download_button("⬇️ Download Industry Data (CSV)", 
                          df_ces.to_csv(index=False),
                          "baltimore_ces_industry.csv",
                          "text/csv")
    
    with tab3:
        st.subheader("Historical Trends & Analysis")
        st.info("💡 **Note:** This section shows long-term trends and year-over-year changes")
        
        # Calculate year-over-year changes for unemployment rate
        df_ur = df_laus[df_laus["series_id"] == LAUS_UR_MSA].copy()
        df_ur = df_ur.sort_values("date").set_index("date")
        df_ur["yoy_change"] = df_ur["value"].diff(12)  # 12-month change
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df_ur.index, y=df_ur["value"], 
                                 name="Unemployment Rate", line=dict(color='royalblue')))
        fig3.update_layout(title="Unemployment Rate Trend",
                          yaxis_title="Unemployment Rate (%)",
                          hovermode='x unified')
        st.plotly_chart(fig3, use_container_width=True)


def display_prices_inflation_dashboard(bls_key: str, fred_key: str):
    """
    Category 2: Prices & Inflation Dashboard
    Shows CPI trends and cost of living metrics
    """
    st.header("💰 Prices & Inflation")
    st.markdown("""
    **Geographic Coverage:** Baltimore-Washington CPI Area (broader than MSA)  
    **Coverage:** Baltimore-Columbia-Towson MSA + Washington-Arlington-Alexandria DC-VA-MD-WV MSA  
    **Data Source:** BLS Consumer Price Index for All Urban Consumers (CPI-U)  
    **Base Period:** 1982-84 = 100
    """)
    
    tab1, tab2 = st.tabs(["📊 Overall CPI", "🏷️ By Category"])
    
    with tab1:
        st.subheader("Consumer Price Index - All Items")
        
        with st.spinner("Fetching CPI data..."):
            cpi_json = bls_fetch_timeseries([CPIU_BALT_ALL], START_YEAR, END_YEAR,
                                           registration_key=bls_key, catalog=True)
            df_cpi = bls_to_dataframe(cpi_json)
        
        if not df_cpi.empty:
            # Calculate inflation rate (year-over-year percentage change)
            df_cpi = df_cpi.sort_values("date").copy()
            df_cpi["inflation_rate"] = df_cpi["value"].pct_change(12) * 100  # 12-month % change
            
            # Display latest values
            latest = df_cpi.iloc[-1]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current CPI Index", 
                         f"{latest['value']:.1f}",
                         delta=None)
                st.caption(f"As of {latest['date'].date()}")
            with col2:
                st.metric("Annual Inflation Rate", 
                         f"{latest['inflation_rate']:.1f}%",
                         delta=None)
                st.caption("Year-over-Year Change")
            
            # Dual-axis chart: CPI index and inflation rate
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_cpi["date"], y=df_cpi["value"],
                                    name="CPI Index", yaxis="y1",
                                    line=dict(color='darkblue')))
            fig.add_trace(go.Scatter(x=df_cpi["date"], y=df_cpi["inflation_rate"],
                                    name="Inflation Rate (%)", yaxis="y2",
                                    line=dict(color='red', dash='dot')))
            fig.update_layout(
                title="CPI Index & Inflation Rate",
                yaxis=dict(title="CPI Index", side="left"),
                yaxis2=dict(title="Inflation Rate (%)", overlaying="y", side="right"),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.download_button("⬇️ Download CPI Data (CSV)",
                              df_cpi.to_csv(index=False),
                              "baltimore_cpi_all.csv",
                              "text/csv")
    
    with tab2:
        st.subheader("CPI by Category")
        st.markdown("*Track price changes in specific spending categories*")
        
        cpi_categories = [CPIU_BALT_FOOD, CPIU_BALT_HOUSING, CPIU_BALT_TRANSP, 
                         CPIU_BALT_MEDICAL, CPIU_BALT_ENERGY]
        with st.spinner("Fetching CPI categories..."):
            cpi_cat_json = bls_fetch_timeseries(cpi_categories, START_YEAR, END_YEAR,
                                               registration_key=bls_key, catalog=True)
            df_cpi_cat = bls_to_dataframe(cpi_cat_json)
        
        cat_names = {
            CPIU_BALT_FOOD: "Food",
            CPIU_BALT_HOUSING: "Housing",
            CPIU_BALT_TRANSP: "Transportation",
            CPIU_BALT_MEDICAL: "Medical Care",
            CPIU_BALT_ENERGY: "Energy",
        }
        df_cpi_cat["metric"] = df_cpi_cat["series_id"].map(cat_names).fillna(df_cpi_cat["series_id"])
        
        selected_cats = st.multiselect(
            "Select categories to compare:",
            options=list(cat_names.values()),
            default=["Food", "Housing", "Energy"]
        )
        
        plot_df = df_cpi_cat[df_cpi_cat["metric"].isin(selected_cats)]
        if not plot_df.empty:
            fig2 = px.line(plot_df, x="date", y="value", color="metric",
                          title="CPI by Category",
                          labels={"value": "Index Value", "date": "Date"})
            fig2.update_layout(hovermode='x unified')
            st.plotly_chart(fig2, use_container_width=True)
        
        st.download_button("⬇️ Download Category CPI (CSV)",
                          df_cpi_cat.to_csv(index=False),
                          "baltimore_cpi_categories.csv",
                          "text/csv")

def display_housing_dashboard(fred_key: str):
    """
    Category 3: Housing & Real Estate Dashboard
    Shows home prices, ownership rates, and affordability metrics
    """
    st.header("🏘️ Housing & Real Estate")
    st.markdown("""
    **Data Sources:** Federal Housing Finance Agency (FHFA) via FRED, Census Bureau  
    **Note:** Some metrics are at MSA level, others at national level where Baltimore-specific data unavailable
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Home Price Index (FHFA)")
        st.markdown("*Baltimore-Columbia-Towson MSA | Quarterly*")
        
        hpi_info = FRED_HOUSING_SERIES["Home Price Index (FHFA)"]
        df_hpi = fred_series_observations(hpi_info["series_id"], fred_key, start=f"{START_YEAR}-01-01")
        
        if not df_hpi.empty:
            latest_hpi = df_hpi.iloc[-1]
            # Calculate year-over-year change
            if len(df_hpi) >= 5:  # Need at least 5 quarters
                yoy_change = ((latest_hpi["value"] / df_hpi.iloc[-5]["value"]) - 1) * 100
                st.metric("Current HPI", 
                         f"{latest_hpi['value']:.2f}",
                         delta=f"{yoy_change:+.1f}% YoY")
            else:
                st.metric("Current HPI", f"{latest_hpi['value']:.2f}")
            st.caption(f"Base: 1995 Q1 = 100 | As of {latest_hpi['date'].date()}")
            
            fig_hpi = px.line(df_hpi, x="date", y="value",
                             title=f"Home Price Index Trend ({START_YEAR}-{END_YEAR})",
                             labels={"value": "Index (1995 Q1=100)", "date": "Date"})
            fig_hpi.update_layout(hovermode='x')
            st.plotly_chart(fig_hpi, use_container_width=True)
            
            st.download_button("⬇️ Download HPI Data (CSV)",
                              df_hpi.to_csv(index=False),
                              "baltimore_hpi.csv",
                              "text/csv")
        else:
            st.warning("No HPI data available")
    
    with col2:
        st.subheader("Home Ownership Rate")
        st.markdown("*United States (National) | Quarterly, Seasonally Adjusted*")
        
        own_info = FRED_HOUSING_SERIES["Home Ownership Rate"]
        df_own = fred_series_observations(own_info["series_id"], fred_key, start=f"{START_YEAR}-01-01")
        
        if not df_own.empty:
            latest_own = df_own.iloc[-1]
            st.metric("Home Ownership Rate", 
                     f"{latest_own['value']:.1f}%")
            st.caption(f"As of {latest_own['date'].date()}")
            
            fig_own = px.line(df_own, x="date", y="value",
                             title="Home Ownership Rate Trend",
                             labels={"value": "Ownership Rate (%)", "date": "Date"})
            fig_own.update_layout(hovermode='x')
            st.plotly_chart(fig_own, use_container_width=True)
            
            st.download_button("⬇️ Download Ownership Data (CSV)",
                              df_own.to_csv(index=False),
                              "us_homeownership.csv",
                              "text/csv")
        else:
            st.warning("No ownership rate data available")


def display_county_map_dashboard(fred_key: str):
    """
    Category 5: Geographic/County Data Dashboard
    Shows choropleth maps of Maryland county unemployment rates
    """
    st.header("🗺️ Maryland County Unemployment Map")
    st.markdown("""
    **Data Source:** FRED Maps (GeoFRED) - County-level unemployment rates  
    **Coverage:** All Maryland counties  
    **Note:** Uses latest available cross-section from FRED Maps API
    """)
    
    with st.spinner("Building Maryland county unemployment choropleth..."):
        cross, region_type, used_date, endpoint = fred_maps_series_cross_section(
            EXAMPLE_FRED_COUNTY_UR, fred_key, date=None
        )
        
        # Data status banner
        data_status = (
            f"📊 **Data Info:** Source: {endpoint.replace('_', ' ').title()} | "
            f"Region Type: {region_type or 'Unknown'} | "
            f"Date: {used_date or 'Latest'} | "
            f"Rows Retrieved: {len(cross)}"
        )
        st.info(data_status)
        
        if cross.empty:
            st.warning("⚠️ No county cross-section returned by FRED Maps. The API may be temporarily unavailable.")
            return
        
        cross = _normalize_code_column(cross)
        if "code" not in cross.columns:
            st.error("❌ FRED Maps did not return a region code column. Cannot render map.")
            with st.expander("🔍 Debug: Show returned columns"):
                st.write("Columns received:", list(cross.columns))
                st.dataframe(cross.head())
            return
        
        # Filter to Maryland counties (FIPS starts with "24")
        cross["code"] = cross["code"].astype(str).str.zfill(5)
        md = cross[cross["code"].str[:2] == "24"].copy()
        
        if md.empty:
            st.warning("⚠️ No Maryland county rows found in the cross-section.")
            return
        
        st.success(f"✅ Loaded {len(md)} Maryland counties")
        
        # Fetch county boundaries GeoJSON
        county_geo = fred_maps_shapes("county", fred_key)
        featureidkey = _best_featureidkey(county_geo, md["code"])
        
        # Create choropleth map
        fig_map = px.choropleth(
            md,
            geojson=county_geo,
            locations="code",
            featureidkey=featureidkey,
            color="value",
            color_continuous_scale="Reds",
            hover_name="region",
            title=f"Maryland County Unemployment Rate — {used_date or 'Latest Available'}",
            labels={"value": "Unemployment Rate (%)"}
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(height=600)
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Data table
        with st.expander("📋 View County Data Table"):
            display_df = md.sort_values("value", ascending=False)[["region", "code", "value"]].copy()
            display_df.columns = ["County", "FIPS Code", "Unemployment Rate (%)"]
            st.dataframe(display_df, use_container_width=True)
        
        st.download_button("⬇️ Download County Data (CSV)",
                          md.to_csv(index=False),
                          "maryland_county_unemployment.csv",
                          "text/csv")


# ================================================================================
# AGENTIC CHATBOT CLASS - AI ASSISTANT FOR ECONOMIC DATA
# ================================================================================
# This chatbot uses CrewAI's multi-agent orchestration framework
# Specialized agents collaborate to fetch data, analyze trends, and explain results

class BaltimoreEconomicChatbot:
    """
    AI Assistant for Baltimore Economic Dashboard
    Uses CrewAI multi-agent system with specialized agents for:
    - Data fetching (BLS & FRED APIs)
    - Economic analysis (trends, calculations)
    - Clear communication (explanations, context)
    """
    
    def __init__(self, bls_key: str, fred_key: str):
        """Initialize the chatbot with API keys and CrewAI multi-agent system"""
        self.bls_key = bls_key
        self.fred_key = fred_key
        self._crewai_agents = {}  # Will store individual agents
        self.agent = self._create_agent() if CREWAI_AVAILABLE else None
    
    def _create_agent(self) -> Optional[Crew]:
        """
        Create CrewAI multi-agent crew with specialized roles for economic data analysis.
        
        CrewAI Architecture:
        - Agents: Specialized roles (data fetcher, analyzer, explainer)
        - Tasks: Specific goals assigned to agents
        - Crew: Orchestrates agent collaboration
        
        Returns None if CrewAI/OpenAI not available
        """
        # Check if CrewAI is available
        if not CREWAI_AVAILABLE:
            return None
        
        # Check for OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print("[info] OPENAI_API_KEY not found - agent mode unavailable")
            return None
        
        try:
            # Set OpenAI API key for CrewAI
            os.environ["OPENAI_API_KEY"] = openai_key
            
            # Disable CrewAI telemetry and execution traces prompt
            # This prevents the "Would you like to view your execution traces?" question
            os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
            
            # Define specialized agents with distinct roles
            data_fetcher = Agent(
                role="Economic Data Fetcher",
                goal="Retrieve accurate and timely economic data from BLS and FRED APIs for Baltimore MSA",
                backstory="""You are a specialist in accessing economic databases. You excel at 
                retrieving unemployment rates, employment statistics, inflation data, and housing prices 
                for the Baltimore-Columbia-Towson Metropolitan Statistical Area. You understand the 
                nuances of BLS LAUS, CES, CPI data and FRED's housing and economic indicators.""",
                verbose=False,
                allow_delegation=True,
            )
            
            data_analyzer = Agent(
                role="Economic Data Analyzer",
                goal="Analyze and interpret economic trends and statistics for Baltimore region",
                backstory="""You are an economic analyst who specializes in interpreting labor market 
                data, price trends, and housing statistics. You can calculate year-over-year changes, 
                identify trends, and compare metrics across time periods and geographic regions. You 
                understand the significance of unemployment rates, CPI changes, and home price indices.""",
                verbose=False,
                allow_delegation=True,
            )
            
            data_explainer = Agent(
                role="Economic Data Communicator",
                goal="Explain economic data clearly to users, providing context and insights",
                backstory="""You are an expert at translating complex economic statistics into clear, 
                understandable insights. You explain the difference between BLS and FRED data sources, 
                clarify geographic coverage (MSA vs county vs state), and help users understand what 
                the numbers mean for Baltimore's economy. You always specify dates, sources, and regions.""",
                verbose=False,
                allow_delegation=False,
            )
            
            # Store agents for task creation
            self._crewai_agents = {
                'fetcher': data_fetcher,
                'analyzer': data_analyzer,
                'explainer': data_explainer
            }
            
            # Create the crew (will be used per-query with dynamic tasks)
            crew = Crew(
                agents=[data_fetcher, data_analyzer, data_explainer],
                tasks=[],  # Tasks will be added dynamically per query
                verbose=False
            )
            
            return crew
            
        except Exception as e:
            print(f"[warn] Could not create CrewAI crew: {e}")
            return None
    
    # ========== TOOL FUNCTIONS (called by agent) ==========
    
    def _tool_get_unemployment(self, query: str = "") -> str:
        """Tool: Get latest unemployment statistics"""
        try:
            data = bls_fetch_timeseries([LAUS_UR_MSA, LAUS_UNEMP_MSA, LAUS_EMP_MSA, LAUS_LF_MSA], 
                                       START_YEAR, END_YEAR, 
                                       registration_key=self.bls_key, catalog=True)
            df = bls_to_dataframe(data)
            if df.empty:
                return "No unemployment data available."
            
            last = df.sort_values("date").groupby("series_id").tail(1)
            result_dict = {row["series_id"]: (row["date"].date().isoformat(), row["value"]) 
                          for _, row in last.iterrows()}
            
            ur_date, ur_val = result_dict.get(LAUS_UR_MSA, ("n/a", None))
            unemp_date, unemp_val = result_dict.get(LAUS_UNEMP_MSA, ("n/a", None))
            emp_date, emp_val = result_dict.get(LAUS_EMP_MSA, ("n/a", None))
            lf_date, lf_val = result_dict.get(LAUS_LF_MSA, ("n/a", None))
            
            return f"""**Latest Labor Market Data (Baltimore-Columbia-Towson MSA)**
- **Source:** BLS Local Area Unemployment Statistics (LAUS)
- **Date:** {ur_date}

- **Unemployment Rate:** {ur_val:.1f}%
- **Unemployed Persons:** {int(unemp_val):,}
- **Employed Persons:** {int(emp_val):,}
- **Labor Force:** {int(lf_val):,}

*Geographic Coverage: Baltimore City + Baltimore, Carroll, Anne Arundel, Howard, Harford, Queen Anne's counties*"""
        except Exception as e:
            return f"Error fetching unemployment data: {e}"
    
    def _tool_get_cpi(self, query: str = "") -> str:
        """Tool: Get CPI and inflation"""
        try:
            data = bls_fetch_timeseries([CPIU_BALT_ALL], START_YEAR, END_YEAR,
                                       registration_key=self.bls_key, catalog=True)
            df = bls_to_dataframe(data)
            if df.empty:
                return "No CPI data available."
            
            df = df.sort_values("date")
            last = df.iloc[-1]
            yoy = df.set_index("date")["value"].pct_change(12).loc[last["date"]]
            
            return f"""**Latest Inflation Data (Baltimore-Washington Area)**
- **Source:** BLS Consumer Price Index for All Urban Consumers (CPI-U)
- **Date:** {last['date'].date().isoformat()}

- **CPI Index:** {last['value']:.1f} (Base: 1982-84=100)
- **Annual Inflation Rate:** {yoy*100:.1f}% (Year-over-Year)

*Geographic Coverage: Baltimore-Columbia-Towson MSA + Washington-Arlington-Alexandria DC-VA-MD-WV MSA*
*Note: CPI area is broader than employment MSA to maintain statistical reliability*"""
        except Exception as e:
            return f"Error fetching CPI data: {e}"
    
    def _tool_get_industry_employment(self, query: str = "") -> str:
        """Tool: Get employment by industry"""
        try:
            ces_series = [CES_TNF_MSA, CES_MANUF_MSA, CES_TRADE_MSA, CES_PROF_MSA,
                         CES_HEALTH_MSA, CES_LEISURE_MSA, CES_GOVT_MSA]
            data = bls_fetch_timeseries(ces_series, START_YEAR, END_YEAR,
                                       registration_key=self.bls_key, catalog=True)
            df = bls_to_dataframe(data)
            if df.empty:
                return "No industry employment data available."
            
            last = df.sort_values("date").groupby("series_id").tail(1)
            
            names = {
                CES_TNF_MSA: "Total Nonfarm",
                CES_MANUF_MSA: "Manufacturing",
                CES_TRADE_MSA: "Retail Trade",
                CES_PROF_MSA: "Professional & Business Services",
                CES_HEALTH_MSA: "Healthcare",
                CES_LEISURE_MSA: "Leisure & Hospitality",
                CES_GOVT_MSA: "Government",
            }
            
            result = "**Employment by Industry (Baltimore-Columbia-Towson MSA)**\n"
            result += f"- **Source:** BLS Current Employment Statistics (CES)\n"
            result += f"- **Date:** {last.iloc[0]['date'].date().isoformat()}\n\n"
            
            for _, row in last.iterrows():
                name = names.get(row["series_id"], row["series_id"])
                result += f"- **{name}:** {int(row['value']):,} jobs\n"
            
            result += "\n*Note: CES counts jobs, not people (one person can hold multiple jobs)*"
            return result
        except Exception as e:
            return f"Error fetching industry employment: {e}"
    
    def _tool_get_hpi(self, query: str = "") -> str:
        """Tool: Get home price index"""
        try:
            hpi_series = FRED_HOUSING_SERIES["Home Price Index (FHFA)"]["series_id"]
            df = fred_series_observations(hpi_series, self.fred_key, start=f"{START_YEAR}-01-01")
            if df.empty:
                return "No home price data available."
            
            df = df.sort_values("date")
            last = df.iloc[-1]
            
            # Calculate YoY change if enough data
            if len(df) >= 5:
                yoy_change = ((last["value"] / df.iloc[-5]["value"]) - 1) * 100
                yoy_str = f"\n- **Year-over-Year Change:** {yoy_change:+.1f}%"
            else:
                yoy_str = ""
            
            return f"""**Home Price Index (Baltimore-Columbia-Towson MSA)**
- **Source:** Federal Housing Finance Agency (FHFA) via FRED
- **Date:** {last['date'].date().isoformat()}

- **HPI:** {last['value']:.2f} (Base: 1995 Q1=100){yoy_str}

*Quarterly data tracking home value changes in the Baltimore MSA*"""
        except Exception as e:
            return f"Error fetching home price data: {e}"
    
    def _tool_get_county_map(self, query: str = "") -> str:
        """Tool: Get county unemployment rates"""
        try:
            cross, _, used_date, endpoint = fred_maps_series_cross_section(
                EXAMPLE_FRED_COUNTY_UR, self.fred_key, date=None
            )
            cross = _normalize_code_column(cross)
            if cross.empty or "code" not in cross.columns:
                return "Could not retrieve Maryland county unemployment data at this time."
            
            md = cross[cross["code"].astype(str).str.zfill(5).str[:2] == "24"].copy()
            if md.empty:
                return "No Maryland counties found in FRED Maps data."
            
            # Get top 5 and bottom 5
            md_sorted = md.sort_values("value", ascending=False)
            top5 = md_sorted.head(5)
            bottom5 = md_sorted.tail(5)
            
            result = f"""**Maryland County Unemployment Rates**
- **Source:** FRED Maps ({endpoint.replace('_', ' ').title()})
- **Date:** {used_date or 'Latest Available'}
- **Counties:** {len(md)} Maryland counties

**Highest Unemployment:**
"""
            for _, row in top5.iterrows():
                result += f"- {row['region']}: {row['value']:.1f}%\n"
            
            result += "\n**Lowest Unemployment:**\n"
            for _, row in bottom5.iterrows():
                result += f"- {row['region']}: {row['value']:.1f}%\n"
            
            return result
        except Exception as e:
            return f"Error fetching county data: {e}"
    
    def _tool_compare_sources(self, query: str = "") -> str:
        """Tool: Explain BLS vs FRED"""
        return """**Understanding BLS vs FRED Data Sources**

**BLS (Bureau of Labor Statistics):**
- Official U.S. government source for labor statistics
- Conducts surveys: Current Employment Statistics (CES), Local Area Unemployment Statistics (LAUS)
- CES: Establishment survey of ~145,000 businesses (counts jobs)
- LAUS: Household survey (Current Population Survey - counts people)
- Generally considered the "source of truth" for employment data
- Used by policymakers, media, and economists

**FRED (Federal Reserve Economic Data):**
- Maintained by the Federal Reserve Bank of St. Louis
- Aggregator/repository of economic data
- Republishes BLS data for convenience
- ALSO includes Fed-specific analyses and series
- May have slight delays vs. BLS direct releases
- Adds value through: Maps API, custom calculations, seasonally adjusted series

**Key Point:** 
When FRED shows employment/unemployment data, it often originates from BLS but is being served through FRED's infrastructure. For this Baltimore dashboard:
- We use BLS directly for official source-of-truth statistics
- We use FRED for housing data (FHFA), state GDP, and geographic mapping

**Why Both?**
- BLS: Best for official labor market statistics
- FRED: Best for broader economic context, housing, and geographic visualizations"""
    
    def chat(self, message: str, chat_history: Optional[List] = None) -> str:
        """
        Process a user message using CrewAI multi-agent system and return a response.
        
        CrewAI Pattern:
        1. Create task dynamically based on user query
        2. Assign task to appropriate agent(s)
        3. Crew.kickoff() orchestrates agent collaboration
        4. Return the final result
        
        Args:
            message: User's input message
            chat_history: List of previous messages (not used in CrewAI pattern)
        
        Returns:
            Assistant's response string
        """
        if chat_history is None:
            chat_history = []
        
        # Try CrewAI agent system first
        if self.agent is not None and CREWAI_AVAILABLE:
            try:
                # Determine which tool/data the user needs
                query_type = self._identify_query_type(message)
                
                # Fetch the data using appropriate tool
                data_result = self._execute_tool(query_type, message)
                
                # Create dynamic tasks for the crew
                fetch_task = Task(
                    description=f"Review the fetched data: {data_result[:500]}...",
                    expected_output="Confirmation that data has been reviewed and is ready for analysis",
                    agent=self._crewai_agents['fetcher']
                )
                
                explain_task = Task(
                    description=f"Explain this economic data to the user in a clear, helpful way. User asked: '{message}'. Data: {data_result}",
                    expected_output="Clear, user-friendly explanation of the economic data with context about dates, sources, and geographic regions",
                    agent=self._crewai_agents['explainer']
                )
                
                # Update crew with dynamic tasks
                self.agent.tasks = [fetch_task, explain_task]
                
                # Execute the crew
                result = self.agent.kickoff()
                
                # CrewAI returns a string result
                return str(result) if result else "I processed your request but couldn't generate a response."
                
            except Exception as e:
                print(f"CrewAI agent error: {e}")
                # Fall back to simple mode
        
        # Fallback: Simple keyword-based routing
        return self._simple_response(message)
    
    def _identify_query_type(self, message: str) -> str:
        """Identify what type of economic data the user is asking about"""
        m = message.lower()
        if any(k in m for k in ["unemployment", "jobless", "labor force"]):
            return "unemployment"
        elif any(k in m for k in ["cpi", "inflation", "price", "cost of living"]):
            return "cpi"
        elif any(k in m for k in ["industry", "sector", "manufacturing", "healthcare"]):
            return "industry"
        elif any(k in m for k in ["home price", "hpi", "house", "real estate", "housing"]):
            return "housing"
        elif any(k in m for k in ["county", "map", "geographic"]):
            return "county"
        elif any(k in m for k in ["bls", "fred", "source", "difference", "data come from"]):
            return "sources"
        return "general"
    
    def _execute_tool(self, query_type: str, query: str) -> str:
        """Execute the appropriate tool based on query type"""
        if query_type == "unemployment":
            return self._tool_get_unemployment(query)
        elif query_type == "cpi":
            return self._tool_get_cpi(query)
        elif query_type == "industry":
            return self._tool_get_industry_employment(query)
        elif query_type == "housing":
            return self._tool_get_hpi(query)
        elif query_type == "county":
            return self._tool_get_county_map(query)
        elif query_type == "sources":
            return self._tool_compare_sources(query)
        return "General economic information about Baltimore requested."
    
    def _simple_response(self, message: str) -> str:
        """Fallback response when agent is unavailable"""
        m = message.lower()
        
        # Route to appropriate tool based on keywords
        if any(k in m for k in ["unemployment", "jobless", "labor force"]):
            return self._tool_get_unemployment()
        elif any(k in m for k in ["cpi", "inflation", "price", "cost of living"]):
            return self._tool_get_cpi()
        elif any(k in m for k in ["industry", "sector", "manufacturing", "healthcare"]):
            return self._tool_get_industry_employment()
        elif any(k in m for k in ["home price", "hpi", "house", "real estate", "housing"]):
            return self._tool_get_hpi()
        elif any(k in m for k in ["county", "map", "geographic"]):
            return self._tool_get_county_map()
        elif any(k in m for k in ["bls", "fred", "source", "difference", "data come from"]):
            return self._tool_compare_sources()
        else:
            return """I can help you with Baltimore economic data! Ask me about:
- 💼 **Unemployment & Labor Market** - "What's the latest unemployment rate?"
- 💰 **Inflation & Prices** - "What's the current inflation rate?"
- 🏭 **Employment by Industry** - "How many jobs are in healthcare?"
- 🏘️ **Home Prices** - "What's happening with home prices?"
- 🗺️ **County Data** - "Show me Maryland county unemployment"
- 📊 **Data Sources** - "What's the difference between BLS and FRED?"

*Note: Agent mode requires OpenAI API key for enhanced responses*"""


def display_chatbot_interface(bls_key: str, fred_key: str):
    """
    Display the integrated chatbot interface
    """
    st.header("🤖 AI Economic Assistant")
    st.markdown("""
    **Your intelligent guide to Baltimore economic data**  
    Ask questions in natural language, and the AI will fetch and explain the data for you.
    """)
    
    # Initialize chatbot if not already done
    if "chatbot" not in st.session_state:
        with st.spinner("Initializing AI Assistant..."):
            st.session_state.chatbot = BaltimoreEconomicChatbot(bls_key, fred_key)
            
            # Determine why agent mode is/isn't available
            if st.session_state.chatbot.agent:
                agent_status = "✅ **Agent Mode Active** - Using CrewAI multi-agent orchestration with GPT-4o"
                st.success(agent_status)
            else:
                # Check why agent isn't available
                if not CREWAI_AVAILABLE:
                    agent_status = "⚠️ **Basic Mode** - CrewAI not installed"
                    st.warning(agent_status)
                    st.info("💡 To enable AI Agent mode, run: `pip install crewai crewai-tools`")
                else:
                    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
                    if not openai_key:
                        agent_status = "⚠️ **Basic Mode** - OpenAI API key not found in .env"
                        st.warning(agent_status)
                        st.info("💡 Add `OPENAI_API_KEY=sk-...` to your .env file to enable AI Agent mode")
                    else:
                        agent_status = "⚠️ **Basic Mode** - Agent initialization failed"
                        st.warning(agent_status)
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    user_input = st.chat_input("Ask me about Baltimore economic data...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # CrewAI handles conversation context internally through task descriptions
                # No need for explicit history conversion like LangChain
                try:
                    response = st.session_state.chatbot.chat(user_input)
                except Exception as e:
                    st.error(f"Agent error: {e}")
                    response = "I encountered an error processing your request. Please try again."
                
                st.markdown(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()  # Rerun to update chat history display

# ================================================================================
# MAIN APPLICATION ENTRY POINT
# ================================================================================

def main():
    """
    Main application function with organized category-based navigation
    """
    # App header
    st.title("📊 " + APP_TITLE)
    st.markdown("""
    **Comprehensive economic intelligence for Baltimore-Columbia-Towson Metropolitan Statistical Area**  
    *Powered by Bureau of Labor Statistics (BLS) and Federal Reserve Economic Data (FRED)*
    """)
    
    # Load API keys
    fred_key, bls_key = require_keys_from_env()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("📂 Navigation")
        st.markdown("Select a category to explore:")
        
        page = st.radio(
            "Choose Dashboard:",
            options=[
                "🏠 Home & Overview",
                "💼 Labor Market",
                "💰 Prices & Inflation",
                "🏘️ Housing & Real Estate",
                "🗺️ Geographic/County Data",
                "🤖 AI Assistant"
            ],
            index=0
        )
        
        st.markdown("---")
        st.caption(f"**Data Period:** {START_YEAR}–{END_YEAR}")
        st.caption(f"**MSA:** Baltimore-Columbia-Towson (12580)")
        st.caption("**CrewAI Status:** " + ("✅ Available" if CREWAI_AVAILABLE else "⚠️ Not Installed"))
        
        # Quick info
        with st.expander("ℹ️ About This Dashboard"):
            st.markdown("""
            **Data Sources:**
            - **BLS**: Official U.S. labor statistics (employment, unemployment, inflation)
            - **FRED**: Federal Reserve data repository (housing, GDP, maps)
            
            **Coverage:**
            - **Labor/Employment**: Baltimore-Columbia-Towson MSA
            - **CPI**: Baltimore-Washington Area
            - **Housing**: Baltimore MSA
            - **Maps**: All Maryland counties
            
            **Features:**
            - ✅ Real-time data fetching
            - ✅ Interactive visualizations
            - ✅ CSV exports
            - ✅ AI chatbot (with OpenAI key)
            """)
    
    # Page routing
    if page == "🏠 Home & Overview":
        st.header("🏠 Welcome to Baltimore Economic Intelligence Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **💼 Labor Market**  
            Unemployment rates, employment levels, and industry breakdowns from BLS LAUS and CES surveys.
            """)
        
        with col2:
            st.info("""
            **💰 Prices & Inflation**  
            Consumer Price Index (CPI) tracking cost of living changes across categories.
            """)
        
        with col3:
            st.info("""
            **🏘️ Housing & Real Estate**  
            Home Price Index (HPI) and ownership rates from FHFA and Census Bureau.
            """)
        
        st.markdown("---")
        
        st.subheader("🎯 Quick Start Guide")
        st.markdown("""
        1. **Use the sidebar** to navigate between categories
        2. **Select metrics** within each dashboard to customize your view
        3. **Download data** as CSV for further analysis
        4. **Ask the AI Assistant** questions in natural language
        
        **Pro Tip:** The AI Assistant can explain data sources, fetch live statistics, and answer questions about regional coverage!
        """)
        
        st.markdown("---")
        
        st.subheader("📋 Understanding BLS vs FRED")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **BLS (Bureau of Labor Statistics)**
            - 🏛️ Official U.S. government agency
            - 📊 Primary source for labor statistics
            - 🔬 Conducts surveys (CES, LAUS, CPI)
            - ✅ "Source of truth" for employment data
            - 📰 Used by policymakers and media
            
            **When to use:** Official employment, unemployment, and inflation statistics
            """)
        
        with col2:
            st.markdown("""
            **FRED (Federal Reserve Economic Data)**
            - 🏦 Maintained by St. Louis Federal Reserve
            - 📚 Aggregator/repository of economic data
            - 🔄 Republishes BLS data + adds Fed analyses
            - 🗺️ Provides geographic mapping tools
            - 📈 Includes housing, GDP, financial indicators
            
            **When to use:** Broader context, housing data, geographic visualization
            """)
        
        st.info("💡 **Important:** FRED often *republishes* BLS data. For example, FRED's unemployment rate for Baltimore ultimately comes from BLS LAUS, but is served through FRED's API for convenience.")
    
    elif page == "💼 Labor Market":
        display_labor_market_dashboard(bls_key, fred_key)
    
    elif page == "💰 Prices & Inflation":
        display_prices_inflation_dashboard(bls_key, fred_key)
    
    elif page == "🏘️ Housing & Real Estate":
        display_housing_dashboard(fred_key)
    
    elif page == "🗺️ Geographic/County Data":
        display_county_map_dashboard(fred_key)
    
    elif page == "🤖 AI Assistant":
        display_chatbot_interface(bls_key, fred_key)
    
    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit | Data from BLS & FRED APIs | AI powered by LangChain & OpenAI")


if __name__ == "__main__":
    main()



# ================================================================================
# Baltimore Economic Dashboard (BLS + Federal Reserve) with AI Chatbot
# ================================================================================
# 
# OVERVIEW:
# This dashboard integrates economic data from two authoritative federal sources:
#   1. Bureau of Labor Statistics (BLS) - Employment, unemployment, inflation data
#   2. Federal Reserve Economic Data (FRED) - Housing prices, regional economic indicators
#
# KEY FEATURES:
#   - Multi-category organization for easy navigation
#   - Regional specificity clearly documented for each metric
#   - Integrated AI chatbot using LangChain agentic workflow
#   - Real-time data fetching with proper error handling
#   - Interactive visualizations with Plotly
#   - CSV export functionality
#
# GEOGRAPHIC COVERAGE:
#   - Primary focus: Baltimore-Columbia-Towson, MD Metropolitan Statistical Area (MSA)
#   - MSA Code: 12580
#   - State FIPS: 24 (Maryland)
#   - Includes: Baltimore City + Baltimore, Carroll, Anne Arundel, Howard, 
#               Harford, and Queen Anne's counties
#
# DATA SOURCES & DOCUMENTATION:
#   - BLS API: https://www.bls.gov/developers/api_signature_v2.htm
#   - FRED API: https://fred.stlouisfed.org/docs/api/fred/
#   - FRED Maps: https://fred.stlouisfed.org/docs/api/geofred/
# ================================================================================
