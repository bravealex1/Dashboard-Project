# from __future__ import annotations  # Enable postponed evaluation of annotations

# import os
# import json
# from datetime import datetime
# from typing import List, Dict, Optional, Tuple, Iterable, TYPE_CHECKING, Any

# import requests
# import pandas as pd
# import numpy as np
# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go
# from dotenv import load_dotenv, find_dotenv
# from dateutil.relativedelta import relativedelta

# os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

# try:
#     from crewai import Agent, Task, Crew, Process
#     from crewai.tools import tool
#     CREWAI_AVAILABLE = True
#     print("[info] CrewAI loaded successfully")
# except ImportError as e:
#     CREWAI_AVAILABLE = False
#     Agent = None
#     Task = None
#     Crew = None
#     tool = None
#     print(f"[info] CrewAI import failed: {e}")
#     print("[info] Install with: pip install crewai")
# except Exception as e:
#     CREWAI_AVAILABLE = False
#     Agent = None
#     Task = None
#     Crew = None
#     tool = None
#     print(f"[error] CrewAI load error: {e}")

# # ================================================================================
# # APP CONFIGURATION
# # ================================================================================
# APP_TITLE = "Baltimore Economic Intelligence Dashboard"
# st.set_page_config(
#     page_title=APP_TITLE,
#     layout="wide",
#     page_icon="📊",
#     initial_sidebar_state="expanded"
# )

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
#     # BLS period codes like M01..M12 (M13 is annual avg; skip it for monthly charts)
#     if period and period.startswith("M") and period != "M13":
#         m = int(period[1:])
#         return pd.Timestamp(year=int(year), month=m, day=1)
#     return None

# # -------------------- API endpoints --------------------
# # BLS v2: https://www.bls.gov/developers/api_signature_v2.htm
# BLS_TIMESERIES_URL   = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# # FRED & GeoFRED docs:
# #  - FRED API: https://fred.stlouisfed.org/docs/api/fred/
# #  - Maps Series Data: https://fred.stlouisfed.org/docs/api/geofred/series_data.html
# #  - Maps Regional Data: https://fred.stlouisfed.org/docs/api/geofred/regional_data.html
# #  - Maps Series Group Info: https://fred.stlouisfed.org/docs/api/geofred/series_group.html
# #  - Maps Shapes: https://fred.stlouisfed.org/docs/api/geofred/shapes.html
# FRED_SERIES_OBS_URL  = "https://api.stlouisfed.org/fred/series/observations"
# FRED_MAPS_SERIES_DATA= "https://api.stlouisfed.org/geofred/series/data"
# FRED_MAPS_REGIONAL   = "https://api.stlouisfed.org/geofred/regional/data"
# FRED_MAPS_GROUP      = "https://api.stlouisfed.org/geofred/series/group"
# FRED_MAPS_SHAPES     = "https://api.stlouisfed.org/geofred/shapes/file"

# # ================================================================================
# # DATA SERIES DEFINITIONS - ORGANIZED BY CATEGORY
# # ================================================================================
# # 
# # IMPORTANT DISTINCTION BETWEEN BLS and FRED DATA:
# # 
# # BLS (Bureau of Labor Statistics):
# #   - Primary source for employment statistics (official government estimates)
# #   - Data comes from surveys: Current Employment Statistics (CES), 
# #     Local Area Unemployment Statistics (LAUS)
# #   - CES: Based on payroll records from ~145,000 businesses and government agencies
# #   - LAUS: Based on Current Population Survey (CPS) household data
# #   - Generally considered the "official" numbers for policy and media
# #
# # FRED (Federal Reserve Economic Data):
# #   - Aggregator/repository maintained by St. Louis Federal Reserve
# #   - Republishes BLS data PLUS adds Fed-specific analyses and series
# #   - Often includes seasonally adjusted versions and custom calculations
# #   - May have slight delays compared to BLS direct releases
# #   - Includes Fed-specific data (e.g., Financial Stress Index, Regional Price Parities)
# #
# # WHY BOTH SOURCES APPEAR SIMILAR:
# #   - FRED often republishes BLS data for convenience
# #   - Example: "BALT524URN" (FRED) ultimately sources from BLS LAUS program
# #   - Use BLS for official source-of-truth; FRED for broader context & Fed analyses
# # ================================================================================

# # ============================================================
# # CATEGORY 1: LABOR MARKET - UNEMPLOYMENT & EMPLOYMENT
# # ============================================================
# # These metrics track job market health in Baltimore MSA

# # --- BLS LAUS (Local Area Unemployment Statistics) ---
# # Source: BLS Current Population Survey (household survey)
# # Region: Baltimore-Columbia-Towson, MD MSA (CBSA code 12580)
# # Coverage: Baltimore City + Baltimore, Carroll, Anne Arundel, Howard, Harford, Queen Anne's counties
# # Frequency: Monthly, Not Seasonally Adjusted (NSA)
# # Note: LAUS provides official unemployment rates used by media and policymakers

# LAUS_UR_MSA     = "LAUMT241258000000003"  # Unemployment rate (%) - Baltimore MSA
# LAUS_UNEMP_MSA  = "LAUMT241258000000004"  # Number of unemployed persons - Baltimore MSA
# LAUS_EMP_MSA    = "LAUMT241258000000005"  # Number of employed persons - Baltimore MSA
# LAUS_LF_MSA     = "LAUMT241258000000006"  # Total labor force size - Baltimore MSA

# # --- BLS CES (Current Employment Statistics) ---
# # Source: BLS establishment survey (employer payroll records)
# # Region: Baltimore-Columbia-Towson, MD MSA
# # Frequency: Monthly, Not Seasonally Adjusted (NSA)
# # Note: CES counts jobs, not people (one person can hold multiple jobs)

# CES_TNF_MSA     = "SMU24125800000000001"  # Total Nonfarm Employment - Baltimore MSA
# CES_GOODS_MSA   = "SMU24125800000000002"  # Goods-Producing Employment - Baltimore MSA
# CES_SERVICE_MSA = "SMU24125800060000001"  # Service-Providing Employment - Baltimore MSA
# CES_MANUF_MSA   = "SMU24125803000000001"  # Manufacturing Employment - Baltimore MSA
# CES_TRADE_MSA   = "SMU24125804200000001"  # Retail Trade Employment - Baltimore MSA
# CES_PROF_MSA    = "SMU24125806054000001"  # Professional & Business Services - Baltimore MSA
# CES_HEALTH_MSA  = "SMU24125806562000001"  # Healthcare Employment - Baltimore MSA
# CES_LEISURE_MSA = "SMU24125807000000001"  # Leisure & Hospitality Employment - Baltimore MSA
# CES_GOVT_MSA    = "SMU24125809000000001"  # Government Employment - Baltimore MSA

# # --- FRED Labor Market Series (republished BLS + Fed calculations) ---
# # These are FRED's versions/analyses of similar data
# FRED_LABOR_SERIES = {
#     "Unemployment Rate (FRED)": {
#         "series_id": "BALT524URN",
#         "region": "Baltimore-Columbia-Towson, MD MSA",
#         "frequency": "Monthly, NSA",
#         "source": "BLS LAUS (via FRED)",
#         "description": "FRED's republication of BLS unemployment rate"
#     },
#     "Labor Force (FRED)": {
#         "series_id": "BALT524LFN",
#         "region": "Baltimore-Columbia-Towson, MD MSA",
#         "frequency": "Monthly, NSA",
#         "source": "BLS LAUS (via FRED)",
#         "description": "Total labor force (employed + unemployed)"
#     },
#     "Employment Level (FRED)": {
#         "series_id": "BALT524EMPN",
#         "region": "Baltimore-Columbia-Towson, MD MSA",
#         "frequency": "Monthly, NSA",
#         "source": "BLS LAUS (via FRED)",
#         "description": "Number of employed persons"
#     },
# }

# # ============================================================
# # CATEGORY 2: PRICES & INFLATION
# # ============================================================
# # Track cost of living and purchasing power changes

# # --- BLS CPI-U (Consumer Price Index for All Urban Consumers) ---
# # Source: BLS Consumer Price Survey
# # Region: Baltimore-Washington CPI Area (broader than MSA)
# # Coverage: Baltimore-Columbia-Towson MSA + Washington-Arlington-Alexandria DC-VA-MD-WV MSA
# # Frequency: Monthly, Not Seasonally Adjusted (NSA)
# # Base Period: 1982-84 = 100
# # Note: Official inflation measure for the region

# CPIU_BALT_ALL   = "CUURS35ESA0"           # CPI-U All Items - Baltimore-Washington area
# CPIU_BALT_FOOD  = "CUURS35ESAF"           # CPI-U Food - Baltimore-Washington area
# CPIU_BALT_HOUSING = "CUURS35ESAH"         # CPI-U Housing - Baltimore-Washington area
# CPIU_BALT_TRANSP = "CUURS35ESAT"          # CPI-U Transportation - Baltimore-Washington area
# CPIU_BALT_MEDICAL = "CUURS35ESAM"         # CPI-U Medical Care - Baltimore-Washington area
# CPIU_BALT_ENERGY = "CUURS35ESAE"          # CPI-U Energy - Baltimore-Washington area

# # --- FRED Price/Inflation Series ---
# FRED_PRICE_SERIES = {
#     "Personal Income Per Capita": {
#         "series_id": "PCPI24510",
#         "region": "Baltimore City, MD",
#         "frequency": "Annual",
#         "source": "Bureau of Economic Analysis (via FRED)",
#         "description": "Total personal income divided by population"
#     },
# }

# # ============================================================
# # CATEGORY 3: HOUSING & REAL ESTATE
# # ============================================================
# # Track housing market trends and affordability

# FRED_HOUSING_SERIES = {
#     "Home Price Index (FHFA)": {
#         "series_id": "ATNHPIUS12580Q",
#         "region": "Baltimore-Columbia-Towson, MD MSA",
#         "frequency": "Quarterly, NSA",
#         "source": "Federal Housing Finance Agency",
#         "description": "House price index tracking home value changes (1995 Q1 = 100)"
#     },
#     "Median Home Sale Price": {
#         "series_id": "MSPUS",  # National series as Baltimore-specific may not be available
#         "region": "United States",
#         "frequency": "Quarterly",
#         "source": "Census Bureau / HUD (via FRED)",
#         "description": "Median sales price of houses sold"
#     },
#     "Home Ownership Rate": {
#         "series_id": "RHORUSQ156N",  # US series
#         "region": "United States",
#         "frequency": "Quarterly, SA",
#         "source": "Census Bureau (via FRED)",
#         "description": "Percentage of housing units owner-occupied"
#     },
# }

# # ============================================================
# # CATEGORY 4: ECONOMIC INDICATORS
# # ============================================================
# # Broader economic health metrics

# FRED_ECONOMIC_SERIES = {
#     "Real GDP (Maryland)": {
#         "series_id": "MDRGSP",
#         "region": "Maryland (State)",
#         "frequency": "Annual",
#         "source": "Bureau of Economic Analysis (via FRED)",
#         "description": "Real Gross Domestic Product for Maryland (millions of 2017 dollars)"
#     },
#     "Per Capita Real GDP (Maryland)": {
#         "series_id": "MDPCRGDP",
#         "region": "Maryland (State)",
#         "frequency": "Annual",
#         "source": "BEA (via FRED)",
#         "description": "Real GDP per person"
#     },
# }

# # ============================================================
# # CATEGORY 5: GEOGRAPHIC/COUNTY DATA (FRED MAPS)
# # ============================================================
# # County-level unemployment for choropleth mapping

# # Example county unemployment series (Maryland - Baltimore City)
# # Note: Any county series in the unemployment group can be used as a "seed"
# # to fetch ALL Maryland county unemployment rates via FRED Maps API
# EXAMPLE_FRED_COUNTY_UR = "MDBALT5URN"  # Baltimore City unemployment rate

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

# # ---- robust utilities for Regional / Series Data ----
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

# def _unwrap_nested_row(row: Dict) -> Dict:
#     """
#     If the row is like {'observation': {...}} or {'@attributes': {...}}, merge the inner dict up.
#     Repeat until there are no single nested-dict keys left.
#     """
#     changed = True
#     while changed:
#         changed = False
#         keys = list(row.keys())
#         if len(keys) == 1 and isinstance(row[keys[0]], dict):
#             inner = row[keys[0]]
#             row = dict(inner)
#             changed = True
#         else:
#             # also lift known nested containers if present
#             for k in keys:
#                 if isinstance(row.get(k), dict) and any(t in row[k].keys() for t in ("code", "region", "value", "series_id")):
#                     inner = row.pop(k)
#                     row.update(inner)
#                     changed = True
#     return row

# def _promote_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     After json_normalize, pick the best candidate columns for 'code' and 'region'
#     if they exist under names like 'observation_code', 'obs.code', etc.
#     """
#     def pick_and_rename(target_name: str, keywords=("code", "region")):
#         lower = [c for c in df.columns if any(k == c.lower().split(".")[-1] or k == c.lower().split("_")[-1] for k in keywords)]
#         if target_name in df.columns:
#             return
#         # prioritize columns that actually have values
#         best = None
#         best_nonnull = -1
#         for c in df.columns:
#             cl = c.lower()
#             if cl.endswith(target_name) or target_name in cl:
#                 nn = int(df[c].notna().sum())
#                 if nn > best_nonnull:
#                     best, best_nonnull = c, nn
#         if best:
#             df.rename(columns={best: target_name}, inplace=True)

#     pick_and_rename("code", keywords=("code",))
#     pick_and_rename("region", keywords=("region",))
#     return df

# def _flatten_maps_payload(js: Dict) -> pd.DataFrame:
#     """
#     Flatten either:
#       - Series Data JSON: {"meta": { ... "data": [ {region, code, value, ...}, ... ]}}
#       - Regional Data JSON: { "<Title>": { "<date>": [ {...}, {...} ] } }
#     Normalize any nested shapes so we end up with columns including 'code', 'region', 'value', 'date'.
#     """
#     # Case 1: Series Data shape
#     if isinstance(js, dict) and "meta" in js and isinstance(js["meta"], dict):
#         meta = js["meta"]
#         data = meta.get("data") or []
#         date = meta.get("date")
#         rows = []
#         for rec in data:
#             row = _as_kv_dict(rec)
#             if row is None:
#                 continue
#             row = _unwrap_nested_row(row)
#             row["date"] = pd.to_datetime(date, errors="coerce")
#             rows.append(row)
#         if not rows:
#             return pd.DataFrame()
#         df = pd.json_normalize(rows, sep="_")
#         df = _promote_nested_columns(df)
#         return df

#     # Case 2: Regional Data shape
#     # Expected: { "<Title>": { "<date>": [ {region, code, value, series_id}, ... ] } }
#     if not isinstance(js, dict) or not js:
#         return pd.DataFrame()

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
#         # arr may be a list of dicts or a dict
#         iterable: Iterable
#         if isinstance(arr, dict):
#             iterable = arr.values()
#         elif isinstance(arr, list):
#             iterable = arr
#         else:
#             continue

#         dt = pd.to_datetime(date_key, errors="coerce")
#         for rec in iterable:
#             row = _as_kv_dict(rec)
#             if row is None:
#                 continue
#             row = _unwrap_nested_row(row)
#             row["date"] = dt
#             rows.append(row)

#     if not rows:
#         return pd.DataFrame()

#     df = pd.json_normalize(rows, sep="_")
#     df = _promote_nested_columns(df)
#     return df

# def _normalize_code_column(df: pd.DataFrame) -> pd.DataFrame:
#     """Ensure we have a 'code' column (check common alternatives)."""
#     if "code" in df.columns:
#         return df
#     for alt in ("region_code", "fips", "FIPS", "GEOID", "geoid", "GEO_ID", "id", "properties.fips"):
#         if alt in df.columns:
#             return df.rename(columns={alt: "code"})
#     # If still missing but we have something like 'observation_code' from a different flattener
#     cand = [c for c in df.columns if c.lower().endswith("code")]
#     if cand:
#         return df.rename(columns={cand[0]: "code"})
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
#     used_date = meta.get("date") if isinstance(meta.get("date"), str) else None

#     df_sd = _flatten_maps_payload(js1)
#     df_sd = _normalize_code_column(df_sd)
#     if not df_sd.empty and "code" in df_sd.columns:
#         # Series Data returns one cross-section
#         return df_sd, region_type, used_date, "series_data"

#     # 2) Regional Data fallback (needs matched params) — docs list required params. :contentReference[oaicite:1]{index=1}
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
#         return df_sd, region_type, used_date, "series_data"

#     attempts = 0
#     df_rd: pd.DataFrame = pd.DataFrame()
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
#         df_rd = _flatten_maps_payload(js2)
#         df_rd = _normalize_code_column(df_rd)

#         if not df_rd.empty and "code" in df_rd.columns:
#             return df_rd, region_type, use_date, "regional_data"

#         # try previous period
#         prev_date = _step_back_date(use_date, freq_code or "m")
#         if prev_date == use_date:
#             break
#         use_date = prev_date
#         attempts += 1

#     return df_sd, region_type, used_date, "regional_data"

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

# # ================================================================================
# # DASHBOARD SECTIONS - ORGANIZED BY CATEGORY
# # ================================================================================

# def display_labor_market_dashboard(bls_key: str, fred_key: str):
#     st.header("Labor Market Indicators")
#     st.markdown("""
#     **Geographic Coverage:** Baltimore-Columbia-Towson, MD Metropolitan Statistical Area (MSA)  
#     **MSA Code:** 12580 | **Counties:** Baltimore City + Baltimore, Carroll, Anne Arundel, Howard, Harford, Queen Anne's  
#     **Data Sources:** BLS LAUS (household survey) + BLS CES (establishment survey) + FRED
#     """)
    
#     tab1, tab2, tab3 = st.tabs(["Overview", "By Industry", "Trends"])
    
#     with tab1:
#         st.subheader("Labor Market Overview")
        
#         # Fetch core LAUS data
#         laus_series = [LAUS_UR_MSA, LAUS_UNEMP_MSA, LAUS_EMP_MSA, LAUS_LF_MSA]
#         with st.spinner("Fetching BLS LAUS data..."):
#             bls_json = bls_fetch_timeseries(laus_series, START_YEAR, END_YEAR, 
#                                            registration_key=bls_key, catalog=True)
#             df_laus = bls_to_dataframe(bls_json)
        
#         # Map series IDs to readable names with source attribution
#         laus_names = {
#             LAUS_UR_MSA: "Unemployment Rate (%) | BLS LAUS",
#             LAUS_UNEMP_MSA: "Unemployed Persons | BLS LAUS",
#             LAUS_EMP_MSA: "Employed Persons | BLS LAUS",
#             LAUS_LF_MSA: "Labor Force | BLS LAUS",
#         }
#         df_laus["metric"] = df_laus["series_id"].map(laus_names).fillna(df_laus["series_id"])
        
#         # Latest values as key metrics
#         col1, col2, col3, col4 = st.columns(4)
#         latest_data = df_laus.sort_values("date").groupby("series_id").tail(1)
        
#         for idx, (sid, row) in enumerate(latest_data.iterrows()):
#             metric_name = laus_names.get(row["series_id"], row["series_id"])
#             with [col1, col2, col3, col4][idx]:
#                 if "Rate" in metric_name:
#                     st.metric(metric_name.split("|")[0].strip(), 
#                              f"{row['value']:.1f}%",
#                              delta=None)
#                 else:
#                     st.metric(metric_name.split("|")[0].strip(), 
#                              f"{int(row['value']):,}",
#                              delta=None)
#                 st.caption(f"As of {row['date'].date()}")
        
#         # Time series plot
#         fig = px.line(df_laus, x="date", y="value", color="metric",
#                      title=f"Labor Market Trends — {START_YEAR} to {END_YEAR}",
#                      labels={"value": "Value", "date": "Date"})
#         fig.update_layout(hovermode='x unified')
#         st.plotly_chart(fig, use_container_width=True)
        
#         with st.expander("View Raw Data"):
#             st.dataframe(df_laus)
        
#         st.download_button("Download LAUS Data (CSV)", 
#                           df_laus.to_csv(index=False), 
#                           "baltimore_laus_data.csv", 
#                           "text/csv")
    
#     with tab2:
#         st.subheader("Employment by Industry")
#         st.markdown("*Source: BLS Current Employment Statistics (CES) - Establishment Survey*")
        
#         # Fetch industry employment data
#         ces_series = [CES_TNF_MSA, CES_MANUF_MSA, CES_TRADE_MSA, CES_PROF_MSA, 
#                      CES_HEALTH_MSA, CES_LEISURE_MSA, CES_GOVT_MSA]
#         with st.spinner("Fetching BLS CES industry data..."):
#             ces_json = bls_fetch_timeseries(ces_series, START_YEAR, END_YEAR,
#                                            registration_key=bls_key, catalog=True)
#             df_ces = bls_to_dataframe(ces_json)
        
#         ces_names = {
#             CES_TNF_MSA: "Total Nonfarm",
#             CES_MANUF_MSA: "Manufacturing",
#             CES_TRADE_MSA: "Retail Trade",
#             CES_PROF_MSA: "Professional & Business Services",
#             CES_HEALTH_MSA: "Healthcare",
#             CES_LEISURE_MSA: "Leisure & Hospitality",
#             CES_GOVT_MSA: "Government",
#         }
#         df_ces["metric"] = df_ces["series_id"].map(ces_names).fillna(df_ces["series_id"])
        
#         # Allow user to select industries to compare
#         selected_industries = st.multiselect(
#             "Select industries to visualize:",
#             options=list(ces_names.values()),
#             default=["Total Nonfarm", "Healthcare", "Professional & Business Services"]
#         )
        
#         plot_df = df_ces[df_ces["metric"].isin(selected_industries)]
#         if not plot_df.empty:
#             fig2 = px.line(plot_df, x="date", y="value", color="metric",
#                           title="Employment by Industry",
#                           labels={"value": "Number of Jobs", "date": "Date"})
#             fig2.update_layout(hovermode='x unified')
#             st.plotly_chart(fig2, use_container_width=True)
            
#             # Latest values table
#             latest_ind = plot_df.sort_values("date").groupby("metric").tail(1)[["metric", "date", "value"]]
#             latest_ind["value"] = latest_ind["value"].apply(lambda x: f"{int(x):,}")
#             latest_ind["date"] = latest_ind["date"].dt.date.astype(str)
#             st.dataframe(latest_ind.set_index("metric"), use_container_width=True)
        
#         st.download_button("Download Industry Data (CSV)", 
#                           df_ces.to_csv(index=False),
#                           "baltimore_ces_industry.csv",
#                           "text/csv")
    
#     with tab3:
#         st.subheader("Historical Trends & Analysis")
#         st.info("Note: This section shows long-term trends and year-over-year changes")
        
#         # Calculate year-over-year changes for unemployment rate
#         df_ur = df_laus[df_laus["series_id"] == LAUS_UR_MSA].copy()
#         df_ur = df_ur.sort_values("date").set_index("date")
#         df_ur["yoy_change"] = df_ur["value"].diff(12)  # 12-month change
        
#         fig3 = go.Figure()
#         fig3.add_trace(go.Scatter(x=df_ur.index, y=df_ur["value"], 
#                                  name="Unemployment Rate", line=dict(color='royalblue')))
#         fig3.update_layout(title="Unemployment Rate Trend",
#                           yaxis_title="Unemployment Rate (%)",
#                           hovermode='x unified')
#         st.plotly_chart(fig3, use_container_width=True)


# def display_prices_inflation_dashboard(bls_key: str, fred_key: str):
#     st.header("Prices & Inflation")
#     st.markdown("""
#     **Geographic Coverage:** Baltimore-Washington CPI Area (broader than MSA)  
#     **Coverage:** Baltimore-Columbia-Towson MSA + Washington-Arlington-Alexandria DC-VA-MD-WV MSA  
#     **Data Source:** BLS Consumer Price Index for All Urban Consumers (CPI-U)  
#     **Base Period:** 1982-84 = 100
#     """)
    
#     tab1, tab2 = st.tabs(["Overall CPI", "By Category"])
    
#     with tab1:
#         st.subheader("Consumer Price Index - All Items")
        
#         with st.spinner("Fetching CPI data..."):
#             cpi_json = bls_fetch_timeseries([CPIU_BALT_ALL], START_YEAR, END_YEAR,
#                                            registration_key=bls_key, catalog=True)
#             df_cpi = bls_to_dataframe(cpi_json)
        
#         if not df_cpi.empty:
#             # Calculate inflation rate (year-over-year percentage change)
#             df_cpi = df_cpi.sort_values("date").copy()
#             df_cpi["inflation_rate"] = df_cpi["value"].pct_change(12) * 100  # 12-month % change
            
#             # Display latest values
#             latest = df_cpi.iloc[-1]
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric("Current CPI Index", 
#                          f"{latest['value']:.1f}",
#                          delta=None)
#                 st.caption(f"As of {latest['date'].date()}")
#             with col2:
#                 st.metric("Annual Inflation Rate", 
#                          f"{latest['inflation_rate']:.1f}%",
#                          delta=None)
#                 st.caption("Year-over-Year Change")
            
#             # Dual-axis chart: CPI index and inflation rate
#             fig = go.Figure()
#             fig.add_trace(go.Scatter(x=df_cpi["date"], y=df_cpi["value"],
#                                     name="CPI Index", yaxis="y1",
#                                     line=dict(color='darkblue')))
#             fig.add_trace(go.Scatter(x=df_cpi["date"], y=df_cpi["inflation_rate"],
#                                     name="Inflation Rate (%)", yaxis="y2",
#                                     line=dict(color='red', dash='dot')))
#             fig.update_layout(
#                 title="CPI Index & Inflation Rate",
#                 yaxis=dict(title="CPI Index", side="left"),
#                 yaxis2=dict(title="Inflation Rate (%)", overlaying="y", side="right"),
#                 hovermode='x unified'
#             )
#             st.plotly_chart(fig, use_container_width=True)
            
#             st.download_button("Download CPI Data (CSV)",
#                               df_cpi.to_csv(index=False),
#                               "baltimore_cpi_all.csv",
#                               "text/csv")
    
#     with tab2:
#         st.subheader("CPI by Category")
#         st.markdown("*Track price changes in specific spending categories*")
        
#         cpi_categories = [CPIU_BALT_FOOD, CPIU_BALT_HOUSING, CPIU_BALT_TRANSP, 
#                          CPIU_BALT_MEDICAL, CPIU_BALT_ENERGY]
#         with st.spinner("Fetching CPI categories..."):
#             cpi_cat_json = bls_fetch_timeseries(cpi_categories, START_YEAR, END_YEAR,
#                                                registration_key=bls_key, catalog=True)
#             df_cpi_cat = bls_to_dataframe(cpi_cat_json)
        
#         cat_names = {
#             CPIU_BALT_FOOD: "Food",
#             CPIU_BALT_HOUSING: "Housing",
#             CPIU_BALT_TRANSP: "Transportation",
#             CPIU_BALT_MEDICAL: "Medical Care",
#             CPIU_BALT_ENERGY: "Energy",
#         }
#         df_cpi_cat["metric"] = df_cpi_cat["series_id"].map(cat_names).fillna(df_cpi_cat["series_id"])
        
#         selected_cats = st.multiselect(
#             "Select categories to compare:",
#             options=list(cat_names.values()),
#             default=["Food", "Housing", "Energy"]
#         )
        
#         plot_df = df_cpi_cat[df_cpi_cat["metric"].isin(selected_cats)]
#         if not plot_df.empty:
#             fig2 = px.line(plot_df, x="date", y="value", color="metric",
#                           title="CPI by Category",
#                           labels={"value": "Index Value", "date": "Date"})
#             fig2.update_layout(hovermode='x unified')
#             st.plotly_chart(fig2, use_container_width=True)
        
#         st.download_button("Download Category CPI (CSV)",
#                           df_cpi_cat.to_csv(index=False),
#                           "baltimore_cpi_categories.csv",
#                           "text/csv")

# def display_housing_dashboard(fred_key: str):
#     st.header("Housing & Real Estate")
#     st.markdown("""
#     **Data Sources:** Federal Housing Finance Agency (FHFA) via FRED, Census Bureau  
#     **Note:** Some metrics are at MSA level, others at national level where Baltimore-specific data unavailable
#     """)
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("Home Price Index (FHFA)")
#         st.markdown("*Baltimore-Columbia-Towson MSA | Quarterly*")
        
#         hpi_info = FRED_HOUSING_SERIES["Home Price Index (FHFA)"]
#         df_hpi = fred_series_observations(hpi_info["series_id"], fred_key, start=f"{START_YEAR}-01-01")
        
#         if not df_hpi.empty:
#             latest_hpi = df_hpi.iloc[-1]
#             # Calculate year-over-year change
#             if len(df_hpi) >= 5:  # Need at least 5 quarters
#                 yoy_change = ((latest_hpi["value"] / df_hpi.iloc[-5]["value"]) - 1) * 100
#                 st.metric("Current HPI", 
#                          f"{latest_hpi['value']:.2f}",
#                          delta=f"{yoy_change:+.1f}% YoY")
#             else:
#                 st.metric("Current HPI", f"{latest_hpi['value']:.2f}")
#             st.caption(f"Base: 1995 Q1 = 100 | As of {latest_hpi['date'].date()}")
            
#             fig_hpi = px.line(df_hpi, x="date", y="value",
#                              title=f"Home Price Index Trend ({START_YEAR}-{END_YEAR})",
#                              labels={"value": "Index (1995 Q1=100)", "date": "Date"})
#             fig_hpi.update_layout(hovermode='x')
#             st.plotly_chart(fig_hpi, use_container_width=True)
            
#             st.download_button("Download HPI Data (CSV)",
#                               df_hpi.to_csv(index=False),
#                               "baltimore_hpi.csv",
#                               "text/csv")
#         else:
#             st.warning("No HPI data available")
    
#     with col2:
#         st.subheader("Home Ownership Rate")
#         st.markdown("*United States (National) | Quarterly, Seasonally Adjusted*")
        
#         own_info = FRED_HOUSING_SERIES["Home Ownership Rate"]
#         df_own = fred_series_observations(own_info["series_id"], fred_key, start=f"{START_YEAR}-01-01")
        
#         if not df_own.empty:
#             latest_own = df_own.iloc[-1]
#             st.metric("Home Ownership Rate", 
#                      f"{latest_own['value']:.1f}%")
#             st.caption(f"As of {latest_own['date'].date()}")
            
#             fig_own = px.line(df_own, x="date", y="value",
#                              title="Home Ownership Rate Trend",
#                              labels={"value": "Ownership Rate (%)", "date": "Date"})
#             fig_own.update_layout(hovermode='x')
#             st.plotly_chart(fig_own, use_container_width=True)
            
#             st.download_button("Download Ownership Data (CSV)",
#                               df_own.to_csv(index=False),
#                               "us_homeownership.csv",
#                               "text/csv")
#         else:
#             st.warning("No ownership rate data available")


# def display_county_map_dashboard(fred_key: str):
#     st.header("Maryland County Unemployment Map")
#     st.markdown("""
#     **Data Source:** FRED Maps (GeoFRED) - County-level unemployment rates  
#     **Coverage:** All Maryland counties  
#     **Note:** Uses latest available cross-section from FRED Maps API
#     """)
    
#     with st.spinner("Building Maryland county unemployment choropleth..."):
#         cross, region_type, used_date, endpoint = fred_maps_series_cross_section(
#             EXAMPLE_FRED_COUNTY_UR, fred_key, date=None
#         )
        
#         data_status = (
#             f"**Data Info:** Source: {endpoint.replace('_', ' ').title()} | "
#             f"Region Type: {region_type or 'Unknown'} | "
#             f"Date: {used_date or 'Latest'} | "
#             f"Rows Retrieved: {len(cross)}"
#         )
#         st.info(data_status)
        
#         if cross.empty:
#             st.warning("No county cross-section returned by FRED Maps. The API may be temporarily unavailable.")
#             return
        
#         cross = _normalize_code_column(cross)
#         if "code" not in cross.columns:
#             st.error("FRED Maps did not return a region code column. Cannot render map.")
#             with st.expander("Debug: Show returned columns"):
#                 st.write("Columns received:", list(cross.columns))
#                 st.dataframe(cross.head())
#             return
        
#         cross["code"] = cross["code"].astype(str).str.zfill(5)
#         md = cross[cross["code"].str[:2] == "24"].copy()
        
#         if md.empty:
#             st.warning("No Maryland county rows found in the cross-section.")
#             return
        
#         st.success(f"Loaded {len(md)} Maryland counties")
        
#         # Fetch county boundaries GeoJSON
#         county_geo = fred_maps_shapes("county", fred_key)
#         featureidkey = _best_featureidkey(county_geo, md["code"])
        
#         # Create choropleth map
#         fig_map = px.choropleth(
#             md,
#             geojson=county_geo,
#             locations="code",
#             featureidkey=featureidkey,
#             color="value",
#             color_continuous_scale="Reds",
#             hover_name="region",
#             title=f"Maryland County Unemployment Rate — {used_date or 'Latest Available'}",
#             labels={"value": "Unemployment Rate (%)"}
#         )
#         fig_map.update_geos(fitbounds="locations", visible=False)
#         fig_map.update_layout(height=600)
#         st.plotly_chart(fig_map, use_container_width=True)
        
#         with st.expander("View County Data Table"):
#             display_df = md.sort_values("value", ascending=False)[["region", "code", "value"]].copy()
#             display_df.columns = ["County", "FIPS Code", "Unemployment Rate (%)"]
#             st.dataframe(display_df, use_container_width=True)
        
#         st.download_button("Download County Data (CSV)",
#                           md.to_csv(index=False),
#                           "maryland_county_unemployment.csv",
#                           "text/csv")


# class BaltimoreEconomicChatbot:
    
#     def __init__(self, bls_key: str, fred_key: str):
#         self.bls_key = bls_key
#         self.fred_key = fred_key
#         self.tools = self._create_tools()
#         self.agents: Dict[str, Agent] = {}
        
#         if CREWAI_AVAILABLE:
#             openai_key = os.getenv("OPENAI_API_KEY", "").strip()
#             if openai_key:
#                 os.environ["OPENAI_API_KEY"] = openai_key
#                 os.environ["CHROMA_OPENAI_API_KEY"] = openai_key
#                 try:
#                     print("[info] Creating CrewAI agents...")
#                     self.agents = self._create_agents()
#                     print(f"[info] Successfully created {len(self.agents)} agents: {list(self.agents.keys())}")
#                 except Exception as e:
#                     import traceback
#                     print(f"[warn] Failed to create agents: {e}")
#                     print(f"[warn] Full traceback:\n{traceback.format_exc()}")
#                     self.agents = {}
#             else:
#                 print("[warn] OPENAI_API_KEY not found in environment")
    
#     def _create_tools(self):
#         if not CREWAI_AVAILABLE or tool is None:
#             return []
        
#         bls_key = self.bls_key
#         fred_key = self.fred_key
        
#         @tool("Get unemployment data")
#         def get_unemployment_data() -> str:
#             """Fetch latest unemployment statistics for Baltimore MSA from BLS LAUS."""
#             try:
#                 data = bls_fetch_timeseries([LAUS_UR_MSA, LAUS_UNEMP_MSA, LAUS_EMP_MSA, LAUS_LF_MSA], 
#                                            START_YEAR, END_YEAR, 
#                                            registration_key=bls_key, catalog=True)
#                 df = bls_to_dataframe(data)
#                 if df.empty:
#                     return "No unemployment data available."
                
#                 last = df.sort_values("date").groupby("series_id").tail(1)
#                 result_dict = {row["series_id"]: (row["date"].date().isoformat(), row["value"]) 
#                               for _, row in last.iterrows()}
                
#                 ur_date, ur_val = result_dict.get(LAUS_UR_MSA, ("n/a", None))
#                 unemp_val = result_dict.get(LAUS_UNEMP_MSA, (None, 0))[1]
#                 emp_val = result_dict.get(LAUS_EMP_MSA, (None, 0))[1]
#                 lf_val = result_dict.get(LAUS_LF_MSA, (None, 0))[1]
                
#                 return f"Baltimore MSA Labor Market (BLS LAUS, {ur_date}): Unemployment Rate: {ur_val:.1f}%, Unemployed: {int(unemp_val):,}, Employed: {int(emp_val):,}, Labor Force: {int(lf_val):,}"
#             except Exception as e:
#                 return f"Error fetching unemployment data: {e}"
        
#         @tool("Get inflation data")
#         def get_cpi_data() -> str:
#             """Fetch Consumer Price Index and inflation rate for Baltimore-Washington area from BLS."""
#             try:
#                 data = bls_fetch_timeseries([CPIU_BALT_ALL], START_YEAR, END_YEAR,
#                                            registration_key=bls_key, catalog=True)
#                 df = bls_to_dataframe(data)
#                 if df.empty:
#                     return "No CPI data available."
                
#                 df = df.sort_values("date")
#                 last = df.iloc[-1]
#                 yoy = df.set_index("date")["value"].pct_change(12).loc[last["date"]]
                
#                 return f"CPI-U Baltimore-Washington Area (BLS, {last['date'].date().isoformat()}): Index: {last['value']:.1f}, Annual Inflation Rate: {yoy*100:.1f}%"
#             except Exception as e:
#                 return f"Error fetching CPI data: {e}"
        
#         @tool("Get industry employment")
#         def get_industry_employment() -> str:
#             """Fetch employment by industry sector for Baltimore MSA from BLS CES."""
#             try:
#                 ces_series = [CES_TNF_MSA, CES_MANUF_MSA, CES_TRADE_MSA, CES_PROF_MSA,
#                              CES_HEALTH_MSA, CES_LEISURE_MSA, CES_GOVT_MSA]
#                 data = bls_fetch_timeseries(ces_series, START_YEAR, END_YEAR,
#                                            registration_key=bls_key, catalog=True)
#                 df = bls_to_dataframe(data)
#                 if df.empty:
#                     return "No industry employment data available."
                
#                 last = df.sort_values("date").groupby("series_id").tail(1)
#                 names = {
#                     CES_TNF_MSA: "Total Nonfarm",
#                     CES_MANUF_MSA: "Manufacturing",
#                     CES_TRADE_MSA: "Retail Trade",
#                     CES_PROF_MSA: "Professional Services",
#                     CES_HEALTH_MSA: "Healthcare",
#                     CES_LEISURE_MSA: "Leisure & Hospitality",
#                     CES_GOVT_MSA: "Government",
#                 }
                
#                 result = f"Baltimore MSA Employment by Industry (BLS CES, {last.iloc[0]['date'].date().isoformat()}): "
#                 parts = [f"{names.get(row['series_id'], row['series_id'])}: {int(row['value']):,}" for _, row in last.iterrows()]
#                 return result + ", ".join(parts)
#             except Exception as e:
#                 return f"Error fetching industry employment: {e}"
        
#         @tool("Get home price index")
#         def get_home_price_index() -> str:
#             """Fetch FHFA Home Price Index for Baltimore MSA from FRED."""
#             try:
#                 hpi_series = FRED_HOUSING_SERIES["Home Price Index (FHFA)"]["series_id"]
#                 df = fred_series_observations(hpi_series, fred_key, start=f"{START_YEAR}-01-01")
#                 if df.empty:
#                     return "No home price data available."
                
#                 df = df.sort_values("date")
#                 last = df.iloc[-1]
                
#                 if len(df) >= 5:
#                     yoy_change = ((last["value"] / df.iloc[-5]["value"]) - 1) * 100
#                     return f"Baltimore MSA Home Price Index (FHFA via FRED, {last['date'].date().isoformat()}): HPI: {last['value']:.2f}, Year-over-Year Change: {yoy_change:+.1f}%"
#                 else:
#                     return f"Baltimore MSA Home Price Index (FHFA via FRED, {last['date'].date().isoformat()}): HPI: {last['value']:.2f}"
#             except Exception as e:
#                 return f"Error fetching home price data: {e}"
        
#         @tool("Get county unemployment map")
#         def get_county_unemployment() -> str:
#             """Fetch unemployment rates for all Maryland counties from FRED Maps."""
#             try:
#                 cross, _, used_date, endpoint = fred_maps_series_cross_section(
#                     EXAMPLE_FRED_COUNTY_UR, fred_key, date=None
#                 )
#                 cross = _normalize_code_column(cross)
#                 if cross.empty or "code" not in cross.columns:
#                     return "Could not retrieve Maryland county unemployment data."
                
#                 md = cross[cross["code"].astype(str).str.zfill(5).str[:2] == "24"].copy()
#                 if md.empty:
#                     return "No Maryland counties found in FRED Maps data."
                
#                 md_sorted = md.sort_values("value", ascending=False)
#                 top5 = md_sorted.head(5)
#                 bottom5 = md_sorted.tail(5)
                
#                 result = f"Maryland County Unemployment (FRED Maps, {used_date or 'Latest'}): "
#                 result += "Highest: " + ", ".join([f"{row['region']}: {row['value']:.1f}%" for _, row in top5.iterrows()])
#                 result += " | Lowest: " + ", ".join([f"{row['region']}: {row['value']:.1f}%" for _, row in bottom5.iterrows()])
#                 return result
#             except Exception as e:
#                 return f"Error fetching county data: {e}"
        
#         return [get_unemployment_data, get_cpi_data, get_industry_employment, 
#                 get_home_price_index, get_county_unemployment]
    
#     def _create_agents(self) -> Dict[str, Agent]:
#         fetcher = Agent(
#             role="Economic Data Fetcher",
#             goal="Use the provided tools to gather the exact Baltimore economic metrics requested by the user.",
#             backstory="You specialize in querying BLS LAUS, BLS CES, CPI-U, FRED housing data, and county unemployment cross-sections for the Baltimore region.",
#             tools=self.tools,
#             verbose=False,
#             allow_delegation=False
#         )
        
#         analyst = Agent(
#             role="Economic Data Analyst",
#             goal="Inspect retrieved metrics, calculate trends, and highlight the most relevant economic context for the user.",
#             backstory="You interpret labor market, price, and housing indicators for Baltimore and summarize what the numbers mean.",
#             verbose=False,
#             allow_delegation=False
#         )
        
#         communicator = Agent(
#             role="Economic Data Communicator",
#             goal="Deliver a clear, user-friendly answer that cites metrics, dates, geographies, and data sources.",
#             backstory="You craft concise explanations so the user quickly understands Baltimore's economic picture.",
#             verbose=False,
#             allow_delegation=False
#         )
        
#         return {
#             "fetcher": fetcher,
#             "analyst": analyst,
#             "communicator": communicator
#         }
    
#     def _build_tasks(self, message: str) -> List[Task]:
#         fetch_task = Task(
#             description=(
#                 "User question: {question}\n"
#                 "Identify which economic datasets are needed. Call the available tools to fetch the actual numbers. "
#                 "Return a structured summary listing metric name, value, unit, geography, date, and data source. "
#                 "If data is unavailable, state that explicitly."
#             ).format(question=message),
#             expected_output="Structured bullet list or JSON with fetched metrics, each including value, unit, geography, date, and data source.",
#             agent=self.agents["fetcher"],
#             tools=self.tools
#         )
        
#         analyze_task = Task(
#             description=(
#                 "Review the fetcher's output and synthesize insights that answer the user's question: {question}. "
#                 "Highlight at most three notable trends or comparisons, referencing specific metrics and dates."
#             ).format(question=message),
#             expected_output="Numbered list of insights referencing the fetched metrics and explaining their significance.",
#             agent=self.agents["analyst"]
#         )
        
#         respond_task = Task(
#             description=(
#                 "Compose the final response for the user question: {question}. "
#                 "Use the prior task outputs, cite data sources (BLS or FRED) and years, and explain the significance in plain language. "
#                 "If any requested data was unavailable, mention it."
#             ).format(question=message),
#             expected_output="Well-structured response (2-4 short paragraphs or bullet points) summarizing the metrics, trends, and caveats.",
#             agent=self.agents["communicator"]
#         )
        
#         return [fetch_task, analyze_task, respond_task]
    
    
#     def chat(self, message: str, chat_history: Optional[List] = None) -> str:
#         if chat_history is None:
#             chat_history = []
        
#         if self.agents:
#             try:
#                 # Ensure environment variables are set before creating Crew
#                 openai_key = os.getenv("OPENAI_API_KEY", "").strip()
#                 if openai_key:
#                     os.environ["OPENAI_API_KEY"] = openai_key
#                     os.environ["CHROMA_OPENAI_API_KEY"] = openai_key
                
#                 crew = Crew(
#                     agents=list(self.agents.values()),
#                     tasks=self._build_tasks(message),
#                     process=Process.sequential,
#                     verbose=True,
#                     memory=True,
#                     embedder={
#                         "provider": "openai",
#                         "config": {
#                             "model": "text-embedding-3-small"
#                        }
#                     }
#                 )
#                 result = crew.kickoff()
                
#                 if result:
#                     if hasattr(result, 'raw'):
#                         return str(result.raw)
#                     elif hasattr(result, 'output'):
#                         return str(result.output)
#                     elif hasattr(result, 'result'):
#                         return str(result.result)
#                     else:
#                         return str(result)
                        
#             except Exception as e:
#                 import traceback
#                 error_details = traceback.format_exc()
#                 print(f"CrewAI error: {e}")
#                 print(f"Full traceback:\n{error_details}")
#                 return f"Agent error (falling back): {str(e)}\n\n{self._fallback_response(message)}"
        
#         return self._fallback_response(message)
    
#     def _fallback_response(self, message: str) -> str:
#         m = message.lower()
        
#         info_text = """Available queries:
# - Unemployment & Labor Market: "latest unemployment rate"
# - Inflation & Prices: "current inflation"
# - Employment by Industry: "industry employment"
# - Home Prices: "home price index"
# - County Data: "Maryland county unemployment"

# Note: CrewAI agent mode requires OpenAI API key."""
        
#         try:
#             if any(k in m for k in ["unemployment", "jobless", "labor"]):
#                 data = bls_fetch_timeseries([LAUS_UR_MSA], START_YEAR, END_YEAR, 
#                                            registration_key=self.bls_key, catalog=True)
#                 df = bls_to_dataframe(data)
#                 if not df.empty:
#                     last = df.iloc[-1]
#                     return f"Baltimore MSA Unemployment Rate: {last['value']:.1f}% (as of {last['date'].date()})"
#             elif any(k in m for k in ["cpi", "inflation"]):
#                 data = bls_fetch_timeseries([CPIU_BALT_ALL], START_YEAR, END_YEAR,
#                                            registration_key=self.bls_key, catalog=True)
#                 df = bls_to_dataframe(data)
#                 if not df.empty:
#                     df = df.sort_values("date")
#                     last = df.iloc[-1]
#                     yoy = df.set_index("date")["value"].pct_change(12).loc[last["date"]]
#                     return f"CPI-U Index: {last['value']:.1f}, Annual Inflation: {yoy*100:.1f}% (as of {last['date'].date()})"
#         except Exception as e:
#             return f"Error: {e}"
        
#         return info_text


# def display_chatbot_interface(bls_key: str, fred_key: str):
#     st.header("AI Economic Assistant")
#     st.markdown("Ask questions about Baltimore economic data in natural language.")
    
#     if "chatbot" not in st.session_state:
#         with st.spinner("Initializing..."):
#             st.session_state.chatbot = BaltimoreEconomicChatbot(bls_key, fred_key)
            
#             if st.session_state.chatbot.agents:
#                 st.success("Agent mode active - using CrewAI")
#             else:
#                 if not CREWAI_AVAILABLE:
#                     st.warning("Basic mode - CrewAI not installed")
#                     st.info("Install with: pip install crewai crewai-tools")
#                 else:
#                     openai_key = os.getenv("OPENAI_API_KEY", "").strip()
#                     if not openai_key:
#                         st.warning("Basic mode - OpenAI API key not found")
#                         st.info("Add OPENAI_API_KEY to .env file")
#                     else:
#                         st.warning("Basic mode - agent initialization failed")
    
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []
    
#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])
    
#     user_input = st.chat_input("Ask about Baltimore economic data...")
    
#     if user_input:
#         with st.chat_message("user"):
#             st.markdown(user_input)
#         st.session_state.chat_history.append({"role": "user", "content": user_input})
        
#         with st.chat_message("assistant"):
#             with st.spinner("Processing..."):
#                 try:
#                     response = st.session_state.chatbot.chat(user_input)
#                 except Exception as e:
#                     st.error(f"Error: {e}")
#                     response = "Error processing request."
                
#                 st.markdown(response)
        
#         st.session_state.chat_history.append({"role": "assistant", "content": response})
#         st.rerun()

# def main():
#     st.title(APP_TITLE)
#     st.markdown("""
#     Comprehensive economic intelligence for Baltimore-Columbia-Towson Metropolitan Statistical Area.
#     Powered by Bureau of Labor Statistics (BLS) and Federal Reserve Economic Data (FRED).
#     """)
    
#     fred_key, bls_key = require_keys_from_env()
    
#     with st.sidebar:
#         st.header("Navigation")
        
#         page = st.radio(
#             "Choose Dashboard:",
#             options=[
#                 "Home & Overview",
#                 "Labor Market",
#                 "Prices & Inflation",
#                 "Housing & Real Estate",
#                 "Geographic/County Data",
#                 "AI Assistant"
#             ],
#             index=0
#         )
        
#         st.markdown("---")
#         st.caption(f"Data Period: {START_YEAR}-{END_YEAR}")
#         st.caption(f"MSA: Baltimore-Columbia-Towson (12580)")
#         st.caption("CrewAI: " + ("Available" if CREWAI_AVAILABLE else "Not Installed"))
        
#         with st.expander("About This Dashboard"):
#             st.markdown("""
#             **Data Sources:**
#             - BLS: Official labor statistics
#             - FRED: Federal Reserve data
            
#             **Coverage:**
#             - Labor/Employment: Baltimore MSA
#             - CPI: Baltimore-Washington Area
#             - Housing: Baltimore MSA
#             - Maps: Maryland counties
            
#             **Features:**
#             - Real-time data fetching
#             - Interactive visualizations
#             - CSV exports
#             - AI chatbot with OpenAI key
#             """)
    
#     if page == "Home & Overview":
#         st.header("Welcome to Baltimore Economic Intelligence Dashboard")
        
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             st.info("""
#             **Labor Market**  
#             Unemployment rates, employment levels, and industry breakdowns from BLS LAUS and CES surveys.
#             """)
        
#         with col2:
#             st.info("""
#             **Prices & Inflation**  
#             Consumer Price Index tracking cost of living changes across categories.
#             """)
        
#         with col3:
#             st.info("""
#             **Housing & Real Estate**  
#             Home Price Index and ownership rates from FHFA and Census Bureau.
#             """)
        
#         st.markdown("---")
        
#         st.subheader("Quick Start Guide")
#         st.markdown("""
#         1. Use the sidebar to navigate between categories
#         2. Select metrics within each dashboard to customize your view
#         3. Download data as CSV for further analysis
#         4. Ask the AI Assistant questions in natural language
        
#         The AI Assistant can explain data sources, fetch live statistics, and answer questions about regional coverage.
#         """)
        
#         st.markdown("---")
        
#         st.subheader("Understanding BLS vs FRED")
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("""
#             **BLS (Bureau of Labor Statistics)**
#             - Official U.S. government agency
#             - Primary source for labor statistics
#             - Conducts surveys (CES, LAUS, CPI)
#             - "Source of truth" for employment data
#             - Used by policymakers and media
            
#             **When to use:** Official employment, unemployment, and inflation statistics
#             """)
        
#         with col2:
#             st.markdown("""
#             **FRED (Federal Reserve Economic Data)**
#             - Maintained by St. Louis Federal Reserve
#             - Aggregator/repository of economic data
#             - Republishes BLS data and adds Fed analyses
#             - Provides geographic mapping tools
#             - Includes housing, GDP, financial indicators
            
#             **When to use:** Broader context, housing data, geographic visualization
#             """)
        
#         st.info("FRED often republishes BLS data. For example, FRED's unemployment rate for Baltimore ultimately comes from BLS LAUS, but is served through FRED's API.")
    
#     elif page == "Labor Market":
#         display_labor_market_dashboard(bls_key, fred_key)
    
#     elif page == "Prices & Inflation":
#         display_prices_inflation_dashboard(bls_key, fred_key)
    
#     elif page == "Housing & Real Estate":
#         display_housing_dashboard(fred_key)
    
#     elif page == "Geographic/County Data":
#         display_county_map_dashboard(fred_key)
    
#     elif page == "AI Assistant":
#         display_chatbot_interface(bls_key, fred_key)
    
#     st.markdown("---")
#     st.caption("Built with Streamlit | Data from BLS & FRED APIs | AI powered by CrewAI & OpenAI")


# if __name__ == "__main__":
#     main()





from __future__ import annotations  # Enable postponed evaluation of annotations

import os
import json
import re
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

os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool
    CREWAI_AVAILABLE = True
    print("[info] CrewAI loaded successfully")
except ImportError as e:
    CREWAI_AVAILABLE = False
    Agent = None
    Task = None
    Crew = None
    tool = None
    print(f"[info] CrewAI import failed: {e}")
    print("[info] Install with: pip install crewai")
except Exception as e:
    CREWAI_AVAILABLE = False
    Agent = None
    Task = None
    Crew = None
    tool = None
    print(f"[error] CrewAI load error: {e}")

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
FRED_SERIES_OBS_URL  = "https://api.stlouisfed.org/fred/series/observations"
FRED_MAPS_SERIES_DATA= "https://api.stlouisfed.org/geofred/series/data"
FRED_MAPS_REGIONAL   = "https://api.stlouisfed.org/geofred/regional/data"
FRED_MAPS_GROUP      = "https://api.stlouisfed.org/geofred/series/group"
FRED_MAPS_SHAPES     = "https://api.stlouisfed.org/geofred/shapes/file"

# ================================================================================
# DATA SERIES DEFINITIONS - ORGANIZED BY CATEGORY
# ================================================================================

# ============================================================
# CATEGORY 1: LABOR MARKET - UNEMPLOYMENT & EMPLOYMENT
# ============================================================

LAUS_UR_MSA     = "LAUMT241258000000003"  # Unemployment rate (%) - Baltimore MSA
LAUS_UNEMP_MSA  = "LAUMT241258000000004"  # Number of unemployed persons - Baltimore MSA
LAUS_EMP_MSA    = "LAUMT241258000000005"  # Number of employed persons - Baltimore MSA
LAUS_LF_MSA     = "LAUMT241258000000006"  # Total labor force size - Baltimore MSA

CES_TNF_MSA     = "SMU24125800000000001"  # Total Nonfarm Employment - Baltimore MSA
CES_GOODS_MSA   = "SMU24125800000000002"  # Goods-Producing Employment - Baltimore MSA
CES_SERVICE_MSA = "SMU24125800060000001"  # Service-Providing Employment - Baltimore MSA
CES_MANUF_MSA   = "SMU24125803000000001"  # Manufacturing Employment - Baltimore MSA
CES_TRADE_MSA   = "SMU24125804200000001"  # Retail Trade Employment - Baltimore MSA
CES_PROF_MSA    = "SMU24125806054000001"  # Professional & Business Services - Baltimore MSA
CES_HEALTH_MSA  = "SMU24125806562000001"  # Healthcare Employment - Baltimore MSA
CES_LEISURE_MSA = "SMU24125807000000001"  # Leisure & Hospitality Employment - Baltimore MSA
CES_GOVT_MSA    = "SMU24125809000000001"  # Government Employment - Baltimore MSA

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

CPIU_BALT_ALL   = "CUURS35ESA0"           # CPI-U All Items - Baltimore-Washington area
CPIU_BALT_FOOD  = "CUURS35ESAF"           # CPI-U Food - Baltimore-Washington area
CPIU_BALT_HOUSING = "CUURS35ESAH"         # CPI-U Housing - Baltimore-Washington area
CPIU_BALT_TRANSP = "CUURS35ESAT"          # CPI-U Transportation - Baltimore-Washington area
CPIU_BALT_MEDICAL = "CUURS35ESAM"         # CPI-U Medical Care - Baltimore-Washington area
CPIU_BALT_ENERGY = "CUURS35ESAE"          # CPI-U Energy - Baltimore-Washington area

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

FRED_HOUSING_SERIES = {
    "Home Price Index (FHFA)": {
        "series_id": "ATNHPIUS12580Q",
        "region": "Baltimore-Columbia-Towson, MD MSA",
        "frequency": "Quarterly, NSA",
        "source": "Federal Housing Finance Agency",
        "description": "House price index tracking home value changes (1995 Q1 = 100)"
    },
    "Median Home Sale Price": {
        "series_id": "MSPUS",
        "region": "United States",
        "frequency": "Quarterly",
        "source": "Census Bureau / HUD (via FRED)",
        "description": "Median sales price of houses sold"
    },
    "Home Ownership Rate": {
        "series_id": "RHORUSQ156N",
        "region": "United States",
        "frequency": "Quarterly, SA",
        "source": "Census Bureau (via FRED)",
        "description": "Percentage of housing units owner-occupied"
    },
}

# ============================================================
# CATEGORY 4: ECONOMIC INDICATORS
# ============================================================

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

EXAMPLE_FRED_COUNTY_UR = "MDBALT5URN"  # Baltimore City unemployment rate

# ================================================================================
# BLS helpers
# ================================================================================

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

def parse_bls_value(val: Any) -> float:
    """
    Robustly parse BLS 'value' field.

    BLS can return a dash '-' to represent missing/unavailable data in the API. :contentReference[oaicite:1]{index=1}
    We convert any non-numeric / missing markers to np.nan rather than raising.
    """
    if val is None:
        return float("nan")

    # Already numeric?
    if isinstance(val, (int, float, np.integer, np.floating)):
        try:
            return float(val)
        except Exception:
            return float("nan")

    # Strings: handle commas, missing markers, and stray footnote markers
    if isinstance(val, str):
        s = val.strip()

        # common missing markers
        if s in {"", "-", "—", ".", "NA", "N/A", "nan", "NaN", "None"}:
            return float("nan")

        # Remove thousands separators
        s = s.replace(",", "")

        # Sometimes values may include non-numeric annotations; keep digits, sign, decimal, exponent
        # Examples handled: "123.4", "-0.2", "1.2E3", "123(P)" -> "123"
        cleaned = re.sub(r"[^0-9eE\.\-\+]", "", s)

        # avoid edge cases like "" or just "+" or "-"
        if cleaned in {"", "+", "-", ".", "+.", "-."}:
            return float("nan")

        try:
            return float(cleaned)
        except Exception:
            return float("nan")

    # Fallback: try float conversion
    try:
        return float(val)
    except Exception:
        return float("nan")

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
                "value": parse_bls_value(val),
                "date": dt,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("date")

# ================================================================================
# FRED helpers
# ================================================================================

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
        if target_name in df.columns:
            return
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
    cand = [c for c in df.columns if c.lower().endswith("code")]
    if cand:
        return df.rename(columns={cand[0]: "code"})
    return df

def _freq_to_code(freq_str: Optional[str]) -> Optional[str]:
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
        return df_sd, region_type, used_date, "series_data"

    meta2 = _fred_maps_group_meta(series_id, api_key)
    region_type = meta2.get("region_type") or region_type
    series_group = meta2.get("series_group")
    season      = meta2.get("season")
    units       = meta2.get("units")
    freq_code   = _freq_to_code(meta2.get("frequency"))
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
# DASHBOARD SECTIONS
# ================================================================================

def _fmt_metric_value(value: float, is_rate: bool) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if is_rate:
        return f"{value:.1f}%"
    # many BLS series are whole counts; safely cast to int
    return f"{int(round(float(value))):,}"

def display_labor_market_dashboard(bls_key: str, fred_key: str):
    st.header("Labor Market Indicators")
    st.markdown("""
    **Geographic Coverage:** Baltimore-Columbia-Towson, MD Metropolitan Statistical Area (MSA)  
    **MSA Code:** 12580 | **Counties:** Baltimore City + Baltimore, Carroll, Anne Arundel, Howard, Harford, Queen Anne's  
    **Data Sources:** BLS LAUS (household survey) + BLS CES (establishment survey) + FRED
    """)

    tab1, tab2, tab3 = st.tabs(["Overview", "By Industry", "Trends"])

    with tab1:
        st.subheader("Labor Market Overview")

        laus_series = [LAUS_UR_MSA, LAUS_UNEMP_MSA, LAUS_EMP_MSA, LAUS_LF_MSA]
        with st.spinner("Fetching BLS LAUS data..."):
            bls_json = bls_fetch_timeseries(laus_series, START_YEAR, END_YEAR,
                                           registration_key=bls_key, catalog=True)
            df_laus = bls_to_dataframe(bls_json)

        if df_laus.empty:
            st.warning("No LAUS data returned for the selected period.")
            return

        laus_names = {
            LAUS_UR_MSA: "Unemployment Rate (%) | BLS LAUS",
            LAUS_UNEMP_MSA: "Unemployed Persons | BLS LAUS",
            LAUS_EMP_MSA: "Employed Persons | BLS LAUS",
            LAUS_LF_MSA: "Labor Force | BLS LAUS",
        }
        df_laus["metric"] = df_laus["series_id"].map(laus_names).fillna(df_laus["series_id"])

        col1, col2, col3, col4 = st.columns(4)
        latest_data = df_laus.sort_values("date").groupby("series_id", as_index=False).tail(1)

        cols = [col1, col2, col3, col4]
        for idx, (_, row) in enumerate(latest_data.iterrows()):
            if idx >= len(cols):
                break
            metric_name = laus_names.get(row["series_id"], row["series_id"])
            is_rate = "Rate" in metric_name
            with cols[idx]:
                st.metric(metric_name.split("|")[0].strip(), _fmt_metric_value(row["value"], is_rate), delta=None)
                st.caption(f"As of {row['date'].date()}")

        fig = px.line(df_laus, x="date", y="value", color="metric",
                     title=f"Labor Market Trends — {START_YEAR} to {END_YEAR}",
                     labels={"value": "Value", "date": "Date"})
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View Raw Data"):
            st.dataframe(df_laus)

        st.download_button("Download LAUS Data (CSV)",
                          df_laus.to_csv(index=False),
                          "baltimore_laus_data.csv",
                          "text/csv")

    with tab2:
        st.subheader("Employment by Industry")
        st.markdown("*Source: BLS Current Employment Statistics (CES) - Establishment Survey*")

        ces_series = [CES_TNF_MSA, CES_MANUF_MSA, CES_TRADE_MSA, CES_PROF_MSA,
                     CES_HEALTH_MSA, CES_LEISURE_MSA, CES_GOVT_MSA]
        with st.spinner("Fetching BLS CES industry data..."):
            ces_json = bls_fetch_timeseries(ces_series, START_YEAR, END_YEAR,
                                           registration_key=bls_key, catalog=True)
            df_ces = bls_to_dataframe(ces_json)

        if df_ces.empty:
            st.warning("No CES data returned for the selected period.")
            return

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

            latest_ind = plot_df.sort_values("date").groupby("metric", as_index=False).tail(1)[["metric", "date", "value"]]
            latest_ind["value"] = latest_ind["value"].apply(lambda x: "n/a" if pd.isna(x) else f"{int(round(float(x))):,}")
            latest_ind["date"] = latest_ind["date"].dt.date.astype(str)
            st.dataframe(latest_ind.set_index("metric"), use_container_width=True)

        st.download_button("Download Industry Data (CSV)",
                          df_ces.to_csv(index=False),
                          "baltimore_ces_industry.csv",
                          "text/csv")

    with tab3:
        st.subheader("Historical Trends & Analysis")
        st.info("Note: This section shows long-term trends and year-over-year changes")

        df_ur = df_laus[df_laus["series_id"] == LAUS_UR_MSA].copy()
        if df_ur.empty:
            st.warning("No unemployment rate series available for trend analysis.")
            return

        df_ur = df_ur.sort_values("date").set_index("date")
        df_ur["yoy_change"] = df_ur["value"].diff(12)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df_ur.index, y=df_ur["value"],
                                 name="Unemployment Rate"))
        fig3.update_layout(title="Unemployment Rate Trend",
                          yaxis_title="Unemployment Rate (%)",
                          hovermode='x unified')
        st.plotly_chart(fig3, use_container_width=True)

def display_prices_inflation_dashboard(bls_key: str, fred_key: str):
    st.header("Prices & Inflation")
    st.markdown("""
    **Geographic Coverage:** Baltimore-Washington CPI Area (broader than MSA)  
    **Coverage:** Baltimore-Columbia-Towson MSA + Washington-Arlington-Alexandria DC-VA-MD-WV MSA  
    **Data Source:** BLS Consumer Price Index for All Urban Consumers (CPI-U)  
    **Base Period:** 1982-84 = 100
    """)

    tab1, tab2 = st.tabs(["Overall CPI", "By Category"])

    with tab1:
        st.subheader("Consumer Price Index - All Items")

        with st.spinner("Fetching CPI data..."):
            cpi_json = bls_fetch_timeseries([CPIU_BALT_ALL], START_YEAR, END_YEAR,
                                           registration_key=bls_key, catalog=True)
            df_cpi = bls_to_dataframe(cpi_json)

        if df_cpi.empty:
            st.warning("No CPI data returned for the selected period.")
            return

        df_cpi = df_cpi.sort_values("date").copy()
        df_cpi["inflation_rate"] = df_cpi["value"].pct_change(12) * 100

        latest = df_cpi.iloc[-1]
        col1, col2 = st.columns(2)
        with col1:
            v = latest["value"]
            st.metric("Current CPI Index", "n/a" if pd.isna(v) else f"{v:.1f}", delta=None)
            st.caption(f"As of {latest['date'].date()}")
        with col2:
            ir = latest["inflation_rate"]
            st.metric("Annual Inflation Rate", "n/a" if pd.isna(ir) else f"{ir:.1f}%", delta=None)
            st.caption("Year-over-Year Change")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_cpi["date"], y=df_cpi["value"],
                                name="CPI Index", yaxis="y1"))
        fig.add_trace(go.Scatter(x=df_cpi["date"], y=df_cpi["inflation_rate"],
                                name="Inflation Rate (%)", yaxis="y2",
                                line=dict(dash='dot')))
        fig.update_layout(
            title="CPI Index & Inflation Rate",
            yaxis=dict(title="CPI Index", side="left"),
            yaxis2=dict(title="Inflation Rate (%)", overlaying="y", side="right"),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("Download CPI Data (CSV)",
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

        if df_cpi_cat.empty:
            st.warning("No CPI category data returned for the selected period.")
            return

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

        st.download_button("Download Category CPI (CSV)",
                          df_cpi_cat.to_csv(index=False),
                          "baltimore_cpi_categories.csv",
                          "text/csv")

def display_housing_dashboard(fred_key: str):
    st.header("Housing & Real Estate")
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
            if len(df_hpi) >= 5 and pd.notna(latest_hpi["value"]) and pd.notna(df_hpi.iloc[-5]["value"]):
                yoy_change = ((latest_hpi["value"] / df_hpi.iloc[-5]["value"]) - 1) * 100
                st.metric("Current HPI",
                         f"{latest_hpi['value']:.2f}",
                         delta=f"{yoy_change:+.1f}% YoY")
            else:
                st.metric("Current HPI", "n/a" if pd.isna(latest_hpi["value"]) else f"{latest_hpi['value']:.2f}")
            st.caption(f"Base: 1995 Q1 = 100 | As of {latest_hpi['date'].date()}")

            fig_hpi = px.line(df_hpi, x="date", y="value",
                             title=f"Home Price Index Trend ({START_YEAR}-{END_YEAR})",
                             labels={"value": "Index (1995 Q1=100)", "date": "Date"})
            fig_hpi.update_layout(hovermode='x')
            st.plotly_chart(fig_hpi, use_container_width=True)

            st.download_button("Download HPI Data (CSV)",
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
                     "n/a" if pd.isna(latest_own["value"]) else f"{latest_own['value']:.1f}%")
            st.caption(f"As of {latest_own['date'].date()}")

            fig_own = px.line(df_own, x="date", y="value",
                             title="Home Ownership Rate Trend",
                             labels={"value": "Ownership Rate (%)", "date": "Date"})
            fig_own.update_layout(hovermode='x')
            st.plotly_chart(fig_own, use_container_width=True)

            st.download_button("Download Ownership Data (CSV)",
                              df_own.to_csv(index=False),
                              "us_homeownership.csv",
                              "text/csv")
        else:
            st.warning("No ownership rate data available")

def display_county_map_dashboard(fred_key: str):
    st.header("Maryland County Unemployment Map")
    st.markdown("""
    **Data Source:** FRED Maps (GeoFRED) - County-level unemployment rates  
    **Coverage:** All Maryland counties  
    **Note:** Uses latest available cross-section from FRED Maps API
    """)

    with st.spinner("Building Maryland county unemployment choropleth..."):
        cross, region_type, used_date, endpoint = fred_maps_series_cross_section(
            EXAMPLE_FRED_COUNTY_UR, fred_key, date=None
        )

        data_status = (
            f"**Data Info:** Source: {endpoint.replace('_', ' ').title()} | "
            f"Region Type: {region_type or 'Unknown'} | "
            f"Date: {used_date or 'Latest'} | "
            f"Rows Retrieved: {len(cross)}"
        )
        st.info(data_status)

        if cross.empty:
            st.warning("No county cross-section returned by FRED Maps. The API may be temporarily unavailable.")
            return

        cross = _normalize_code_column(cross)
        if "code" not in cross.columns:
            st.error("FRED Maps did not return a region code column. Cannot render map.")
            with st.expander("Debug: Show returned columns"):
                st.write("Columns received:", list(cross.columns))
                st.dataframe(cross.head())
            return

        cross["code"] = cross["code"].astype(str).str.zfill(5)
        md = cross[cross["code"].str[:2] == "24"].copy()

        if md.empty:
            st.warning("No Maryland county rows found in the cross-section.")
            return

        st.success(f"Loaded {len(md)} Maryland counties")

        county_geo = fred_maps_shapes("county", fred_key)
        featureidkey = _best_featureidkey(county_geo, md["code"])

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

        with st.expander("View County Data Table"):
            display_df = md.sort_values("value", ascending=False)[["region", "code", "value"]].copy()
            display_df.columns = ["County", "FIPS Code", "Unemployment Rate (%)"]
            st.dataframe(display_df, use_container_width=True)

        st.download_button("Download County Data (CSV)",
                          md.to_csv(index=False),
                          "maryland_county_unemployment.csv",
                          "text/csv")

class BaltimoreEconomicChatbot:
    def __init__(self, bls_key: str, fred_key: str):
        self.bls_key = bls_key
        self.fred_key = fred_key
        self.tools = self._create_tools()
        self.agents: Dict[str, Agent] = {}

        if CREWAI_AVAILABLE:
            openai_key = os.getenv("OPENAI_API_KEY", "").strip()
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
                os.environ["CHROMA_OPENAI_API_KEY"] = openai_key
                try:
                    print("[info] Creating CrewAI agents...")
                    self.agents = self._create_agents()
                    print(f"[info] Successfully created {len(self.agents)} agents: {list(self.agents.keys())}")
                except Exception as e:
                    import traceback
                    print(f"[warn] Failed to create agents: {e}")
                    print(f"[warn] Full traceback:\n{traceback.format_exc()}")
                    self.agents = {}
            else:
                print("[warn] OPENAI_API_KEY not found in environment")

    def _create_tools(self):
        if not CREWAI_AVAILABLE or tool is None:
            return []

        bls_key = self.bls_key
        fred_key = self.fred_key

        @tool("Get unemployment data")
        def get_unemployment_data() -> str:
            """Fetch latest unemployment statistics for Baltimore MSA from BLS LAUS."""
            try:
                data = bls_fetch_timeseries([LAUS_UR_MSA, LAUS_UNEMP_MSA, LAUS_EMP_MSA, LAUS_LF_MSA],
                                           START_YEAR, END_YEAR,
                                           registration_key=bls_key, catalog=True)
                df = bls_to_dataframe(data)
                if df.empty:
                    return "No unemployment data available."

                last = df.sort_values("date").groupby("series_id", as_index=False).tail(1)

                def _get_val(sid: str) -> Tuple[str, float]:
                    sub = last[last["series_id"] == sid]
                    if sub.empty:
                        return ("n/a", float("nan"))
                    r = sub.iloc[0]
                    return (r["date"].date().isoformat(), r["value"])

                ur_date, ur_val = _get_val(LAUS_UR_MSA)
                _, unemp_val = _get_val(LAUS_UNEMP_MSA)
                _, emp_val = _get_val(LAUS_EMP_MSA)
                _, lf_val = _get_val(LAUS_LF_MSA)

                ur_txt = "n/a" if pd.isna(ur_val) else f"{ur_val:.1f}%"
                unemp_txt = "n/a" if pd.isna(unemp_val) else f"{int(round(float(unemp_val))):,}"
                emp_txt = "n/a" if pd.isna(emp_val) else f"{int(round(float(emp_val))):,}"
                lf_txt = "n/a" if pd.isna(lf_val) else f"{int(round(float(lf_val))):,}"

                return f"Baltimore MSA Labor Market (BLS LAUS, {ur_date}): Unemployment Rate: {ur_txt}, Unemployed: {unemp_txt}, Employed: {emp_txt}, Labor Force: {lf_txt}"
            except Exception as e:
                return f"Error fetching unemployment data: {e}"

        @tool("Get inflation data")
        def get_cpi_data() -> str:
            """Fetch Consumer Price Index and inflation rate for Baltimore-Washington area from BLS."""
            try:
                data = bls_fetch_timeseries([CPIU_BALT_ALL], START_YEAR, END_YEAR,
                                           registration_key=bls_key, catalog=True)
                df = bls_to_dataframe(data)
                if df.empty:
                    return "No CPI data available."

                df = df.sort_values("date")
                last = df.iloc[-1]
                yoy = df.set_index("date")["value"].pct_change(12).loc[last["date"]]

                idx_txt = "n/a" if pd.isna(last["value"]) else f"{last['value']:.1f}"
                yoy_txt = "n/a" if pd.isna(yoy) else f"{yoy*100:.1f}%"

                return f"CPI-U Baltimore-Washington Area (BLS, {last['date'].date().isoformat()}): Index: {idx_txt}, Annual Inflation Rate: {yoy_txt}"
            except Exception as e:
                return f"Error fetching CPI data: {e}"

        @tool("Get industry employment")
        def get_industry_employment() -> str:
            """Fetch employment by industry sector for Baltimore MSA from BLS CES."""
            try:
                ces_series = [CES_TNF_MSA, CES_MANUF_MSA, CES_TRADE_MSA, CES_PROF_MSA,
                             CES_HEALTH_MSA, CES_LEISURE_MSA, CES_GOVT_MSA]
                data = bls_fetch_timeseries(ces_series, START_YEAR, END_YEAR,
                                           registration_key=bls_key, catalog=True)
                df = bls_to_dataframe(data)
                if df.empty:
                    return "No industry employment data available."

                last = df.sort_values("date").groupby("series_id", as_index=False).tail(1)
                names = {
                    CES_TNF_MSA: "Total Nonfarm",
                    CES_MANUF_MSA: "Manufacturing",
                    CES_TRADE_MSA: "Retail Trade",
                    CES_PROF_MSA: "Professional Services",
                    CES_HEALTH_MSA: "Healthcare",
                    CES_LEISURE_MSA: "Leisure & Hospitality",
                    CES_GOVT_MSA: "Government",
                }

                date_txt = last.iloc[0]["date"].date().isoformat() if not last.empty else "n/a"
                parts = []
                for _, r in last.iterrows():
                    v = r["value"]
                    v_txt = "n/a" if pd.isna(v) else f"{int(round(float(v))):,}"
                    parts.append(f"{names.get(r['series_id'], r['series_id'])}: {v_txt}")

                return f"Baltimore MSA Employment by Industry (BLS CES, {date_txt}): " + ", ".join(parts)
            except Exception as e:
                return f"Error fetching industry employment: {e}"

        @tool("Get home price index")
        def get_home_price_index() -> str:
            """Fetch FHFA Home Price Index for Baltimore MSA from FRED."""
            try:
                hpi_series = FRED_HOUSING_SERIES["Home Price Index (FHFA)"]["series_id"]
                df = fred_series_observations(hpi_series, fred_key, start=f"{START_YEAR}-01-01")
                if df.empty:
                    return "No home price data available."

                df = df.sort_values("date")
                last = df.iloc[-1]

                if len(df) >= 5 and pd.notna(last["value"]) and pd.notna(df.iloc[-5]["value"]):
                    yoy_change = ((last["value"] / df.iloc[-5]["value"]) - 1) * 100
                    return f"Baltimore MSA Home Price Index (FHFA via FRED, {last['date'].date().isoformat()}): HPI: {last['value']:.2f}, Year-over-Year Change: {yoy_change:+.1f}%"
                else:
                    hpi_txt = "n/a" if pd.isna(last["value"]) else f"{last['value']:.2f}"
                    return f"Baltimore MSA Home Price Index (FHFA via FRED, {last['date'].date().isoformat()}): HPI: {hpi_txt}"
            except Exception as e:
                return f"Error fetching home price data: {e}"

        @tool("Get county unemployment map")
        def get_county_unemployment() -> str:
            """Fetch unemployment rates for all Maryland counties from FRED Maps."""
            try:
                cross, _, used_date, endpoint = fred_maps_series_cross_section(
                    EXAMPLE_FRED_COUNTY_UR, fred_key, date=None
                )
                cross = _normalize_code_column(cross)
                if cross.empty or "code" not in cross.columns:
                    return "Could not retrieve Maryland county unemployment data."

                md = cross[cross["code"].astype(str).str.zfill(5).str[:2] == "24"].copy()
                if md.empty:
                    return "No Maryland counties found in FRED Maps data."

                md_sorted = md.sort_values("value", ascending=False)
                top5 = md_sorted.head(5)
                bottom5 = md_sorted.tail(5)

                result = f"Maryland County Unemployment (FRED Maps, {used_date or 'Latest'}): "
                result += "Highest: " + ", ".join([f"{row['region']}: {row['value']:.1f}%" for _, row in top5.iterrows()])
                result += " | Lowest: " + ", ".join([f"{row['region']}: {row['value']:.1f}%" for _, row in bottom5.iterrows()])
                return result
            except Exception as e:
                return f"Error fetching county data: {e}"

        return [get_unemployment_data, get_cpi_data, get_industry_employment,
                get_home_price_index, get_county_unemployment]

    def _create_agents(self) -> Dict[str, Agent]:
        fetcher = Agent(
            role="Economic Data Fetcher",
            goal="Use the provided tools to gather the exact Baltimore economic metrics requested by the user.",
            backstory="You specialize in querying BLS LAUS, BLS CES, CPI-U, FRED housing data, and county unemployment cross-sections for the Baltimore region.",
            tools=self.tools,
            verbose=False,
            allow_delegation=False
        )

        analyst = Agent(
            role="Economic Data Analyst",
            goal="Inspect retrieved metrics, calculate trends, and highlight the most relevant economic context for the user.",
            backstory="You interpret labor market, price, and housing indicators for Baltimore and summarize what the numbers mean.",
            verbose=False,
            allow_delegation=False
        )

        communicator = Agent(
            role="Economic Data Communicator",
            goal="Deliver a clear, user-friendly answer that cites metrics, dates, geographies, and data sources.",
            backstory="You craft concise explanations so the user quickly understands Baltimore's economic picture.",
            verbose=False,
            allow_delegation=False
        )

        return {
            "fetcher": fetcher,
            "analyst": analyst,
            "communicator": communicator
        }

    def _build_tasks(self, message: str) -> List[Task]:
        fetch_task = Task(
            description=(
                "User question: {question}\n"
                "Identify which economic datasets are needed. Call the available tools to fetch the actual numbers. "
                "Return a structured summary listing metric name, value, unit, geography, date, and data source. "
                "If data is unavailable, state that explicitly."
            ).format(question=message),
            expected_output="Structured bullet list or JSON with fetched metrics, each including value, unit, geography, date, and data source.",
            agent=self.agents["fetcher"],
            tools=self.tools
        )

        analyze_task = Task(
            description=(
                "Review the fetcher's output and synthesize insights that answer the user's question: {question}. "
                "Highlight at most three notable trends or comparisons, referencing specific metrics and dates."
            ).format(question=message),
            expected_output="Numbered list of insights referencing the fetched metrics and explaining their significance.",
            agent=self.agents["analyst"]
        )

        respond_task = Task(
            description=(
                "Compose the final response for the user question: {question}. "
                "Use the prior task outputs, cite data sources (BLS or FRED) and years, and explain the significance in plain language. "
                "If any requested data was unavailable, mention it."
            ).format(question=message),
            expected_output="Well-structured response (2-4 short paragraphs or bullet points) summarizing the metrics, trends, and caveats.",
            agent=self.agents["communicator"]
        )

        return [fetch_task, analyze_task, respond_task]

    def chat(self, message: str, chat_history: Optional[List] = None) -> str:
        if chat_history is None:
            chat_history = []

        if self.agents:
            try:
                openai_key = os.getenv("OPENAI_API_KEY", "").strip()
                if openai_key:
                    os.environ["OPENAI_API_KEY"] = openai_key
                    os.environ["CHROMA_OPENAI_API_KEY"] = openai_key

                crew = Crew(
                    agents=list(self.agents.values()),
                    tasks=self._build_tasks(message),
                    process=Process.sequential,
                    verbose=True,
                    memory=True,
                    embedder={
                        "provider": "openai",
                        "config": {
                            "model": "text-embedding-3-small"
                       }
                    }
                )
                result = crew.kickoff()

                if result:
                    if hasattr(result, 'raw'):
                        return str(result.raw)
                    elif hasattr(result, 'output'):
                        return str(result.output)
                    elif hasattr(result, 'result'):
                        return str(result.result)
                    else:
                        return str(result)

            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"CrewAI error: {e}")
                print(f"Full traceback:\n{error_details}")
                return f"Agent error (falling back): {str(e)}\n\n{self._fallback_response(message)}"

        return self._fallback_response(message)

    def _fallback_response(self, message: str) -> str:
        m = message.lower()

        info_text = """Available queries:
- Unemployment & Labor Market: "latest unemployment rate"
- Inflation & Prices: "current inflation"
- Employment by Industry: "industry employment"
- Home Prices: "home price index"
- County Data: "Maryland county unemployment"

Note: CrewAI agent mode requires OpenAI API key."""

        try:
            if any(k in m for k in ["unemployment", "jobless", "labor"]):
                data = bls_fetch_timeseries([LAUS_UR_MSA], START_YEAR, END_YEAR,
                                           registration_key=self.bls_key, catalog=True)
                df = bls_to_dataframe(data)
                if not df.empty:
                    last = df.iloc[-1]
                    v = last["value"]
                    v_txt = "n/a" if pd.isna(v) else f"{v:.1f}%"
                    return f"Baltimore MSA Unemployment Rate: {v_txt} (as of {last['date'].date()})"
            elif any(k in m for k in ["cpi", "inflation"]):
                data = bls_fetch_timeseries([CPIU_BALT_ALL], START_YEAR, END_YEAR,
                                           registration_key=self.bls_key, catalog=True)
                df = bls_to_dataframe(data)
                if not df.empty:
                    df = df.sort_values("date")
                    last = df.iloc[-1]
                    yoy = df.set_index("date")["value"].pct_change(12).loc[last["date"]]
                    idx_txt = "n/a" if pd.isna(last["value"]) else f"{last['value']:.1f}"
                    yoy_txt = "n/a" if pd.isna(yoy) else f"{yoy*100:.1f}%"
                    return f"CPI-U Index: {idx_txt}, Annual Inflation: {yoy_txt} (as of {last['date'].date()})"
        except Exception as e:
            return f"Error: {e}"

        return info_text

def display_chatbot_interface(bls_key: str, fred_key: str):
    st.header("AI Economic Assistant")
    st.markdown("Ask questions about Baltimore economic data in natural language.")

    if "chatbot" not in st.session_state:
        with st.spinner("Initializing..."):
            st.session_state.chatbot = BaltimoreEconomicChatbot(bls_key, fred_key)

            if st.session_state.chatbot.agents:
                st.success("Agent mode active - using CrewAI")
            else:
                if not CREWAI_AVAILABLE:
                    st.warning("Basic mode - CrewAI not installed")
                    st.info("Install with: pip install crewai crewai-tools")
                else:
                    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
                    if not openai_key:
                        st.warning("Basic mode - OpenAI API key not found")
                        st.info("Add OPENAI_API_KEY to .env file")
                    else:
                        st.warning("Basic mode - agent initialization failed")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about Baltimore economic data...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    response = st.session_state.chatbot.chat(user_input)
                except Exception as e:
                    st.error(f"Error: {e}")
                    response = "Error processing request."

                st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

def main():
    st.title(APP_TITLE)
    st.markdown("""
    Comprehensive economic intelligence for Baltimore-Columbia-Towson Metropolitan Statistical Area.
    Powered by Bureau of Labor Statistics (BLS) and Federal Reserve Economic Data (FRED).
    """)

    fred_key, bls_key = require_keys_from_env()

    with st.sidebar:
        st.header("Navigation")

        page = st.radio(
            "Choose Dashboard:",
            options=[
                "Home & Overview",
                "Labor Market",
                "Prices & Inflation",
                "Housing & Real Estate",
                "Geographic/County Data",
                "AI Assistant"
            ],
            index=0
        )

        st.markdown("---")
        st.caption(f"Data Period: {START_YEAR}-{END_YEAR}")
        st.caption(f"MSA: Baltimore-Columbia-Towson (12580)")
        st.caption("CrewAI: " + ("Available" if CREWAI_AVAILABLE else "Not Installed"))

        with st.expander("About This Dashboard"):
            st.markdown("""
            **Data Sources:**
            - BLS: Official labor statistics
            - FRED: Federal Reserve data

            **Coverage:**
            - Labor/Employment: Baltimore MSA
            - CPI: Baltimore-Washington Area
            - Housing: Baltimore MSA
            - Maps: Maryland counties

            **Features:**
            - Real-time data fetching
            - Interactive visualizations
            - CSV exports
            - AI chatbot with OpenAI key
            """)

    if page == "Home & Overview":
        st.header("Welcome to Baltimore Economic Intelligence Dashboard")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("""
            **Labor Market**  
            Unemployment rates, employment levels, and industry breakdowns from BLS LAUS and CES surveys.
            """)

        with col2:
            st.info("""
            **Prices & Inflation**  
            Consumer Price Index tracking cost of living changes across categories.
            """)

        with col3:
            st.info("""
            **Housing & Real Estate**  
            Home Price Index and ownership rates from FHFA and Census Bureau.
            """)

        st.markdown("---")

        st.subheader("Quick Start Guide")
        st.markdown("""
        1. Use the sidebar to navigate between categories
        2. Select metrics within each dashboard to customize your view
        3. Download data as CSV for further analysis
        4. Ask the AI Assistant questions in natural language

        The AI Assistant can explain data sources, fetch live statistics, and answer questions about regional coverage.
        """)

        st.markdown("---")

        st.subheader("Understanding BLS vs FRED")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **BLS (Bureau of Labor Statistics)**
            - Official U.S. government agency
            - Primary source for labor statistics
            - Conducts surveys (CES, LAUS, CPI)
            - "Source of truth" for employment data
            - Used by policymakers and media

            **When to use:** Official employment, unemployment, and inflation statistics
            """)

        with col2:
            st.markdown("""
            **FRED (Federal Reserve Economic Data)**
            - Maintained by St. Louis Federal Reserve
            - Aggregator/repository of economic data
            - Republishes BLS data and adds Fed analyses
            - Provides geographic mapping tools
            - Includes housing, GDP, financial indicators

            **When to use:** Broader context, housing data, geographic visualization
            """)

        st.info("FRED often republishes BLS data. For example, FRED's unemployment rate for Baltimore ultimately comes from BLS LAUS, but is served through FRED's API.")

    elif page == "Labor Market":
        display_labor_market_dashboard(bls_key, fred_key)

    elif page == "Prices & Inflation":
        display_prices_inflation_dashboard(bls_key, fred_key)

    elif page == "Housing & Real Estate":
        display_housing_dashboard(fred_key)

    elif page == "Geographic/County Data":
        display_county_map_dashboard(fred_key)

    elif page == "AI Assistant":
        display_chatbot_interface(bls_key, fred_key)

    st.markdown("---")
    st.caption("Built with Streamlit | Data from BLS & FRED APIs | AI powered by CrewAI & OpenAI")

if __name__ == "__main__":
    main()
