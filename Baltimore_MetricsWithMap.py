# """
# dashboard_multi_year.py — Baltimore CHD-style dashboard (multi-year, robust Gazetteer fallback)

# What’s new vs previous:
# - If shapefile polygons aren't available (no GeoPandas), the code first tries to compute
#   centroids FROM THE SHAPEFILE (no extra libs). If that fails, it falls back to the official
#   Census Gazetteer tracts file and now:
#     * Auto-detects delimiter (tab/comma/whitespace)
#     * Accepts INTPTLONG / INTPTLON / INTPTLNG (case-insensitive)
#     * Tries 2025 → 2024 → 2023 single-state Gazetteer files and logs the winner

# Metrics (ACS 5-year SUBJECT tables) aligned with City Health Dashboard definitions:
#   • S2801_C02_017E — Broadband connection (%)  [“Broadband such as cable, fiber optic or DSL”]
#   • S1701_C03_002E — Children in poverty (%)   [Percent under 18 below poverty level]
#   • S1501_C02_014E — High school completion, 25+ (%)

# Census confirms variable labels:
#   S2801_C02_017E (Broadband such as cable/fiber/DSL): https://api.census.gov/data/2022/acs/acs5/subject/groups/S2801.html
#   S1701_C03_002E (Under 18 below poverty):            https://api.census.gov/data/2018/acs/acs5/subject/groups/S1701.html
#   S1501_C02_014E (25+ HS grad or higher):             https://api.census.gov/data/2019/acs/acs5/subject/groups/S1501.html

# Gazetteer record layouts (columns include INTPTLAT/INTPTLONG):
#   2023 layout: https://www.census.gov/.../gaz23-record-layouts.html
#   2025 layout: https://www.census.gov/.../gaz-record-layouts.html
# """

# from __future__ import annotations
# import os, io, json, zipfile, struct, requests, pandas as pd
# import plotly.graph_objects as go

# # ---------------- CONFIG ----------------
# STATE_FIPS  = "24"   # Maryland
# COUNTY_FIPS = "510"  # Baltimore city
# YEARS       = [2018, 2020, 2022, 2023]   # edit as needed

# VARS = {
#     "broadband_pct":     ("S2801_C02_017E", "Broadband connection (%)"),
#     "child_poverty_pct": ("S1701_C03_002E", "Children in poverty (%)"),
#     "hs_complete_pct":   ("S1501_C02_014E", "High school completion, 25+ (%)"),
# }

# HERE     = os.path.dirname(os.path.abspath(__file__))
# DATA_RAW = os.path.join(HERE, "data", "raw")
# DATA_GEO = os.path.join(HERE, "data", "geo")
# OUT_HTML = os.path.join(HERE, "output", "dashboard_multi_year.html")
# os.makedirs(DATA_RAW, exist_ok=True)
# os.makedirs(DATA_GEO, exist_ok=True)
# os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)

# # MD tract shapefile ZIP (cartographic boundary; polygons if GeoPandas installed)
# SHAPEFILE_ZIP_URL  = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_24_tract_500k.zip"
# SHAPEFILE_ZIP_PATH = os.path.join(DATA_GEO, "cb_2022_24_tract_500k.zip")

# # Gazetteer single-state MD tracts (try newest first)
# GAZETTEER_URLS = [
#     "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2025_Gazetteer/2025_gaz_tracts_24.txt",
#     "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_gaz_tracts_24.txt",
#     "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_gaz_tracts_24.txt",
# ]
# GAZETTEER_TXT_PATH = os.path.join(DATA_GEO, "gaz_tracts_24.txt")

# MAPBOX_STYLE = "open-street-map"
# CITY_CENTER  = {"lat": 39.2992, "lon": -76.6094}
# DEFAULT_ZOOM = 10


# # ---------------- basic download helper ----------------
# def _download(url: str, dest_path: str) -> bool:
#     try:
#         print(f"Downloading: {url}")
#         r = requests.get(url, timeout=120)
#         r.raise_for_status()
#         with open(dest_path, "wb") as f:
#             f.write(r.content)
#         print(f"Saved -> {dest_path}")
#         return True
#     except Exception as e:
#         print(f"[warn] download failed: {e}")
#         return False


# def ensure_geo_data() -> dict:
#     """
#     Ensure we have at least *one* usable geographic source:
#       - shapefile zip (for polygons OR shapefile-based centroids)
#       - gazetteer txt (for centroids)
#     """
#     have_shp = os.path.exists(SHAPEFILE_ZIP_PATH)
#     if not have_shp:
#         ok = _download(SHAPEFILE_ZIP_URL, SHAPEFILE_ZIP_PATH)
#         have_shp = ok and os.path.exists(SHAPEFILE_ZIP_PATH)

#     have_gaz = os.path.exists(GAZETTEER_TXT_PATH)
#     if not have_gaz:
#         for u in GAZETTEER_URLS:
#             if _download(u, GAZETTEER_TXT_PATH):
#                 have_gaz = True
#                 print(f"[info] Gazetteer source: {u}")
#                 break

#     if not have_shp and not have_gaz:
#         raise SystemExit(
#             "Could not obtain geographic data.\n"
#             "Tried: MD tract shapefile (2022) and MD Gazetteer (2025/2024/2023)."
#         )
#     return {"shapefile_zip": SHAPEFILE_ZIP_PATH if have_shp else None,
#             "gazetteer_txt": GAZETTEER_TXT_PATH if have_gaz else None}


# # ---------------- ACS JSON → tidy ----------------
# def load_metric_df(var_code: str, years: list[int]) -> pd.DataFrame:
#     frames = []
#     for y in years:
#         path = os.path.join(DATA_RAW, f"baltimore_{var_code}_{y}.json")
#         if not os.path.exists(path):
#             print(f"[skip] missing {path}")
#             continue
#         with open(path, "r") as f:
#             raw = json.load(f)
#         cols = raw[0]; rows = raw[1:]
#         df = pd.DataFrame(rows, columns=cols)
#         df[var_code] = pd.to_numeric(df[var_code], errors="coerce")
#         df["year"] = y
#         df["GEOID"] = df["state"].str.zfill(2) + df["county"].str.zfill(3) + df["tract"].str.zfill(6)
#         frames.append(df[["GEOID", var_code, "year"]])
#     return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["GEOID", var_code, "year"])

# def combine_metrics(years: list[int]) -> pd.DataFrame:
#     pieces = []
#     for _, (var_code, _) in VARS.items():
#         pieces.append(load_metric_df(var_code, years))
#     if not pieces:
#         return pd.DataFrame(columns=["GEOID","year"]+[v for v,_ in VARS.values()])
#     out = pieces[0]
#     for nxt in pieces[1:]:
#         out = out.merge(nxt, on=["GEOID","year"], how="outer")
#     # keep Baltimore City
#     out = out[out["GEOID"].str.startswith(STATE_FIPS + COUNTY_FIPS)]
#     return out


# # ---------------- Shapefile → polygons or centroids ----------------
# def try_geopandas_polygons(zip_path: str):
#     """If GeoPandas is available, return a GeoDataFrame [GEOID, geometry] for Baltimore tracts."""
#     try:
#         import geopandas as gpd
#         gdf = gpd.read_file(f"zip://{zip_path}")
#         if "GEOID" not in gdf.columns:
#             gdf["GEOID"] = (
#                 gdf["STATEFP"].astype(str).str.zfill(2)
#                 + gdf["COUNTYFP"].astype(str).str.zfill(3)
#                 + gdf["TRACTCE"].astype(str).str.zfill(6)
#             )
#         gdf = gdf[gdf["GEOID"].str.startswith(STATE_FIPS + COUNTY_FIPS)][["GEOID","geometry"]]
#         return gdf
#     except Exception as e:
#         print(f"[info] GeoPandas polygons not available: {e}")
#         return None


# def centroids_from_shapefile(zip_path: str) -> pd.DataFrame:
#     """
#     Dependency-free centroid extraction from SHP bounding boxes.
#     *Not* geometric centroids but fine for bubble maps.
#     """
#     with zipfile.ZipFile(zip_path, "r") as zf:
#         shp_name = next(n for n in zf.namelist() if n.lower().endswith(".shp"))
#         dbf_name = next(n for n in zf.namelist() if n.lower().endswith(".dbf"))
#         shp = zf.read(shp_name)
#         dbf = zf.read(dbf_name)

#     def read_dbf(dbf_bytes: bytes):
#         version, yr, mo, dy = struct.unpack("<BBBB", dbf_bytes[0:4])
#         num_records      = struct.unpack("<I", dbf_bytes[4:8])[0]
#         header_len       = struct.unpack("<H", dbf_bytes[8:10])[0]
#         record_len       = struct.unpack("<H", dbf_bytes[10:12])[0]
#         fields, pos = [], 32
#         while pos < header_len - 1:
#             fd = dbf_bytes[pos:pos+32]
#             if not fd or fd[0] == 0x0D: break
#             name  = fd[:11].split(b"\x00",1)[0].decode("ascii","ignore").strip()
#             ftype = chr(fd[11]); flen = fd[16]
#             fields.append((name, ftype, flen)); pos += 32
#         recs = []
#         base = header_len
#         for i in range(num_records):
#             rec = dbf_bytes[base + i*record_len : base + (i+1)*record_len]
#             if not rec or rec[0] == 0x2A:  # deleted
#                 continue
#             off, vals = 1, {}
#             for (name, _ft, flen) in fields:
#                 raw = rec[off:off+flen]
#                 vals[name] = raw.decode("latin1","ignore").strip()
#                 off += flen
#             recs.append(vals)
#         return recs

#     def read_shp_centroids(shp_bytes: bytes):
#         pos, cents = 100, []
#         while pos + 8 <= len(shp_bytes):
#             # Record header (big-endian)
#             cont_len_words = int.from_bytes(shp_bytes[pos+4:pos+8], "big", signed=True)
#             pos += 8
#             content = shp_bytes[pos:pos + cont_len_words*2]
#             pos += cont_len_words*2
#             if len(content) < 44:
#                 continue
#             stype = int.from_bytes(content[0:4], "little", signed=True)
#             if stype in (3,5):  # polyline/polygon
#                 xmin = struct.unpack("<d", content[4:12])[0]
#                 ymin = struct.unpack("<d", content[12:20])[0]
#                 xmax = struct.unpack("<d", content[20:28])[0]
#                 ymax = struct.unpack("<d", content[28:36])[0]
#                 lon  = (xmin+xmax)/2.0
#                 lat  = (ymin+ymax)/2.0
#                 cents.append((lon,lat))
#         return cents

#     recs  = read_dbf(dbf)
#     cents = read_shp_centroids(shp)
#     rows = []
#     for rec, (lon,lat) in zip(recs, cents):
#         geoid = rec.get("GEOID","").strip()
#         if not geoid:
#             geoid = (rec.get("STATEFP","").zfill(2)
#                      + rec.get("COUNTYFP","").zfill(3)
#                      + rec.get("TRACTCE","").zfill(6))
#         if geoid.startswith(STATE_FIPS + COUNTY_FIPS):
#             rows.append({"GEOID": geoid, "lon": float(lon), "lat": float(lat)})
#     out = pd.DataFrame(rows)
#     if out.empty:
#         raise RuntimeError("No centroid rows parsed from shapefile.")
#     return out


# # ---------------- Gazetteer centroids (robust) ----------------
# def _read_gazetteer_any_delim(path: str) -> pd.DataFrame:
#     """
#     Read Gazetteer with robust delimiter detection:
#       1) Try tab
#       2) Try comma
#       3) Try python engine sep=None (sniffer) or whitespace regex
#     """
#     # try tab
#     try:
#         return pd.read_csv(path, sep="\t", dtype=str)
#     except Exception:
#         pass
#     # try comma
#     try:
#         return pd.read_csv(path, sep=",", dtype=str)
#     except Exception:
#         pass
#     # try sniffer / whitespace
#     try:
#         return pd.read_csv(path, sep=None, engine="python", dtype=str)
#     except Exception:
#         return pd.read_csv(path, sep=r"\s+", engine="python", dtype=str)

# def centroids_from_gazetteer(txt_path: str) -> pd.DataFrame:
#     df = _read_gazetteer_any_delim(txt_path)

#     # Build case-insensitive name map
#     name_map = {c.lower(): c for c in df.columns}
#     # Column aliases for longitude just in case some variant appears
#     lon_candidates = ["intptlong", "intptlon", "intptlng"]
#     lat_candidates = ["intptlat"]

#     def pick(colnames):
#         for k in colnames:
#             if k in name_map:
#                 return name_map[k]
#         # try relaxed contains (handles accidental leading/trailing spaces)
#         for c in df.columns:
#             if any(k == c.strip().lower() for k in colnames):
#                 return c
#         raise KeyError(f"None of {colnames} found in Gazetteer columns: {list(df.columns)}")

#     geoid_col = name_map.get("geoid") or pick(["geoid"])
#     lon_col   = pick(lon_candidates)
#     lat_col   = pick(lat_candidates)

#     out = df[[geoid_col, lon_col, lat_col]].rename(columns={
#         geoid_col: "GEOID", lon_col: "lon", lat_col: "lat"
#     })
#     out["lon"] = pd.to_numeric(out["lon"].astype(str).str.strip(), errors="coerce")
#     out["lat"] = pd.to_numeric(out["lat"].astype(str).str.strip(), errors="coerce")
#     out = out[out["GEOID"].str.startswith(STATE_FIPS + COUNTY_FIPS)].dropna(subset=["lon","lat"])
#     if out.empty:
#         raise RuntimeError("No Baltimore City rows parsed from Gazetteer.")
#     return out


# # ---------------- Build interactive Plotly figure ----------------
# def build_figure(df_metrics: pd.DataFrame, geo):
#     """
#     Two dropdowns:
#       - METRIC (Broadband / Children in poverty / HS completion)
#       - YEAR (2018/2020/2022/2023 or whatever you downloaded)
#     If 'geo' has .geometry -> polygon choropleth; else -> centroid bubble map.
#     """
#     metric_order = [
#         ("broadband_pct",     VARS["broadband_pct"][1]),
#         ("child_poverty_pct", VARS["child_poverty_pct"][1]),
#         ("hs_complete_pct",   VARS["hs_complete_pct"][1]),
#     ]
#     color_scales = {"broadband_pct": "Viridis", "child_poverty_pct": "Plasma", "hs_complete_pct": "Cividis"}
#     years = sorted(df_metrics["year"].dropna().unique().tolist())

#     fig = go.Figure()
#     trace_index = []  # (metric_key, year, idx)

#     # --- polygons (choropleth) if GeoDataFrame
#     if hasattr(geo, "geometry"):
#         geojson = geo.set_index("GEOID").__geo_interface__
#         for mkey, mlabel in metric_order:
#             var_code = VARS[mkey][0]
#             for y in years:
#                 sub = df_metrics[df_metrics["year"] == y][["GEOID", var_code]].dropna()
#                 tr = go.Choroplethmapbox(
#                     geojson=geojson,
#                     locations=sub["GEOID"],
#                     z=sub[var_code],
#                     featureidkey="properties.GEOID",
#                     colorscale=color_scales[mkey],
#                     zmin=float(sub[var_code].min()) if not sub.empty else 0,
#                     zmax=float(sub[var_code].max()) if not sub.empty else 100,
#                     marker_line_width=0,
#                     visible=False,
#                     colorbar=dict(title=mlabel),
#                 )
#                 fig.add_trace(tr)
#                 trace_index.append((mkey, y, len(fig.data)-1))
#     else:
#         # --- centroids (bubble map)
#         pts = geo[["GEOID","lon","lat"]].copy()
#         for mkey, mlabel in metric_order:
#             var_code = VARS[mkey][0]
#             for y in years:
#                 sub = df_metrics[df_metrics["year"] == y][["GEOID", var_code]].dropna()
#                 merged = pts.merge(sub, on="GEOID", how="inner")
#                 tr = go.Scattermapbox(
#                     lon=merged["lon"], lat=merged["lat"], mode="markers",
#                     marker=dict(
#                         size=8,
#                         color=merged[var_code],
#                         colorscale=color_scales[mkey],
#                         cmin=float(merged[var_code].min()) if not merged.empty else 0,
#                         cmax=float(merged[var_code].max()) if not merged.empty else 100,
#                         showscale=True,
#                         colorbar=dict(title=mlabel),
#                     ),
#                     text=merged["GEOID"],
#                     hovertemplate="Tract: %{text}<br>" + f"{mlabel}: " + "%{marker.color:.1f}%<extra></extra>",
#                     visible=False,
#                     name=f"{mlabel} {y}",
#                 )
#                 fig.add_trace(tr)
#                 trace_index.append((mkey, y, len(fig.data)-1))

#     if not trace_index:
#         raise SystemExit("No traces built—check data/raw JSONs.")

#     # make first trace visible
#     m0, y0, idx0 = trace_index[0]
#     fig.data[idx0].visible = True

#     def vis_for(metric_sel, year_sel):
#         vis = [False] * len(fig.data)
#         for mkey, y, idx in trace_index:
#             if mkey == metric_sel and y == year_sel:
#                 vis[idx] = True
#         return vis

#     metric_buttons = []
#     for mkey, mlabel in metric_order:
#         metric_buttons.append(dict(
#             label=mlabel,
#             method="update",
#             args=[{"visible": vis_for(mkey, y0)}, {"title": f"{mlabel} — {y0}"}],
#         ))

#     year_buttons = []
#     for y in years:
#         year_buttons.append(dict(
#             label=str(y),
#             method="update",
#             args=[{"visible": vis_for(m0, y)}, {"title": f"{VARS[m0][1]} — {y}"}],
#         ))

#     fig.update_layout(
#         title=f"{VARS[m0][1]} — {y0}",
#         mapbox=dict(style=MAPBOX_STYLE, center=CITY_CENTER, zoom=DEFAULT_ZOOM),
#         margin=dict(l=0, r=0, t=50, b=0),
#         updatemenus=[
#             dict(buttons=metric_buttons, direction="down",
#                  x=0.01, y=0.99, xanchor="left", yanchor="top", bgcolor="white"),
#             dict(buttons=year_buttons, direction="down",
#                  x=0.35, y=0.99, xanchor="left", yanchor="top", bgcolor="white"),
#         ],
#         annotations=[
#             dict(text="Metric", x=0.00, y=1.05, xref="paper", yref="paper", showarrow=False),
#             dict(text="Year",   x=0.34, y=1.05, xref="paper", yref="paper", showarrow=False),
#         ],
#     )
#     return fig


# # ---------------- main ----------------
# def main():
#     geo_avail = ensure_geo_data()

#     # Load ACS subject-table metrics for the years you downloaded
#     df = combine_metrics(YEARS)
#     if df.empty:
#         raise SystemExit("No metric data found under data/raw/. Confirm filenames: baltimore_{VAR}_{YEAR}.json")

#     # Prefer polygons if GeoPandas; else shapefile centroids; else Gazetteer
#     geo = None
#     if geo_avail["shapefile_zip"]:
#         gdf = try_geopandas_polygons(geo_avail["shapefile_zip"])
#         if gdf is not None:
#             geo = gdf
#         else:
#             try:
#                 print("[info] Using centroids from shapefile (no GeoPandas).")
#                 geo = centroids_from_shapefile(geo_avail["shapefile_zip"])
#             except Exception as e:
#                 print(f"[warn] shapefile-centroid fallback failed: {e}")

#     if geo is None:
#         print("[info] Using Gazetteer centroids.")
#         geo = centroids_from_gazetteer(geo_avail["gazetteer_txt"])

#     # Build + save interactive dashboard
#     fig = build_figure(df_metrics=df, geo=geo)
#     fig.write_html(OUT_HTML, include_plotlyjs="cdn", full_html=True)
#     print(f"\nWrote: {OUT_HTML}\nOpen this file in your browser.")

# if __name__ == "__main__":
#     main()




"""
Baltimore CHD-style dashboard (multi-year, robust geo handling)

Fixes in this version:
- Detects if cb_2022_24_tract_500k.zip is KML (wrong format for shapefile reading) and
  skips SHP parsing in that case, with a clear message.
- Gazetteer centroids reader now hard-codes sep="|" and validates expected columns.

Docs:
- Cartographic Boundary Files (formats & use): https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html
- 2022 SHP listing (contains cb_2022_24_tract_500k.zip): https://www2.census.gov/geo/tiger/GENZ2022/shp/
- Gazetteer files are pipe-delimited; have INTPTLAT/INTPTLONG: 
  https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html
  https://www.census.gov/programs-surveys/geography/technical-documentation/records-layout/gaz-record-layouts/gaz25-record-layouts.html
- ACS Subject variables:
  S2801_C02_017E (broadband such as cable/fiber/DSL): https://api.census.gov/data/2022/acs/acs5/subject/groups/S2801.html
  S1701_C03_002E (under 18 below poverty):            see S1701 group listing
  S1501_C02_014E (25+ high school grad or higher):    see S1501 group listing
"""

from __future__ import annotations
import os, io, json, zipfile, struct, requests, pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
import argparse
warnings.filterwarnings('ignore')

# Flask imports for integrated server functionality
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("[info] Flask not available - server mode will not work. Install with: pip install flask flask-cors")

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if present
    print("[info] Loaded environment variables from .env file")
except ImportError:
    print("[info] python-dotenv not available - will use system environment variables")

# Try to import AutoGen for chatbot functionality (multi-agent framework)
try:
    from autogen import ConversableAgent, UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager
    import autogen
    AUTOGEN_AVAILABLE = True
    print("[info] ✅ AutoGen loaded successfully - Multi-agent AI mode available")
except ImportError as e:
    AUTOGEN_AVAILABLE = False
    print(f"[info] ⚠️ AutoGen import failed: {e}")
    print("[info] Chatbot will use basic keyword matching mode")
    print("[info] Required package: pip install pyautogen")
except Exception as e:
    AUTOGEN_AVAILABLE = False
    print(f"[error] Unexpected error loading AutoGen: {e}")
    print("[info] Chatbot will use basic keyword matching mode")

# ---------------- CONFIG ----------------
STATE_FIPS  = "24"   # Maryland
COUNTY_FIPS = "510"  # Baltimore city
YEARS       = [2018, 2020, 2022, 2023]   # edit as needed

VARS = {
    "broadband_pct":     ("S2801_C02_017E", "Broadband connection (%)"),
    "child_poverty_pct": ("S1701_C03_002E", "Children in poverty (%)"),
    "hs_complete_pct":   ("S1501_C02_014E", "High school completion, 25+ (%)"),
}

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(HERE, "data", "raw")
DATA_GEO = os.path.join(HERE, "data", "geo")
OUT_HTML = os.path.join(HERE, "output", "dashboard_multi_year.html")
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_GEO, exist_ok=True)
os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)

# Expected SHP zip (NOT the KML zip)
SHP_ZIP_URL  = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_24_tract_500k.zip"
SHP_ZIP_PATH = os.path.join(DATA_GEO, "cb_2022_24_tract_500k.zip")

# Gazetteer (we’ll use the 2025 MD tracts file you already downloaded)
GAZ_TXT_PATH = os.path.join(DATA_GEO, "gaz_tracts_24.txt")

MAPBOX_STYLE = "open-street-map"
CITY_CENTER  = {"lat": 39.2992, "lon": -76.6094}
DEFAULT_ZOOM = 10

def _download(url: str, dest_path: str) -> bool:
    try:
        print(f"Downloading: {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
        print(f"Saved -> {dest_path}")
        return True
    except Exception as e:
        print(f"[warn] download failed: {e}")
        return False

def _zip_has_extensions(zip_path: str, exts: tuple[str,...]) -> bool:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [n.lower() for n in zf.namelist()]
        return any(name.endswith(ext) for name in names for ext in exts)
    except Exception:
        return False

def ensure_geo() -> dict:
    """
    Ensure we have *something* usable:
      - A real SHP zip (contains .shp + .dbf) — not KML.
      - Or a Gazetteer TXT.
    """
    shp_ok = False
    if os.path.exists(SHP_ZIP_PATH):
        # verify we didn’t accidentally save the KML zip under the same name
        shp_ok = _zip_has_extensions(SHP_ZIP_PATH, (".shp", ".dbf"))
        if not shp_ok:
            print("[info] Found cb_2022_24_tract_500k.zip but it does not contain .shp/.dbf "
                  "(likely a KML zip). Will skip SHP parsing.")
    else:
        print("[info] SHP zip missing; attempting to download the SHP (not KML)...")
        if _download(SHP_ZIP_URL, SHP_ZIP_PATH):
            shp_ok = _zip_has_extensions(SHP_ZIP_PATH, (".shp", ".dbf"))
            if not shp_ok:
                print("[warn] Downloaded zip does not contain .shp/.dbf — looks like the wrong file.")

    gaz_ok = os.path.exists(GAZ_TXT_PATH)
    if not gaz_ok:
        raise SystemExit("Gazetteer file not found at data/geo/gaz_tracts_24.txt. "
                         "Place the file there (pipe-delimited), or let me know and I’ll add an auto-download.")

    return {"shp_zip": SHP_ZIP_PATH if shp_ok else None, "gaz_txt": GAZ_TXT_PATH}

# ---------------- ACS JSON → tidy ----------------
def load_metric_df(var_code: str, years: list[int]) -> pd.DataFrame:
    frames = []
    for y in years:
        path = os.path.join(DATA_RAW, f"baltimore_{var_code}_{y}.json")
        if not os.path.exists(path):
            print(f"[skip] missing {path}")
            continue
        with open(path, "r") as f:
            raw = json.load(f)
        cols = raw[0]; rows = raw[1:]
        df = pd.DataFrame(rows, columns=cols)
        df[var_code] = pd.to_numeric(df[var_code], errors="coerce")
        df["year"] = y
        df["GEOID"] = df["state"].str.zfill(2) + df["county"].str.zfill(3) + df["tract"].str.zfill(6)
        frames.append(df[["GEOID", var_code, "year"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["GEOID", var_code, "year"])

def combine_metrics(years: list[int]) -> pd.DataFrame:
    pieces = []
    for _, (var_code, _) in VARS.items():
        pieces.append(load_metric_df(var_code, years))
    if not pieces:
        return pd.DataFrame(columns=["GEOID","year"]+[v for v,_ in VARS.values()])
    out = pieces[0]
    for nxt in pieces[1:]:
        out = out.merge(nxt, on=["GEOID","year"], how="outer")
    # Baltimore City filter
    out = out[out["GEOID"].str.startswith(STATE_FIPS + COUNTY_FIPS)]
    return out

# SHP centroids
def centroids_from_shapefile(zip_path: str) -> pd.DataFrame:
    """Derive bbox midpoints from SHP content (requires real .shp + .dbf inside the zip)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        shp_name = next(n for n in zf.namelist() if n.lower().endswith(".shp"))
        dbf_name = next(n for n in zf.namelist() if n.lower().endswith(".dbf"))
        shp = zf.read(shp_name)
        dbf = zf.read(dbf_name)

    def read_dbf(dbf_bytes: bytes):
        import struct
        version, yr, mo, dy = struct.unpack("<BBBB", dbf_bytes[0:4])
        num_records      = struct.unpack("<I", dbf_bytes[4:8])[0]
        header_len       = struct.unpack("<H", dbf_bytes[8:10])[0]
        record_len       = struct.unpack("<H", dbf_bytes[10:12])[0]
        fields, pos = [], 32
        while pos < header_len - 1:
            fd = dbf_bytes[pos:pos+32]
            if not fd or fd[0] == 0x0D: break
            name  = fd[:11].split(b"\x00",1)[0].decode("ascii","ignore").strip()
            ftype = chr(fd[11]); flen = fd[16]
            fields.append((name, ftype, flen)); pos += 32
        recs = []
        base = header_len
        for i in range(num_records):
            rec = dbf_bytes[base + i*record_len : base + (i+1)*record_len]
            if not rec or rec[0] == 0x2A:  # deleted
                continue
            off, vals = 1, {}
            for (name, _ft, flen) in fields:
                raw = rec[off:off+flen]
                vals[name] = raw.decode("latin1","ignore").strip()
                off += flen
            recs.append(vals)
        return recs

    def read_shp_centroids(shp_bytes: bytes):
        import struct
        pos, cents = 100, []
        while pos + 8 <= len(shp_bytes):
            cont_len_words = int.from_bytes(shp_bytes[pos+4:pos+8], "big", signed=True)
            pos += 8
            content = shp_bytes[pos:pos + cont_len_words*2]
            pos += cont_len_words*2
            if len(content) < 44:
                continue
            stype = int.from_bytes(content[0:4], "little", signed=True)
            if stype in (3,5):  # polyline / polygon
                xmin = struct.unpack("<d", content[4:12])[0]
                ymin = struct.unpack("<d", content[12:20])[0]
                xmax = struct.unpack("<d", content[20:28])[0]
                ymax = struct.unpack("<d", content[28:36])[0]
                lon  = (xmin+xmax)/2.0
                lat  = (ymin+ymax)/2.0
                cents.append((lon,lat))
        return cents

    recs  = read_dbf(dbf)
    cents = read_shp_centroids(shp)
    rows = []
    for rec, (lon,lat) in zip(recs, cents):
        geoid = rec.get("GEOID","").strip()
        if not geoid:
            geoid = (rec.get("STATEFP","").zfill(2)
                     + rec.get("COUNTYFP","").zfill(3)
                     + rec.get("TRACTCE","").zfill(6))
        if geoid.startswith(STATE_FIPS + COUNTY_FIPS):
            rows.append({"GEOID": geoid, "lon": float(lon), "lat": float(lat)})
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No centroid rows parsed from shapefile.")
    return out

# ---------------- Gazetteer centroids (force pipe delimiter) ----------------
def centroids_from_gazetteer(txt_path: str) -> pd.DataFrame:
    # Census: “Files are pipe-delimited text …” (see docs)
    df = pd.read_csv(txt_path, sep="|", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for need in ("GEOID", "INTPTLAT", "INTPTLONG"):
        if need not in df.columns:
            raise KeyError(f"Gazetteer is missing expected column '{need}'. Columns seen: {list(df.columns)}")
    out = df[["GEOID", "INTPTLONG", "INTPTLAT"]].rename(columns={"INTPTLONG": "lon", "INTPTLAT": "lat"})
    out["lon"] = pd.to_numeric(out["lon"].astype(str).str.strip(), errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"].astype(str).str.strip(), errors="coerce")
    out = out[out["GEOID"].str.startswith(STATE_FIPS + COUNTY_FIPS)].dropna(subset=["lon","lat"])
    if out.empty:
        raise RuntimeError("No Baltimore City rows parsed from Gazetteer.")
    return out

# ---------------- Build interactive figure ----------------
def build_figure(df_metrics: pd.DataFrame, geo):
    metric_order = [
        ("broadband_pct",     VARS["broadband_pct"][1]),
        ("child_poverty_pct", VARS["child_poverty_pct"][1]),
        ("hs_complete_pct",   VARS["hs_complete_pct"][1]),
    ]
    color_scales = {"broadband_pct": "Viridis", "child_poverty_pct": "Plasma", "hs_complete_pct": "Cividis"}
    years = sorted(df_metrics["year"].dropna().unique().tolist())

    fig = go.Figure()
    trace_index = []  # (metric_key, year, idx)

    if isinstance(geo, pd.DataFrame) and {"GEOID","lon","lat"}.issubset(geo.columns):
        # centroid bubble map
        pts = geo[["GEOID","lon","lat"]].copy()
        for mkey, mlabel in metric_order:
            var_code = VARS[mkey][0]
            for y in years:
                sub = df_metrics[df_metrics["year"] == y][["GEOID", var_code]].dropna()
                merged = pts.merge(sub, on="GEOID", how="inner")
                tr = go.Scattermapbox(
                    lon=merged["lon"], lat=merged["lat"], mode="markers",
                    marker=dict(
                        size=8,
                        color=merged[var_code],
                        colorscale=color_scales[mkey],
                        cmin=float(merged[var_code].min()) if not merged.empty else 0,
                        cmax=float(merged[var_code].max()) if not merged.empty else 100,
                        showscale=True,
                        colorbar=dict(title=mlabel),
                    ),
                    text=merged["GEOID"],
                    hovertemplate="Tract: %{text}<br>" + f"{mlabel}: " + "%{marker.color:.1f}%<extra></extra>",
                    visible=False,
                    name=f"{mlabel} {y}",
                )
                fig.add_trace(tr)
                trace_index.append((mkey, y, len(fig.data)-1))
    else:
        raise SystemExit("Internal error: geo must be centroid DataFrame here.")

    if not trace_index:
        raise SystemExit("No traces built — check that your JSON files have data under data/raw/.")

    # Make the first one visible
    m0, y0, idx0 = trace_index[0]
    fig.data[idx0].visible = True

    def vis_for(metric_sel, year_sel):
        vis = [False] * len(fig.data)
        for mkey, y, idx in trace_index:
            if mkey == metric_sel and y == year_sel:
                vis[idx] = True
        return vis

    metric_buttons = []
    for mkey, mlabel in metric_order:
        metric_buttons.append(dict(
            label=mlabel,
            method="update",
            args=[{"visible": vis_for(mkey, y0)}, {"title": f"{mlabel} — {y0}"}],
        ))

    year_buttons = []
    for y in years:
        year_buttons.append(dict(
            label=str(y),
            method="update",
            args=[{"visible": vis_for(m0, y)}, {"title": f"{VARS[m0][1]} — {y}"}],
        ))

    fig.update_layout(
        title=f"{VARS[m0][1]} — {y0}",
        mapbox=dict(style=MAPBOX_STYLE, center=CITY_CENTER, zoom=DEFAULT_ZOOM),
        margin=dict(l=0, r=0, t=50, b=0),
        updatemenus=[
            dict(buttons=metric_buttons, direction="down",
                 x=0.01, y=0.99, xanchor="left", yanchor="top", bgcolor="white"),
            dict(buttons=year_buttons, direction="down",
                 x=0.35, y=0.99, xanchor="left", yanchor="top", bgcolor="white"),
        ],
        annotations=[
            dict(text="Metric", x=0.00, y=1.05, xref="paper", yref="paper", showarrow=False),
            dict(text="Year",   x=0.34, y=1.05, xref="paper", yref="paper", showarrow=False),
        ],
    )
    return fig


# ---------------- CHATBOT FUNCTIONALITY ----------------
"""
CHATBOT AGENT INITIALIZATION - MICROSOFT AUTOGEN MULTI-AGENT FRAMEWORK

This chatbot implementation uses Microsoft's AutoGen framework for multi-agent conversations.
AutoGen enables complex agent interactions with tool calling and code execution capabilities.

KEY FEATURES OF AUTOGEN IMPLEMENTATION:

1. **Multi-Agent Architecture**:
   - AssistantAgent: Powered by GPT-4o for intelligent responses
   - UserProxyAgent: Handles tool execution and code running
   - GroupChat: Manages conversations between multiple agents
   - Enables sophisticated agent-to-agent collaboration
   
2. **Tool Integration**:
   - Custom function_map registers Python functions as tools
   - Agents can call tools autonomously based on conversation
   - Supports data querying, statistics, and visualization
   
3. **Robust Error Handling**:
   - Try-except wraps agent creation
   - Gracefully falls back to rule-based mode if AutoGen fails
   - Tracks initialization errors in self.agent_init_error
   
4. **API Key Validation**:
   - Checks if API key exists before attempting agent creation
   - Validates API key format (should start with 'sk-')
   - Clear warning messages when API key is missing/invalid
   
5. **Conversation Management**:
   - Maintains conversation history across interactions
   - Supports multi-turn conversations with context
   - Agents can call multiple tools in sequence
   
6. **Server Status Endpoints**:
   - /health - Quick health check with agent status
   - /status - Detailed diagnostic information
   - Returns agent_init_error so frontend knows why agent failed

AUTOGEN VS LANGCHAIN:

AutoGen focuses on:
- Multi-agent conversations and collaboration
- Code execution and validation
- Tool calling through function registration
- Agent-to-agent communication patterns

This makes it ideal for complex data analysis tasks where agents need to:
- Query data from multiple sources
- Execute calculations and transformations
- Generate visualizations programmatically
- Validate results through multi-agent review
"""

class BaltimoreHealthChatbot:
    """
    Chatbot class that integrates with the existing Baltimore Health Dashboard data.
    Uses Microsoft AutoGen framework for multi-agent AI conversations.
    
    Features:
    - Multiple specialized agents (assistant, user proxy)
    - Tool calling for data access and analysis
    - Code execution for complex calculations
    - Comprehensive error handling and status tracking
    """

    def __init__(self, openai_api_key: str = None):
        """
        Initialize the chatbot with existing dashboard data.

        Args:
            openai_api_key: OpenAI API key for AutoGen (optional)
        """
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.agent = None  # Initialize as None by default
        self.agent_init_error = None  # Track initialization errors
        
        # Use existing data loading functions
        self.data = self._load_dashboard_data()
        self.geo_data = self._load_geographic_data()

        # Initialize the agent if AutoGen is available
        if AUTOGEN_AVAILABLE:
            if not self.api_key:
                self.agent_init_error = "No OpenAI API key provided"
                print("⚠️  Warning: No OpenAI API key found. Set OPENAI_API_KEY in .env file or environment variable.")
                print("   Chatbot will use fallback mode without AI agent.")
            else:
                # Try to create the agent with comprehensive error handling
                try:
                    print("🔧 Initializing AutoGen multi-agent system...")
                    self.agent = self._create_agent()
                    if self.agent:
                        print("✅ AutoGen agents successfully initialized!")
                    else:
                        self.agent_init_error = "Agent creation returned None"
                        print("⚠️  Warning: Agent creation returned None (unexpected)")
                except Exception as e:
                    self.agent_init_error = str(e)
                    print(f"⚠️  Warning: Failed to initialize agent: {e}")
                    print("   Chatbot will use fallback mode without AI agent.")
        else:
            self.agent_init_error = "AutoGen not available"
            print("⚠️  AutoGen not available - chatbot will use fallback mode")

    def _load_dashboard_data(self) -> pd.DataFrame:
        """
        Load dashboard data using existing functions and transform to long format.

        Returns:
            Combined DataFrame with all metrics and years in long format
        """
        try:
            # Use existing data loading logic (wide format)
            df_wide = combine_metrics(YEARS)
            if df_wide.empty:
                print("Warning: No metric data found. Please ensure JSON files are in data/raw/ directory.")
                return pd.DataFrame()

            # Transform from wide format to long format for chatbot
            df_list = []

            for _, (var_code, description) in VARS.items():
                if var_code in df_wide.columns:
                    # Create a long format entry for this metric
                    metric_df = df_wide[['GEOID', 'year', var_code]].copy()
                    metric_df = metric_df.rename(columns={var_code: var_code})
                    metric_df['metric_key'] = self._get_metric_key_from_var_code(var_code)
                    metric_df['metric_name'] = description
                    metric_df['var_code'] = var_code

                    # Only include rows with valid data
                    metric_df = metric_df.dropna(subset=[var_code])
                    df_list.append(metric_df)

            if not df_list:
                print("Warning: No valid metric data found after transformation.")
                return pd.DataFrame()

            # Combine all metrics into long format
            df = pd.concat(df_list, ignore_index=True)

            print(f"Loaded dashboard data for chatbot: {len(df)} tract-year-metric combinations")
            print(f"Available metrics: {df['metric_key'].unique()}")

            return df

        except Exception as e:
            print(f"Error loading dashboard data for chatbot: {e}")
            return pd.DataFrame()

    def _get_metric_key(self, row) -> str:
        """Helper to determine metric key from variable codes."""
        for key, (var_code, _) in VARS.items():
            if var_code in row.index and pd.notna(row[var_code]):
                return key
        return "unknown"

    def _get_metric_key_from_var_code(self, var_code: str) -> str:
        """Helper to determine metric key from variable code."""
        var_to_key = {
            'S2801_C02_017E': 'broadband_pct',
            'S1701_C03_002E': 'child_poverty_pct',
            'S1501_C02_014E': 'hs_complete_pct'
        }
        return var_to_key.get(var_code, 'unknown')

    def _load_geographic_data(self) -> pd.DataFrame:
        """
        Load geographic data using existing functions.

        Returns:
            DataFrame with GEOID, latitude, longitude
        """
        try:
            geo_avail = ensure_geo()
            if geo_avail["gaz_txt"]:
                geo = centroids_from_gazetteer(geo_avail["gaz_txt"])
                print(f"Loaded geographic data for chatbot: {len(geo)} Baltimore City tracts")
                return geo
        except Exception as e:
            print(f"Error loading geographic data for chatbot: {e}")
        return pd.DataFrame()

    def _create_agent(self) -> Dict:
        """
        Create AutoGen multi-agent system with custom functions for data interaction.
        
        AutoGen uses a different architecture than LangChain:
        - Multiple specialized agents (AssistantAgent, UserProxyAgent)
        - Function registration through function_map
        - Conversational agent interactions
        
        Returns:
            Dict containing {'assistant': AssistantAgent, 'user_proxy': UserProxyAgent}
            or None if creation fails
        """
        if not AUTOGEN_AVAILABLE:
            print("   AutoGen not available")
            return None
        
        if not self.api_key:
            print("   Cannot create agent: No API key")
            return None

        try:
            # Validate API key format (basic check)
            if not self.api_key.startswith('sk-'):
                print(f"   Warning: API key doesn't start with 'sk-' - might be invalid")
            
            # Configure LLM for AutoGen
            llm_config = {
                "config_list": [
                    {
                        "model": "gpt-4o",
                        "api_key": self.api_key,
                        "temperature": 0,
                    }
                ],
                "timeout": 120,
            }
            
            # System message for the assistant
            system_message = """You are a helpful assistant for the Baltimore City Health Dashboard.
            
You can help users understand health metrics data including:
- Broadband connection (%)
- Children in poverty (%)
- High school completion, 25+ (%)

Available years: 2018, 2020, 2022, 2023

Available functions:
- query_metric_data: Query health metric data by location, metric type, or year
- get_tract_info: Get information about a specific census tract
- compare_metrics: Compare different metrics or years
- find_extreme_values: Find tracts with highest/lowest values for a metric
- get_summary_stats: Get summary statistics for metrics

When answering questions, use the appropriate functions and be specific about which metric, year, and location you're discussing.

IMPORTANT: After providing your complete answer, always end your response with the word TERMINATE on a new line to signal completion."""

            # Create AssistantAgent (powered by GPT-4o)
            print(f"   Creating AssistantAgent with model: gpt-4o")
            assistant = AssistantAgent(
                name="BaltimoreHealthAssistant",
                system_message=system_message,
                llm_config=llm_config,
            )
            
            # Create UserProxyAgent (executes functions)
            print(f"   Creating UserProxyAgent for function execution...")
            user_proxy = UserProxyAgent(
                name="UserProxy",
                human_input_mode="NEVER",  # Never ask for human input
                max_consecutive_auto_reply=3,  # Reduced to prevent infinite loops
                code_execution_config=False,  # Disable code execution for security
                is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE") or "TERMINATE" in x.get("content", ""),
                function_map={
                    "query_metric_data": self._query_metric_data,
                    "get_tract_info": self._get_tract_info,
                    "compare_metrics": self._compare_metrics,
                    "find_extreme_values": self._find_extreme_values,
                    "get_summary_stats": self._get_summary_stats,
                }
            )
            
            print(f"   AutoGen multi-agent system created successfully")
            
            return {
                'assistant': assistant,
                'user_proxy': user_proxy
            }
            
        except Exception as e:
            print(f"   Error creating agent: {type(e).__name__}: {str(e)}")
            # Re-raise the exception so it can be caught by __init__
            raise

    def _query_metric_data(self, query: str) -> str:
        """
        Query health metric data based on user specifications.

        Args:
            query: Natural language query specifying metric, location, year, etc.

        Returns:
            Formatted string with query results
        """
        # Parse the query to extract parameters
        query_lower = query.lower()

        # Determine metric
        metric_key = None
        for key, (_, description) in VARS.items():
            if any(word in query_lower for word in description.lower().split()):
                metric_key = key
                break

        # Determine year
        year = None
        for y in YEARS:
            if str(y) in query:
                year = y
                break

        # Filter data based on parameters
        filtered_data = self.data.copy()

        if metric_key and metric_key != "unknown":
            filtered_data = filtered_data[filtered_data['metric_key'] == metric_key]

        if year:
            filtered_data = filtered_data[filtered_data['year'] == year]

        # Format results
        if filtered_data.empty:
            return "No data found matching your query. Please check your parameters."

        # Group by metric for summary
        summary = filtered_data.groupby(['metric_key', 'year']).agg({
            filtered_data.select_dtypes(include=[np.number]).columns[0]: ['count', 'mean', 'min', 'max']
        }).round(2)

        result = "Query Results:\n\n"
        for (metric_key, year), stats in summary.iterrows():
            metric_name = VARS.get(metric_key, ("", "Unknown"))[1]
            result += f"{metric_name} ({year}):\n"
            result += f"  - Tracts with data: {int(stats.iloc[0])}\n"
            result += f"  - Average: {stats.iloc[1]:.2f}%\n"
            result += f"  - Range: {stats.iloc[2]:.2f}% - {stats.iloc[3]:.2f}%\n\n"

        return result

    def _get_tract_info(self, tract_geoid: str) -> str:
        """
        Get detailed information about a specific census tract.

        Args:
            tract_geoid: The GEOID of the tract to look up

        Returns:
            Formatted string with tract information
        """
        if tract_geoid not in self.data['GEOID'].values:
            return f"Tract {tract_geoid} not found in the dataset."

        tract_data = self.data[self.data['GEOID'] == tract_geoid]

        result = f"Information for Census Tract {tract_geoid}:\n\n"

        # Show data for each metric and year
        for _, row in tract_data.iterrows():
            metric_name = VARS.get(row['metric_key'], ("", "Unknown"))[1]
            # Get the metric value using the proper column mapping
            metric_column_map = {
                'broadband_pct': 'S2801_C02_017E',
                'child_poverty_pct': 'S1701_C03_002E',
                'hs_complete_pct': 'S1501_C02_014E'
            }
            metric_column = metric_column_map.get(row['metric_key'], 'S2801_C02_017E')
            value = row[metric_column]
            result += f"{metric_name} ({row['year']}): {value:.2f}%\n"

        return result

    def _compare_metrics(self, query: str) -> str:
        """
        Compare different metrics or time periods.

        Args:
            query: Query specifying what to compare

        Returns:
            Formatted comparison results
        """
        if 'vs' in query.lower() or 'compare' in query.lower():
            # Basic comparison between metrics
            recent_data = self.data[self.data['year'] == max(YEARS)]

            comparison = recent_data.groupby('metric_key').agg({
                recent_data.select_dtypes(include=[np.number]).columns[0]: 'mean'
            }).round(2)

            result = "Comparison of Average Values (Most Recent Year):\n\n"
            for metric_key, avg_value in comparison.iterrows():
                metric_name = VARS.get(metric_key, ("", "Unknown"))[1]
                result += f"{metric_name}: {avg_value.iloc[0]:.2f}%\n"

            return result

        return "Please specify what you'd like to compare (e.g., 'compare broadband vs poverty rates')"

    def _find_extreme_values(self, query: str) -> str:
        """
        Find tracts with highest or lowest values for a metric.

        Args:
            query: Query specifying metric and extreme to find

        Returns:
            Formatted results showing extreme values
        """
        query_lower = query.lower()

        # Determine metric
        metric_key = None
        for key, (_, description) in VARS.items():
            if any(word in query_lower for word in description.lower().split()):
                metric_key = key
                break

        if not metric_key:
            return "Please specify which metric you want to analyze (broadband, poverty, or education)."

        # Determine if looking for highest or lowest
        is_highest = any(word in query_lower for word in ['highest', 'best', 'most', 'top'])
        is_lowest = any(word in query_lower for word in ['lowest', 'worst', 'least', 'bottom'])

        if not is_highest and not is_lowest:
            return "Please specify if you want highest or lowest values (e.g., 'tracts with highest broadband access')."

        # Determine year
        year = max(YEARS)  # Default to most recent
        for y in YEARS:
            if str(y) in query:
                year = y
                break

        # Filter data
        metric_data = self.data[
            (self.data['metric_key'] == metric_key) &
            (self.data['year'] == year)
        ]

        if metric_data.empty:
            metric_name = VARS.get(metric_key, ("", "Unknown"))[1]
            return f"No data found for {metric_name} in {year}."

        # Sort and get top/bottom 5
        ascending = is_lowest
        top_tracts = metric_data.nlargest(5, metric_data.columns[2]) if not ascending else metric_data.nsmallest(5, metric_data.columns[2])

        metric_name = VARS.get(metric_key, ("", "Unknown"))[1]
        extreme_word = "highest" if is_highest else "lowest"

        result = f"Tracts with {extreme_word} {metric_name.lower()} ({year}):\n\n"

        for _, row in top_tracts.iterrows():
            # Get the metric value using the proper column
            metric_column = 'S2801_C02_017E'  # This is in the _find_extreme_values function for broadband
            value = row[metric_column]
            result += f"Tract {row['GEOID']}: {value:.2f}%\n"

        return result

    def _get_summary_stats(self, query: str) -> str:
        """
        Get summary statistics for metrics.

        Args:
            query: Query specifying what statistics to provide

        Returns:
            Formatted summary statistics
        """
        # Basic summary across all metrics and years
        summary = self.data.groupby(['metric_key', 'year']).agg({
            self.data.select_dtypes(include=[np.number]).columns[0]: ['count', 'mean', 'std', 'min', 'max']
        }).round(2)

        result = "Summary Statistics for Baltimore Health Metrics:\n\n"

        for (metric_key, year), stats in summary.iterrows():
            metric_name = VARS.get(metric_key, ("", "Unknown"))[1]
            result += f"{metric_name} ({year}):\n"
            result += f"  - Tracts: {int(stats.iloc[0])}\n"
            result += f"  - Mean: {stats.iloc[1]:.2f}%\n"
            result += f"  - Std Dev: {stats.iloc[2]:.2f}%\n"
            result += f"  - Range: {stats.iloc[3]:.2f}% - {stats.iloc[4]:.2f}%\n\n"

        return result

    def _create_visualization(self, query: str) -> str:
        """
        Create interactive visualization for query results.

        Args:
            query: Query specifying what to visualize

        Returns:
            Path to generated visualization file or error message
        """
        query_lower = query.lower()

        # Determine what type of visualization to create
        if 'map' in query_lower or 'geographic' in query_lower:
            return self._create_map_visualization(query)
        elif 'trend' in query_lower or 'time' in query_lower or 'year' in query_lower:
            return self._create_trend_visualization(query)
        elif 'comparison' in query_lower or 'compare' in query_lower:
            return self._create_comparison_visualization(query)
        else:
            return self._create_summary_visualization(query)

    def _create_map_visualization(self, query: str) -> str:
        """
        Create an interactive map visualization showing geographic distribution.

        Args:
            query: Query specifying the map visualization details

        Returns:
            Path to generated HTML file
        """
        try:
            # Determine metric and year
            metric_key = None
            year = max(YEARS)  # Default to most recent

            for key, (_, description) in VARS.items():
                if any(word in query.lower() for word in description.lower().split()):
                    metric_key = key
                    break

            # Check for specific year
            for y in YEARS:
                if str(y) in query:
                    year = y
                    break

            if not metric_key:
                return "Please specify which metric to visualize (broadband, poverty, or education)."

            # Get data for visualization
            map_data = self.data[
                (self.data['metric_key'] == metric_key) &
                (self.data['year'] == year)
            ].copy()

            if map_data.empty:
                metric_name = VARS.get(metric_key, ("", "Unknown"))[1]
                return f"No data available for {metric_name} in {year}."

            # Merge with geographic data
            map_data = map_data.merge(self.geo_data, on='GEOID', how='inner')

            # Create the map using existing dashboard styling
            metric_name = VARS.get(metric_key, ("", "Unknown"))[1]
            color_scales = {"broadband_pct": "Viridis", "child_poverty_pct": "Plasma", "hs_complete_pct": "Cividis"}

            fig = go.Figure()

            # Add scatter points for each tract
            fig.add_trace(go.Scattermapbox(
                lon=map_data['lon'],
                lat=map_data['lat'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=map_data.iloc[:, 2],  # The metric value column
                    colorscale=color_scales.get(metric_key, 'Viridis'),
                    cmin=float(map_data.iloc[:, 2].min()),
                    cmax=float(map_data.iloc[:, 2].max()),
                    showscale=True,
                    colorbar=dict(title=metric_name),
                ),
                text=[f"Tract {geoid}<br>{metric_name}: {val:.2f}%" for geoid, val in zip(map_data['GEOID'], map_data.iloc[:, 2])],
                hovertemplate="%{text}<extra></extra>",
                name=f"{metric_name} ({year})"
            ))

            # Update layout using existing dashboard settings
            fig.update_layout(
                title=f"{metric_name} by Census Tract ({year})",
                mapbox=dict(
                    style=MAPBOX_STYLE,
                    center=CITY_CENTER,
                    zoom=DEFAULT_ZOOM
                ),
                margin=dict(l=0, r=0, t=50, b=0),
                height=600
            )

            # Save visualization
            viz_path = os.path.join(os.path.dirname(OUT_HTML), f"chatbot_map_{metric_key}_{year}.html")
            fig.write_html(viz_path, include_plotlyjs="cdn", full_html=True)

            return f"Interactive map visualization created: {viz_path}"

        except Exception as e:
            return f"Error creating map visualization: {str(e)}"

    def _create_trend_visualization(self, query: str) -> str:
        """
        Create a trend visualization showing changes over time.

        Args:
            query: Query specifying the trend visualization details

        Returns:
            Path to generated HTML file
        """
        try:
            # Determine metric
            metric_key = None
            for key, (_, description) in VARS.items():
                if any(word in query.lower() for word in description.lower().split()):
                    metric_key = key
                    break

            if not metric_key:
                return "Please specify which metric to visualize (broadband, poverty, or education)."

            # Get trend data
            trend_data = self.data[self.data['metric_key'] == metric_key].copy()

            # Calculate yearly averages
            yearly_avg = trend_data.groupby('year').agg({
                trend_data.select_dtypes(include=[np.number]).columns[0]: 'mean'
            }).reset_index()

            metric_name = VARS.get(metric_key, ("", "Unknown"))[1]

            # Create trend chart
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=yearly_avg['year'],
                y=yearly_avg.iloc[:, 1],
                mode='lines+markers',
                marker=dict(size=10),
                line=dict(width=3),
                name=metric_name
            ))

            fig.update_layout(
                title=f"{metric_name} Trend (2018-2023)",
                xaxis_title="Year",
                yaxis_title=f"{metric_name} (%)",
                template='plotly_white',
                height=500
            )

            # Save visualization
            viz_path = os.path.join(os.path.dirname(OUT_HTML), f"chatbot_trend_{metric_key}.html")
            fig.write_html(viz_path, include_plotlyjs="cdn", full_html=True)

            return f"Trend visualization created: {viz_path}"

        except Exception as e:
            return f"Error creating trend visualization: {str(e)}"

    def _create_comparison_visualization(self, query: str) -> str:
        """
        Create a comparison visualization between metrics.

        Args:
            query: Query specifying the comparison details

        Returns:
            Path to generated HTML file
        """
        try:
            # Get most recent year data for comparison
            recent_year = max(YEARS)
            comparison_data = self.data[self.data['year'] == recent_year].copy()

            # Calculate average for each metric
            metric_avgs = comparison_data.groupby('metric_name').agg({
                comparison_data.select_dtypes(include=[np.number]).columns[0]: 'mean'
            }).reset_index()

            # Create comparison bar chart
            fig = go.Figure()

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            for i, (_, row) in enumerate(metric_avgs.iterrows()):
                fig.add_trace(go.Bar(
                    x=[row['metric_name']],
                    y=[row.iloc[1]],
                    name=row['metric_name'],
                    marker_color=colors[i % len(colors)]
                ))

            fig.update_layout(
                title=f"Baltimore Health Metrics Comparison ({recent_year})",
                xaxis_title="Metric",
                yaxis_title="Average Value (%)",
                template='plotly_white',
                height=500
            )

            # Save visualization
            viz_path = os.path.join(os.path.dirname(OUT_HTML), f"chatbot_comparison_{recent_year}.html")
            fig.write_html(viz_path, include_plotlyjs="cdn", full_html=True)

            return f"Comparison visualization created: {viz_path}"

        except Exception as e:
            return f"Error creating comparison visualization: {str(e)}"

    def _create_summary_visualization(self, query: str) -> str:
        """
        Create a summary dashboard visualization.

        Args:
            query: Query specifying the summary details

        Returns:
            Path to generated HTML file
        """
        # Simplified implementation for now
        return "Summary visualization creation is not yet implemented in this version."

    def chat(self, message: str, chat_history: List = None) -> str:
        """
        Process a user message through AutoGen multi-agent system and return a response.
        
        AutoGen Pattern:
        1. User_proxy initiates the conversation with the assistant
        2. Assistant processes the message (may call functions)
        3. User_proxy executes any requested functions
        4. Conversation continues until termination

        Args:
            message: User's natural language message
            chat_history: Previous conversation messages (not used in AutoGen pattern)

        Returns:
            Assistant's response to the user
        """
        if chat_history is None:
            chat_history = []

        if self.agent is None:
            # Fallback to simple rule-based responses if AutoGen not available
            return self._simple_chat_response(message)

        try:
            # AutoGen uses a dict with 'assistant' and 'user_proxy' agents
            assistant = self.agent['assistant']
            user_proxy = self.agent['user_proxy']
            
            # Clear previous chat history using AutoGen's clear_history method
            # This prevents context accumulation across different queries
            try:
                if hasattr(user_proxy, 'clear_history'):
                    user_proxy.clear_history()
                if hasattr(assistant, 'clear_history'):
                    assistant.clear_history()
            except Exception as clear_error:
                # If clear_history doesn't exist or fails, continue anyway
                print(f"Note: Could not clear history: {clear_error}")
            
            # Initiate chat - user_proxy sends message to assistant
            # The conversation will automatically handle function calls
            chat_result = user_proxy.initiate_chat(
                assistant,
                message=message,
                max_turns=2,  # Limit conversation to prevent loops
                clear_history=True,  # Clear history at start of new conversation
                silent=True  # Suppress verbose output
            )
            
            # Extract the response from chat_result
            response_text = None
            
            # Method 1: Try to get from chat_result.summary (most reliable)
            if chat_result and hasattr(chat_result, 'summary'):
                response_text = chat_result.summary
            
            # Method 2: Try to get from chat_result.chat_history
            if not response_text and chat_result and hasattr(chat_result, 'chat_history'):
                # Get the last assistant message from chat history
                for msg in reversed(chat_result.chat_history):
                    if isinstance(msg, dict) and msg.get('role') == 'assistant' and msg.get('content'):
                        response_text = msg['content']
                        break
            
            # Method 3: Fallback to assistant's last_message
            if not response_text and hasattr(assistant, 'last_message'):
                last_msg = assistant.last_message(user_proxy)
                if last_msg and isinstance(last_msg, dict):
                    response_text = last_msg.get('content', '')
            
            # Method 4: Try assistant's chat_messages (read-only property)
            if not response_text and hasattr(assistant, 'chat_messages'):
                try:
                    conversation = assistant.chat_messages.get(user_proxy, [])
                    if conversation:
                        for msg in reversed(conversation):
                            if isinstance(msg, dict) and msg.get('role') == 'assistant' and msg.get('content'):
                                response_text = msg['content']
                                break
                except Exception as e:
                    print(f"Note: Could not read chat_messages: {e}")
            
            # Clean up the response (remove TERMINATE keyword and extra whitespace)
            if response_text:
                # Remove TERMINATE and any trailing whitespace
                response_text = str(response_text).replace("TERMINATE", "").strip()
                if response_text:
                    return response_text
            
            # Fallback if we can't extract the response
            return "Hello! I'm the Baltimore Health Dashboard assistant. I can help you explore health metrics including broadband access, child poverty rates, and educational attainment across Baltimore census tracts. What would you like to know?"

        except Exception as e:
            print(f"AutoGen chat error: {e}")
            import traceback
            traceback.print_exc()
            return f"I apologize, but I encountered an error processing your request: {str(e)}. Please try rephrasing your question."

    def _simple_chat_response(self, message: str) -> str:
        """
        Fallback chat response when AutoGen is not available.
        Routes queries directly to the appropriate tool functions (same as agent uses).

        Args:
            message: User's message

        Returns:
            Response from tool functions
        """
        message_lower = message.lower().strip()

        # Route to appropriate tool based on query pattern
        # These are the SAME tools the agent uses!
        
        # Check for extreme value queries (highest/lowest)
        if any(word in message_lower for word in ['highest', 'lowest', 'best', 'worst', 'top', 'bottom', 'most', 'least']):
            return self._find_extreme_values(message)
        
        # Check for specific tract lookup
        import re
        if re.search(r'\b\d{11}\b', message):
            tract_match = re.search(r'\b\d{11}\b', message)
            if tract_match:
                return self._get_tract_info(tract_match.group())
        
        # Check for comparison queries
        if any(word in message_lower for word in ['compare', 'versus', 'vs', 'difference']):
            return self._compare_metrics(message)
        
        # Check for visualization requests
        if any(word in message_lower for word in ['map', 'chart', 'graph', 'visualization', 'plot']):
            return self._create_visualization(message)
        
        # Check for summary statistics
        if any(word in message_lower for word in ['summary', 'statistics', 'stats', 'overall', 'average', 'mean']):
            return self._get_summary_stats(message)
        
        # Default to general query (handles "what is", "show me", etc.)
        return self._query_metric_data(message)


# Global chatbot instance
chatbot_instance = None


# ---------------- FLASK SERVER (integrated from chatbot_server.py) ----------------
def create_flask_app():
    """
    Create and configure the Flask application with chatbot endpoints.
    This integrates the functionality from chatbot_server.py.
    """
    if not FLASK_AVAILABLE:
        raise RuntimeError("Flask is not available. Install with: pip install flask flask-cors")
    
    app = Flask(__name__)
    CORS(app)  # Enable CORS for local development
    
    # Initialize the chatbot instance globally with comprehensive error handling
    global chatbot_instance
    if chatbot_instance is None:
        print("\n" + "="*60)
        print("🚀 Initializing BaltimoreHealthChatbot for server mode...")
        print("="*60)
        try:
            chatbot_instance = BaltimoreHealthChatbot()
            if chatbot_instance.agent is not None:
                print("✅ Chatbot initialized with AI agent!")
            else:
                print("⚠️  Chatbot initialized in fallback mode (no agent)")
                if chatbot_instance.agent_init_error:
                    print(f"   Reason: {chatbot_instance.agent_init_error}")
        except Exception as e:
            print(f"❌ Error initializing chatbot: {e}")
            print("   Server will start but chatbot may not work correctly")
            # Create a minimal instance so the server doesn't crash
            chatbot_instance = None
        print("="*60 + "\n")
    
    @app.route('/chat', methods=['POST'])
    def chat():
        """
        Handle chat requests from the web interface.
        Uses the REAL BaltimoreHealthChatbot with agent.
        """
        try:
            if chatbot_instance is None:
                return jsonify({
                    'error': 'Chatbot not initialized. Check server logs for details.',
                    'using_agent': False
                }), 500
            
            data = request.get_json()
            user_message = data.get('message', '')
            
            if not user_message:
                return jsonify({'error': 'No message provided'}), 400
            
            # Call the REAL chatbot (uses agent if available)
            response = chatbot_instance.chat(user_message)
            
            return jsonify({
                'response': response,
                'using_agent': chatbot_instance.agent is not None,
                'agent_init_error': chatbot_instance.agent_init_error
            })
        
        except Exception as e:
            print(f"Error in chat endpoint: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint with detailed diagnostics."""
        if chatbot_instance is None:
            return jsonify({
                'status': 'error',
                'agent_available': False,
                'error': 'Chatbot not initialized'
            }), 500
        
        return jsonify({
            'status': 'ok',
            'agent_available': chatbot_instance.agent is not None,
            'agent_init_error': chatbot_instance.agent_init_error,
            'autogen_available': AUTOGEN_AVAILABLE,
            'api_key_set': bool(chatbot_instance.api_key),
            'data_loaded': not chatbot_instance.data.empty if hasattr(chatbot_instance, 'data') else False
        })
    
    @app.route('/status', methods=['GET'])
    def status():
        """Detailed status endpoint for debugging."""
        if chatbot_instance is None:
            return jsonify({
                'initialized': False,
                'error': 'Chatbot instance is None'
            })
        
        return jsonify({
            'initialized': True,
            'agent_available': chatbot_instance.agent is not None,
            'agent_type': str(type(chatbot_instance.agent)) if chatbot_instance.agent else None,
            'agent_init_error': chatbot_instance.agent_init_error,
            'autogen_available': AUTOGEN_AVAILABLE,
            'flask_available': FLASK_AVAILABLE,
            'api_key_present': bool(chatbot_instance.api_key),
            'api_key_starts_with_sk': chatbot_instance.api_key.startswith('sk-') if chatbot_instance.api_key else False,
            'data_rows': len(chatbot_instance.data) if hasattr(chatbot_instance, 'data') else 0,
            'geo_data_rows': len(chatbot_instance.geo_data) if hasattr(chatbot_instance, 'geo_data') else 0
        })
    
    return app


def run_server(host='0.0.0.0', port=5001, debug=True):
    """
    Run the Flask server for chatbot functionality.
    This replaces the need for a separate chatbot_server.py file.
    """
    app = create_flask_app()
    
    print("\n" + "="*60)
    print("🚀 Baltimore Health Chatbot Server")
    print("="*60)
    
    if chatbot_instance:
        if chatbot_instance.agent is not None:
            print("✅ Agent Status: ACTIVE - Using AI-powered responses")
        else:
            print("⚠️  Agent Status: FALLBACK MODE - Using rule-based responses")
            if chatbot_instance.agent_init_error:
                print(f"   Error: {chatbot_instance.agent_init_error}")
    else:
        print("❌ Agent Status: NOT INITIALIZED")
    
    print(f"\n🌐 Server running on: http://localhost:{port}")
    print(f"\n📊 Endpoints:")
    print(f"   • POST http://localhost:{port}/chat     - Chat with the bot")
    print(f"   • GET  http://localhost:{port}/health   - Health check")
    print(f"   • GET  http://localhost:{port}/status   - Detailed status")
    print("\n📂 Open dashboard_multi_year.html in your browser")
    print("="*60 + "\n")
    
    app.run(host=host, port=port, debug=debug)


def create_integrated_dashboard(fig, df, geo):
    """
    Create a single integrated dashboard with embedded chatbot.
    This replaces both dashboard_multi_year.html and enhanced_dashboard_with_chatbot.html
    """
    global chatbot_instance
    
    # Initialize chatbot if needed
    if chatbot_instance is None:
        try:
            chatbot_instance = BaltimoreHealthChatbot()
        except Exception as e:
            print(f"[warn] Could not initialize chatbot: {e}")
    
    # Prepare data for embedding
    # Transform to long format for chatbot
    data_for_js = []
    for mkey, (var_code, description) in VARS.items():
        if var_code in df.columns:
            for _, row in df[['GEOID', 'year', var_code]].dropna(subset=[var_code]).iterrows():
                data_for_js.append({
                    'tract': str(row['GEOID']),
                    'year': int(row['year']),
                    'metric_key': mkey,
                    'metric_name': description,
                    'value': float(row[var_code])
                })
    
    # Convert to JSON for embedding
    import json
    embedded_data = json.dumps(data_for_js)
    
    # First, create the base HTML with Plotly
    html_string = fig.to_html(include_plotlyjs='cdn', full_html=False, div_id='plotly-map')
    
    # Now wrap it with our integrated interface
    integrated_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Baltimore City Health Dashboard with AI Chatbot</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            overflow: hidden;
            background: #ffffff;
        }}
        
        .dashboard-container {{
            display: flex;
            height: 100vh;
            position: relative;
        }}
        
        /* Map Section */
        .map-section {{
            flex: 1;
            position: relative;
            transition: all 0.3s ease;
        }}
        
        .map-section.chat-open {{
            margin-right: 400px;
        }}
        
        #plotly-map {{
            width: 100%;
            height: 100%;
        }}
        
        /* Chat Sidebar */
        #chatSidebar {{
            position: fixed;
            right: 0;
            top: 0;
            width: 400px;
            height: 100vh;
            background: white;
            box-shadow: -2px 0 20px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
            z-index: 1000;
            display: flex;
            flex-direction: column;
        }}
        
        #chatSidebar.collapsed {{
            transform: translateX(100%);
        }}
        
        /* Toggle Button */
        #toggleChat {{
            position: fixed;
            right: 20px;
            top: 20px;
            z-index: 1001;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            width: 56px;
            height: 56px;
            border-radius: 50%;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        #toggleChat:hover {{
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }}
        
        #toggleChat i {{
            font-size: 24px;
        }}
        
        /* Chat Header */
        .chat-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 0;
        }}
        
        .chat-header h5 {{
            margin: 0;
            font-weight: 600;
        }}
        
        /* Quick Questions */
        .quick-questions {{
            padding: 15px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            overflow-x: auto;
            white-space: nowrap;
        }}
        
        .quick-question-btn {{
            display: inline-block;
            padding: 8px 16px;
            margin: 4px;
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.85rem;
        }}
        
        .quick-question-btn:hover {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: transparent;
        }}
        
        /* Chat Messages */
        .chat-messages {{
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }}
        
        .message {{
            margin-bottom: 15px;
            animation: fadeIn 0.3s ease;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .message-bubble {{
            padding: 12px 16px;
            border-radius: 16px;
            max-width: 85%;
            word-wrap: break-word;
        }}
        
        .user-message .message-bubble {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }}
        
        .bot-message .message-bubble {{
            background: white;
            border: 1px solid #dee2e6;
            border-bottom-left-radius: 4px;
        }}
        
        /* Chat Input */
        .chat-input {{
            padding: 20px;
            background: white;
            border-top: 1px solid #dee2e6;
        }}
        
        .input-group {{
            display: flex;
            gap: 10px;
        }}
        
        .input-group input {{
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #dee2e6;
            border-radius: 25px;
            font-size: 14px;
            transition: all 0.2s;
        }}
        
        .input-group input:focus {{
            outline: none;
            border-color: #667eea;
        }}
        
        .input-group button {{
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .input-group button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        
        /* Loading */
        .loading {{
            display: none;
            text-align: center;
            padding: 10px;
            color: #667eea;
        }}
        
        .loading i {{
            margin-right: 8px;
        }}
        
        /* Scrollbar */
        .chat-messages::-webkit-scrollbar {{
            width: 6px;
        }}
        
        .chat-messages::-webkit-scrollbar-track {{
            background: #f1f1f1;
        }}
        
        .chat-messages::-webkit-scrollbar-thumb {{
            background: #667eea;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Map Section -->
        <div class="map-section" id="mapSection">
            {html_string}
        </div>
        
        <!-- Toggle Button -->
        <button id="toggleChat" onclick="toggleChatSidebar()">
            <i class="fas fa-comments"></i>
        </button>
        
        <!-- Chat Sidebar -->
        <div id="chatSidebar">
            <div class="chat-header">
                <h5><i class="fas fa-robot me-2"></i>AI Assistant</h5>
                <small>Ask me about Baltimore's health data</small>
            </div>
            
            <div class="quick-questions">
                <span class="quick-question-btn" onclick="sendQuickQuestion('Show me tracts with highest broadband access in 2023')">
                    <i class="fas fa-wifi me-1"></i>📶 Broadband
                </span>
                <span class="quick-question-btn" onclick="sendQuickQuestion('Which areas have lowest poverty rates?')">
                    <i class="fas fa-chart-line me-1"></i>💰 Poverty
                </span>
                <span class="quick-question-btn" onclick="sendQuickQuestion('Tell me about tract 24510010100')">
                    <i class="fas fa-map-marker-alt me-1"></i>📍 Tract Info
                </span>
                <span class="quick-question-btn" onclick="sendQuickQuestion('What is the average education level?')">
                    <i class="fas fa-graduation-cap me-1"></i>🎓 Education
                </span>
            </div>
            
            <!-- Chat Messages -->
            <div class="chat-messages" id="chatMessages">
                    <div class="bot-message message">
                        <div class="message-bubble">
                            <strong><i class="fas fa-robot me-1"></i>AI Assistant with AutoGen:</strong><br>
                            Hello! I use Microsoft's AutoGen multi-agent framework powered by GPT-4o.<br><br>
                            <strong>🤖 Agent Capabilities:</strong><br>
                            ✅ Find tracts with highest/lowest values<br>
                            ✅ Look up specific census tracts<br>
                            ✅ Calculate averages and statistics<br>
                            ✅ Show historical trends (2018-2023)<br>
                            ✅ Multi-agent collaboration & reasoning<br><br>
                            <strong>💡 Try asking:</strong><br>
                            • "Show me tracts with highest broadband access in 2023"<br>
                            • "What is the average poverty rate?"<br>
                            • "Tell me about tract 24510010100"<br><br>
                            <small>⚠️ <strong>Server Required:</strong> Make sure the server is running!<br>
                            Run: <code>python3 Baltimore_MetricsWithMap.py --server</code></small>
                        </div>
                    </div>
            </div>
            
            <!-- Loading Indicator -->
            <div class="loading" id="loading">
                <i class="fas fa-spinner fa-spin"></i> Thinking...
            </div>
            
            <!-- Chat Input -->
            <div class="chat-input">
                <div class="input-group">
                    <input type="text" id="userInput" placeholder="Ask me anything about Baltimore's health data..." 
                           onkeypress="if(event.key==='Enter') sendMessage()">
                    <button onclick="sendMessage()">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Embedded dashboard data - REAL DATA FROM PYTHON
        const dashboardData = {embedded_data};
        let chatHistory = [];
        let agentStatus = null;
        
        // Check server status on load
        async function checkServerStatus() {{
            try {{
                const response = await fetch('http://localhost:5001/status');
                if (response.ok) {{
                    const status = await response.json();
                    agentStatus = status;
                    
                    // Update welcome message with actual status
                    const welcomeMsg = document.querySelector('.bot-message .message-bubble');
                    if (status.agent_available) {{
                        welcomeMsg.innerHTML = `
                            <strong><i class="fas fa-robot me-1"></i>✅ AI Agent: ACTIVE</strong><br>
                            Hello! I'm powered by GPT-4o with Microsoft AutoGen framework.<br><br>
                            <strong>🤖 Agent Capabilities:</strong><br>
                            ✅ Find tracts with highest/lowest values<br>
                            ✅ Look up specific census tracts<br>
                            ✅ Calculate averages and statistics<br>
                            ✅ Show historical trends (2018-2023)<br>
                            ✅ Multi-agent collaboration<br><br>
                            <strong>💡 Try asking:</strong><br>
                            • "Show me tracts with highest broadband access in 2023"<br>
                            • "What is the average poverty rate?"<br>
                            • "Tell me about tract 24510010100"
                        `;
                    }} else {{
                        welcomeMsg.innerHTML = `
                            <strong><i class="fas fa-exclamation-triangle me-1"></i>⚠️ Agent: FALLBACK MODE</strong><br>
                            I'm using rule-based responses (AI agent not available).<br><br>
                            <strong>Reason:</strong> ${{status.agent_init_error || 'Unknown'}}<br><br>
                            <strong>I can still help with:</strong><br>
                            ✅ Find tracts with highest/lowest values<br>
                            ✅ Look up specific census tracts<br>
                            ✅ Calculate averages and statistics<br>
                            ✅ Show trends (2018-2023)<br><br>
                            <small>💡 To enable AI agent:<br>
                            1. Set OPENAI_API_KEY in .env file<br>
                            2. Restart server: <code>python3 Baltimore_MetricsWithMap.py --server</code></small>
                        `;
                    }}
                }}
            }} catch (error) {{
                console.log('Server not reachable:', error);
            }}
        }}
        
        // Check status when page loads
        window.addEventListener('DOMContentLoaded', checkServerStatus);
        
        function toggleChatSidebar() {{
            const sidebar = document.getElementById('chatSidebar');
            const toggleBtn = document.getElementById('toggleChat');
            sidebar.classList.toggle('collapsed');
            
            // Update icon
            const icon = toggleBtn.querySelector('i');
            if (sidebar.classList.contains('collapsed')) {{
                icon.className = 'fas fa-comments';
            }} else {{
                icon.className = 'fas fa-times';
            }}
        }}
        
        function addMessage(text, isUser) {{
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${{isUser ? 'user-message' : 'bot-message'}}`;
            
            const bubble = document.createElement('div');
            bubble.className = 'message-bubble';
            bubble.innerHTML = text;
            
            messageDiv.appendChild(bubble);
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }}
        
        function sendQuickQuestion(question) {{
            document.getElementById('userInput').value = question;
            sendMessage();
        }}
        
        async function sendMessage() {{
            const userInput = document.getElementById('userInput');
            const message = userInput.value.trim();
            
            if (!message) return;
            
            // Add user message
            addMessage(message, true);
            chatHistory.push({{type: 'human', content: message}});
            
            // Clear input and show loading
            userInput.value = '';
            loading.style.display = 'block';
            
            // Call the REAL Python backend (Flask server with agent)
            try {{
                const response = await fetch('http://localhost:5001/chat', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ message: message }})
                }});
                
                if (!response.ok) {{
                    throw new Error(`Server error: ${{response.status}}`);
                }}
                
                const data = await response.json();
                loading.style.display = 'none';
                
                // Show if using agent with appropriate styling
                let responseText = data.response;
                let prefix = '';
                
                if (data.using_agent) {{
                    prefix = '<div style="background: #e8f5e9; padding: 5px 10px; border-radius: 5px; margin-bottom: 8px;">' +
                            '<span style="color: #2e7d32;"><i class="fas fa-robot"></i> <strong>✅ AI Agent Response</strong></span>' +
                            '</div>';
                }} else {{
                    prefix = '<div style="background: #fff3e0; padding: 5px 10px; border-radius: 5px; margin-bottom: 8px;">' +
                            '<span style="color: #e65100;"><i class="fas fa-cog"></i> <strong>⚙️ Fallback Mode</strong></span>';
                    if (data.agent_init_error) {{
                        prefix += '<br><small style="color: #666;">(' + data.agent_init_error + ')</small>';
                    }}
                    prefix += '</div>';
                }}
                
                addMessage(prefix + responseText, false);
                chatHistory.push({{type: 'ai', content: data.response}});
                
            }} catch (error) {{
                loading.style.display = 'none';
                console.error('Error:', error);
                
                // Fallback error message
                const errorMsg = `<strong>⚠️ Cannot connect to chatbot server</strong><br><br>` +
                                `Please start the server:<br>` +
                                `<code>python3 Baltimore_MetricsWithMap.py --server</code><br><br>` +
                                `Then refresh this page.<br><br>` +
                                `<small>Error: ${{error.message}}</small>`;
                addMessage(errorMsg, false);
            }}
        }}
    </script>
</body>
</html>"""
    
    # Write the integrated dashboard
    with open(OUT_HTML, 'w', encoding='utf-8') as f:
        f.write(integrated_html)
    
    print(f"\nWrote integrated dashboard: {OUT_HTML}")
    print("✅ Dashboard now includes embedded AI chatbot!")
    print("Open it in your browser for the complete experience.")


def create_enhanced_dashboard_html(geo_avail_input, df_input, geo_input, dashboard_fig_input):
    """
    DEPRECATED: This function is no longer used.
    Use create_integrated_dashboard() instead.
    """
    print("[warn] create_enhanced_dashboard_html is deprecated. Use create_integrated_dashboard instead.")
    return


def main(args=None):
    """
    Main function to generate the multi-year dashboard or run the server.
    
    Args:
        args: Parsed command-line arguments (optional)
    """
    global chatbot_instance
    
    # If running in server mode, start the Flask server
    if args and args.server:
        print("[info] Starting server mode...")
        if not FLASK_AVAILABLE:
            print("[error] Flask is not available. Install with: pip install flask flask-cors")
            return
        run_server(host=args.host, port=args.port, debug=args.debug)
        return
    
    # Dashboard generation mode (default)
    print("[info] Starting dashboard generation mode...")
    print("[info] Loaded environment variables from .env file")
    
    # Check AutoGen availability
    if not AUTOGEN_AVAILABLE:
        print("[info] AutoGen not available - chatbot features will be limited")
    
    # Load geographic data
    geo_avail = ensure_geo()
    print(f"[info] Using centroids derived from the {geo_avail}.")
    
    # Create the combined metrics DataFrame for all years
    df = combine_metrics(YEARS)
    
    # Get geographic centroids
    if geo_avail == 'SHP zip':
        geo = centroids_from_shapefile(SHP_ZIP_PATH)
    else:
        geo = centroids_from_gazetteer(GAZ_TXT_PATH)
    
    # Build the multi-year figure
    fig = build_figure(df, geo)
    
    # Write the main dashboard HTML
    fig.write_html(
        OUT_HTML,
        include_plotlyjs='cdn',
        config={'displayModeBar': True, 'responsive': True}
    )
    print(f"Wrote main dashboard: {OUT_HTML}")
    
    # Create the integrated dashboard with chatbot
    try:
        create_integrated_dashboard(fig, df, geo)
    except Exception as e:
        print(f"[warn] Could not create integrated dashboard: {e}")
    
    print("\n" + "="*60)
    print("✅ Dashboard generation complete!")
    print("="*60)
    print("\nTo use the chatbot:")
    print("1. Run the server: python Baltimore_MetricsWithMap.py --server")
    print("2. Open the dashboard in your browser")
    print("="*60 + "\n")
    
    return


if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Baltimore Health Dashboard - Generate visualizations or run chatbot server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dashboard (default):
  python Baltimore_MetricsWithMap.py
  
  # Run chatbot server:
  python Baltimore_MetricsWithMap.py --server
  
  # Run server on custom port:
  python Baltimore_MetricsWithMap.py --server --port 8080
        """
    )
    
    parser.add_argument(
        '--server',
        action='store_true',
        help='Run in server mode for chatbot functionality (requires Flask)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host address for server mode (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5001,
        help='Port for server mode (default: 5001)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        default=True,
        help='Run server in debug mode (default: True)'
    )
    
    args = parser.parse_args()
    main(args)





# TODO: Operational data and transacational data. How to capture the data, assumptive
# Give a narrow window, be comprehensive. More types of data. e.g., what does it look like to have different fields to scale