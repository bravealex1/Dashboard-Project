
"""
download_data_only.py

Utility script to *download* ACS subject table variables for Baltimore City (state 24, county 510)
for multiple years, saving JSON files under data/raw/. This lets you fetch data first, then build
the interactive map by running dashboard_multi_year.py.

Usage:
  python download_data_only.py
"""

import os, json, requests

STATE_FIPS = "24"
COUNTY_FIPS = "510"

# Years to fetch — edit this list as needed
YEARS = [2018, 2020, 2022, 2023]

# Variables to fetch (var_code: human label)
VARS = {
    "S2801_C02_017E": "Broadband connection (%)",
    "S1701_C03_002E": "Children in poverty (%)",
    "S1501_C02_014E": "High school completion, 25+ (%)",
}

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

def url(year, var_code):
    return (f"https://api.census.gov/data/{year}/acs/acs5/subject"
            f"?get=NAME,{var_code}&for=tract:*&in=state:{STATE_FIPS}+county:{COUNTY_FIPS}")

def main():
    for y in YEARS:
        for var_code in VARS.keys():
            out = os.path.join(DATA_DIR, f"baltimore_{var_code}_{y}.json")
            if os.path.exists(out):
                print(f"[skip] {os.path.basename(out)} exists")
                continue
            u = url(y, var_code)
            print(f"GET {u}")
            r = requests.get(u, timeout=60)
            r.raise_for_status()
            with open(out, "wb") as f:
                f.write(r.content)
            print(f"[ok]  {out}")

if __name__ == "__main__":
    main()
