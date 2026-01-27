# Baltimore Economic & Health Intelligence Dashboards

This project includes two connected dashboards for Baltimore City:

- **Health Metrics Dashboard**: Census-tract health indicators (broadband, child poverty, education) mapped across multiple years.
- **Economic Intelligence Dashboard**: Real-time labor, inflation, and economic indicators from BLS and FRED.

Both dashboards can run locally. AI chat is optional and works only if you add an OpenAI API key.

---

## Requirements

- Python 3.9+ recommended
- Internet access (for API data downloads)

---

## Quick Start (Health Dashboard)

1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Generate the dashboard HTML:
```bash
python Baltimore_MetricsWithMap.py
```

3) Open the HTML in your browser:
```bash
open output/dashboard_multi_year.html
```
(On Windows, double-click the file or use `start output/dashboard_multi_year.html`.  
On Linux, use `xdg-open output/dashboard_multi_year.html`.)

---

## Optional: View in Streamlit

This is a simple Streamlit wrapper around the generated HTML.

```bash
streamlit run dashboard_multi_year_streamlit.py
```

If the HTML path is different on your machine, change it in the sidebar input.

---

## Optional: Chatbot Server (Health Dashboard)

The health dashboard includes a chatbot panel that talks to a local Flask server.

1) Copy the env template and add your key:
```bash
cp env_config.txt .env
```
Add your key to `.env`:
```
OPENAI_API_KEY=your_key_here
```

2) Start the server:
```bash
python Baltimore_MetricsWithMap.py --server
```

3) Open `output/dashboard_multi_year.html` and use the chat panel.

Notes:
- If you skip the API key, the chatbot still runs in fallback mode (basic responses).
- The server defaults to `http://localhost:5001`.

---

## Economic Dashboard (BLS + FRED)

This dashboard is Streamlit-based and needs API keys.

1) Add these to your `.env` file:
```
BLS_API_KEY=your_key_here
FRED_API_KEY=your_key_here
```
Optional (for AI features):
```
OPENAI_API_KEY=your_key_here
```

2) Run the app:
```bash
streamlit run BLS_and_FRED_Data_Interface.py
```

---

## Optional: Download Data Only

If you want to prefetch ACS data (used by the health dashboard):

```bash
python download_data_only.py
```

---

## Project Structure (Key Files)

- `Baltimore_MetricsWithMap.py`: Builds the health dashboard HTML and optional chatbot server.
- `dashboard_multi_year_streamlit.py`: Streamlit wrapper for the HTML dashboard.
- `BLS_and_FRED_Data_Interface.py`: Economic dashboard (Streamlit).
- `download_data_only.py`: Fetches ACS data only (no dashboard).
- `data/`: Raw and geographic data downloads.
- `output/`: Generated dashboard HTML.

---

## Notes

- If `geopandas` is installed, the map uses tract polygons; otherwise it falls back to centroids.
- The health dashboard downloads Census shapefiles and Gazetteer files automatically if missing.
- All AI features are optional. The dashboards still work without them.
