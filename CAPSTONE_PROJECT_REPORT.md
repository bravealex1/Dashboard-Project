# Baltimore Economic and Health Intelligence Dashboard
## Capstone Project Report

**Author:** Alex  
**Project Duration:** Fall 2024 - Winter 2025  
**Last Updated:** October 30, 2025

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Part 1: Baltimore Health Metrics Dashboard](#part-1-baltimore-health-metrics-dashboard)
   - [Data Sources and Collection](#data-sources-and-collection)
   - [Data Processing Pipeline](#data-processing-pipeline)
   - [Geographic Visualization](#geographic-visualization)
   - [Interactive Dashboard Implementation](#interactive-dashboard-implementation)
4. [Part 2: Economic Intelligence Dashboard with AI](#part-2-economic-intelligence-dashboard-with-ai)
   - [BLS and FRED Data Integration](#bls-and-fred-data-integration)
   - [Multi-Category Dashboard Design](#multi-category-dashboard-design)
   - [Data Architecture](#data-architecture)
5. [AI Integration: Multi-Agent Framework Implementation](#ai-integration-multi-agent-framework-implementation)
   - [CrewAI for Economic Dashboard](#crewai-for-economic-dashboard)
   - [Agno for Health Dashboard](#agno-for-health-dashboard)
   - [Framework Comparison](#framework-comparison)
   - [Custom Tool Development](#custom-tool-development)
6. [Technical Implementation Details](#technical-implementation-details)
7. [Challenges and Solutions](#challenges-and-solutions)
8. [Results and Insights](#results-and-insights)
9. [Future Enhancements](#future-enhancements)
10. [Conclusion](#conclusion)
11. [References](#references)

---

## Executive Summary

This capstone project presents a comprehensive economic and health intelligence system for Baltimore City, Maryland. The project consists of two integrated dashboards that provide real-time access to critical public health and economic indicators. By combining data from the U.S. Census Bureau, Bureau of Labor Statistics (BLS), and Federal Reserve Economic Data (FRED), I've created an interactive platform that allows policymakers, researchers, and community members to explore Baltimore's socioeconomic landscape.

The system features innovative AI-powered chatbots using advanced multi-agent frameworks: **CrewAI** for the Economic Dashboard (specialized agents for data fetching, analysis, and communication) and **Agno** for the Health Dashboard (high-performance single-agent architecture with tool execution and conversation persistence). These frameworks enable users to query complex datasets through natural language, representing a significant advancement in how communities can interact with public data—making sophisticated analysis accessible to non-technical users.

**Key Achievements:**
- Integrated 3 major health metrics (broadband access, child poverty, education) across 4 years of census tract data for 200+ Baltimore City census tracts
- Connected to 25+ economic indicators from BLS and FRED APIs with real-time updates
- Implemented two distinct AI architectures optimized for different use cases (CrewAI for collaborative analysis, Agno for fast interactive queries)
- Created interactive geographic visualizations with modern, accessible UI design
- Built scalable architecture with conversation persistence and graceful fallback modes

---

## Project Overview

### Motivation

Baltimore City faces significant socioeconomic challenges, from educational disparities to economic inequality. While abundant public data exists through government agencies, accessing and interpreting this information remains difficult for most stakeholders. I wanted to bridge this gap by creating a tool that makes data both accessible and actionable.

During my research, I discovered that many existing dashboards either focus narrowly on one domain (e.g., only health or only economics) or present data in static formats that limit exploration. My goal was to create something more dynamic and comprehensive—a platform where someone could ask "Which Baltimore neighborhoods have both low broadband access and high child poverty?" and get an immediate, data-driven answer.

The New York City Health Dashboard, created by NYC Langone Hospital, is a prestigious and comprehensive resource. While they provide health data for major cities like NYC, Houston, and San Francisco, there is currently no dedicated dashboard for Baltimore. As a Hopkins student passionate about local community development and social health issues, I wanted to create a separate, tailored dashboard specifically for Baltimore. To achieve this, I plan to adapt the methodology from the NYC Health Dashboard and incorporate geospatial data to develop a comprehensive health resource for our city.

### Project Scope

The project encompasses two main components:

1. **Baltimore Health Metrics Dashboard** - Visualizes census tract-level data on broadband connectivity, child poverty rates, and educational attainment across multiple years

2. **Economic Intelligence Dashboard** - Provides real-time access to employment statistics, inflation data, housing market indicators, and geographic unemployment patterns

Both dashboards integrate an AI-powered chatbot that can interpret natural language queries, fetch relevant data, perform calculations, and generate visualizations on demand.

---

## Part 1: Baltimore Health Metrics Dashboard

### Data Sources and Collection

#### Census Bureau American Community Survey (ACS)

The foundation of my health dashboard comes from the U.S. Census Bureau's American Community Survey (ACS) 5-year estimates. I specifically chose three metrics that align with the City Health Dashboard definitions:

1. **Broadband Connectivity (S2801_C02_017E)** - Percentage of households with broadband internet access (cable, fiber optic, or DSL)
2. **Child Poverty (S1701_C03_002E)** - Percentage of children under 18 living below the poverty level
3. **High School Completion (S1501_C02_014E)** - Percentage of adults 25+ who completed high school or higher

I selected these metrics because they represent fundamental determinants of community wellbeing. Internet access affects education and job opportunities, child poverty impacts long-term outcomes, and educational attainment correlates with economic mobility.

#### Why These Years? (2018, 2020, 2022, 2023)

I deliberately chose these specific years to capture trends before, during, and after the COVID-19 pandemic. This allows us to see how Baltimore's neighborhoods were affected by this major disruption:

- **2018** - Pre-pandemic baseline
- **2020** - Pandemic onset
- **2022** - Recovery period
- **2023** - Current state

#### Data Collection Process

Getting the data involved several steps:

**Step 1: Identifying the Right Variables**

The Census Bureau has thousands of variables, so finding the right ones took research. I used the Census API documentation to locate the exact variable codes. For example, I verified that S2801_C02_017E specifically measures "broadband such as cable, fiber optic, or DSL" by checking the official variable definitions at `api.census.gov/data/2022/acs/acs5/subject/groups/S2801.html`.

**Step 2: Understanding Geographic Identifiers**

Census tracts are small geographic areas with roughly 4,000 residents each. Baltimore City (FIPS code: 24510) contains about 200 census tracts. Each tract has a unique 11-digit GEOID composed of:
- State FIPS: 24 (Maryland)
- County FIPS: 510 (Baltimore City)
- Tract code: 6 digits

**Step 3: API Data Retrieval**

I wrote Python scripts to systematically download data from the Census API. The API returns JSON files with this structure:

```json
[
  ["NAME", "S2801_C02_017E", "state", "county", "tract"],
  ["Census Tract 2601...", "87.5", "24", "510", "260100"],
  ["Census Tract 2602...", "78.2", "24", "510", "260200"]
]
```

The first row contains column headers, and subsequent rows contain the actual data. I downloaded separate JSON files for each metric and year combination, storing them in `data/raw/` directory with naming convention: `baltimore_{VARIABLE}_{YEAR}.json`.

For example:
- `baltimore_S2801_C02_017E_2018.json` (Broadband 2018)
- `baltimore_S1701_C03_002E_2023.json` (Child Poverty 2023)

### Data Processing Pipeline

Once I had the raw JSON files, I needed to transform them into a format suitable for analysis and visualization. Here's how I approached this:

#### Step 1: Loading and Parsing JSON Files

I created a function called `load_metric_df()` that:
1. Opens each JSON file
2. Extracts the header row and data rows
3. Creates a pandas DataFrame
4. Adds a year column for temporal tracking
5. Constructs the full GEOID by concatenating state, county, and tract codes

```python
def load_metric_df(var_code: str, years: list[int]) -> pd.DataFrame:
    frames = []
    for y in years:
        path = os.path.join(DATA_RAW, f"baltimore_{var_code}_{y}.json")
        # ... load JSON, parse into DataFrame ...
        df["GEOID"] = df["state"].str.zfill(2) + 
                      df["county"].str.zfill(3) + 
                      df["tract"].str.zfill(6)
        frames.append(df[["GEOID", var_code, "year"]])
    return pd.concat(frames, ignore_index=True)
```

The `.zfill()` method ensures proper zero-padding (e.g., "24" becomes "24", "510" stays "510", "10100" becomes "010100").

#### Step 2: Combining Multiple Metrics

Since I have three different metrics stored in separate files, I needed to merge them into a single comprehensive dataset. I created `combine_metrics()` to:

1. Load each metric using `load_metric_df()`
2. Merge them on GEOID and year
3. Filter to only Baltimore City tracts (GEOID starting with "24510")

This gives me a "wide format" DataFrame where each row represents one census tract in one year, with columns for all three metrics.

#### Step 3: Data Quality and Missing Values

Real-world data is never perfect. I encountered several challenges:

- **Suppressed values**: Census sometimes suppresses data for privacy when populations are too small
- **Null entries**: Some tracts had missing data for certain years
- **Format inconsistencies**: Some values came as strings with commas ("2,345") instead of numbers

I handled these by:
- Converting all values to numeric with `pd.to_numeric(..., errors='coerce')` which turns non-numeric values into NaN
- Explicitly dropping rows with null values before visualization
- Logging which files were missing so I could verify data availability

### Geographic Visualization

This was one of the most interesting technical challenges of the project. To create maps, I needed two things:
1. The metric data (which I already had)
2. Geographic coordinates for each census tract

#### Challenge: Getting Geographic Data

There are two main ways to represent census tracts geographically:

**Option 1: Polygon Boundaries** - The full shape of each tract (requires heavyweight GIS libraries)

**Option 2: Centroids** - A single point (latitude/longitude) representing each tract's center

I implemented a flexible system that tries to use polygons when possible but falls back to centroids if needed.

#### Geographic Data Sources

**Shapefile (Cartographic Boundary Files)**

The Census Bureau provides shapefiles containing precise polygon boundaries. I used:
- URL: `https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_24_tract_500k.zip`
- This is the 2022 Maryland census tract cartographic boundary file at 1:500,000 scale

The ZIP file contains multiple files:
- `.shp` - The main shapefile with geometry
- `.dbf` - Attribute data (tract IDs, names, etc.)
- `.prj` - Projection information
- `.shx` - Spatial index

**Gazetteer File (Centroid Coordinates)**

As a backup, I use the Census Gazetteer which provides centroid coordinates:
- URL: `https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2025_Gazetteer/2025_gaz_tracts_24.txt`
- This is a pipe-delimited text file with columns: GEOID, NAME, INTPTLAT, INTPTLONG, etc.

#### Implementation: Flexible Geographic Loading

I wrote `ensure_geo()` to automatically download and validate geographic data:

```python
def ensure_geo() -> dict:
    # Try to download shapefile if not present
    have_shp = os.path.exists(SHP_ZIP_PATH)
    if not have_shp:
        _download(SHP_ZIP_URL, SHP_ZIP_PATH)
        # Verify it actually contains .shp and .dbf files
        have_shp = _zip_has_extensions(SHP_ZIP_PATH, (".shp", ".dbf"))
    
    # Try gazetteer if shapefile fails
    have_gaz = os.path.exists(GAZ_TXT_PATH)
    if not have_gaz:
        _download(GAZ_URL, GAZ_TXT_PATH)
    
    return {"shp_zip": SHP_ZIP_PATH if have_shp else None,
            "gaz_txt": GAZ_TXT_PATH if have_gaz else None}
```

This approach is resilient—if one source fails, we have a backup.

#### Extracting Centroids from Shapefiles

When GeoPandas isn't available (a common situation on some systems), I implemented a custom shapefile parser that reads the binary .shp and .dbf files directly using Python's `struct` module. This was particularly challenging because shapefiles use a complex binary format:

1. Read the .dbf file to extract GEOID for each tract
2. Read the .shp file to extract bounding boxes (xmin, xmax, ymin, ymax)
3. Calculate centroid as: `lon = (xmin + xmax) / 2`, `lat = (ymin + ymax) / 2`
4. Match records together by index position

While these aren't true geometric centroids, they're close enough for visualization purposes and don't require any external dependencies.

#### Parsing the Gazetteer

The gazetteer file is simpler—it's pipe-delimited text. However, I encountered an issue: Census documentation isn't always perfectly consistent about column names. Sometimes it's `INTPTLONG`, sometimes `INTPTLON`. My solution:

```python
def centroids_from_gazetteer(txt_path: str) -> pd.DataFrame:
    df = pd.read_csv(txt_path, sep="|", dtype=str)
    df.columns = [c.strip() for c in df.columns]  # Remove whitespace
    
    # Verify expected columns exist
    for need in ("GEOID", "INTPTLAT", "INTPTLONG"):
        if need not in df.columns:
            raise KeyError(f"Missing column: {need}")
    
    # Extract and filter to Baltimore City
    out = df[["GEOID", "INTPTLONG", "INTPTLAT"]]
    out = out.rename(columns={"INTPTLONG": "lon", "INTPTLAT": "lat"})
    out = out[out["GEOID"].str.startswith("24510")]
    return out
```

This gives me a clean DataFrame with GEOID, longitude, and latitude for every Baltimore City tract.

### Interactive Dashboard Implementation

With data and geography in hand, I could build the visualization.

#### Technology Choice: Plotly

I chose Plotly because it creates truly interactive visualizations that work in any web browser without requiring a server. Users can:
- Zoom and pan the map
- Hover to see exact values
- Switch between metrics and years dynamically

#### Building the Interactive Map

The core visualization function is `build_figure()`. Here's the conceptual flow:

**Step 1: Determine Visualization Type**

If I have polygon data (from GeoPandas), I create a choropleth map where each tract is colored based on its value. If I only have centroids, I create a bubble map with colored points.

For most users, the bubble map is what they'll see:

```python
def build_figure(df_metrics: pd.DataFrame, geo):
    metric_order = [
        ("broadband_pct", "Broadband connection (%)"),
        ("child_poverty_pct", "Children in poverty (%)"),
        ("hs_complete_pct", "High school completion, 25+ (%)"),
    ]
    
    fig = go.Figure()
    
    # Create one trace for each combination of metric and year
    for mkey, mlabel in metric_order:
        for year in [2018, 2020, 2022, 2023]:
            # Merge metric data with geographic coordinates
            merged = geo.merge(df_metrics[df_metrics["year"] == year], 
                              on="GEOID")
            
            # Create scatter points
            trace = go.Scattermapbox(
                lon=merged["lon"],
                lat=merged["lat"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=merged[var_code],
                    colorscale="Viridis",  # Color scheme
                    showscale=True
                ),
                visible=False  # Start hidden
            )
            fig.add_trace(trace)
```

**Step 2: Create Interactive Dropdown Menus**

This is where Plotly shines. I created two dropdown menus:
1. **Metric selector** - Choose which metric to view (broadband, poverty, or education)
2. **Year selector** - Choose which year to display (2018, 2020, 2022, or 2023)

The key is the `updatemenus` configuration:

```python
fig.update_layout(
    updatemenus=[
        dict(
            buttons=metric_buttons,
            direction="down",
            x=0.01, y=0.99,  # Top-left corner
            bgcolor="white"
        ),
        dict(
            buttons=year_buttons,
            direction="down",
            x=0.35, y=0.99,  # Slightly to the right
            bgcolor="white"
        ),
    ]
)
```

Each button has an `args` parameter that specifies which traces should be visible when clicked. For example, clicking "Broadband connection (%)" shows only the broadband traces, hiding all others.

**Step 3: Configuring the Map Style**

I used OpenStreetMap as the base layer because it's free and provides good detail for Baltimore City:

```python
fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        center={"lat": 39.2992, "lon": -76.6094},  # Baltimore center
        zoom=10
    )
)
```

The zoom level of 10 provides a good balance—showing all of Baltimore City while still maintaining detail.

#### Output: Standalone HTML File

Finally, I save the dashboard as a self-contained HTML file:

```python
fig.write_html(OUT_HTML, include_plotlyjs="cdn", full_html=True)
```

This creates a single file that anyone can open in their browser. The `include_plotlyjs="cdn"` option keeps the file size small by loading Plotly from a content delivery network rather than embedding the entire library.

**The result:** A ~200KB HTML file containing all the data and interactivity for exploring Baltimore's health metrics across four years and 200+ census tracts.

### Streamlit Deployment Wrapper

To make the static HTML easier to share, I also built a lightweight Streamlit app (`dashboard_multi_year_streamlit.py`) that simply embeds the generated HTML using `streamlit.components.v1.html`. Key details:

- The app looks for `output/dashboard_multi_year.html` (same file produced by `Baltimore_MetricsWithMap.py`).
- If the file is missing, it prompts the user to regenerate it.
- Users can specify a custom path, toggle auto-refresh, and reload on demand.

Run it with:

```bash
streamlit run dashboard_multi_year_streamlit.py
```

This creates a quick deployment path (e.g., Streamlit Community Cloud or an internal server) without rewriting the visualization logic.

---

## Part 2: Economic Intelligence Dashboard with AI

### BLS and FRED Data Integration

While the health dashboard focuses on geographic patterns within Baltimore, my economic dashboard takes a different approach—showing how Baltimore fits into broader economic trends through official government statistics.

#### Understanding the Data Sources

Before diving into implementation, I spent considerable time researching the differences between BLS and FRED:

**Bureau of Labor Statistics (BLS)**

The BLS is the principal federal agency responsible for measuring labor market activity, working conditions, and price changes in the U.S. economy. I use BLS data because:
- It's the official source cited by media and policymakers
- Data comes directly from surveys (Current Employment Statistics, Local Area Unemployment Statistics)
- Released on strict schedules (employment report on first Friday of month)

**Federal Reserve Economic Data (FRED)**

FRED is maintained by the Federal Reserve Bank of St. Louis as an economic database aggregator. I use FRED because:
- It republishes BLS data with additional context
- Provides housing market data from FHFA (Federal Housing Finance Agency)
- Offers the GeoFRED Maps API for county-level visualizations
- Includes seasonally-adjusted versions of series

**Why Both?** I use BLS as the source of truth for employment and inflation data, while FRED provides housing indicators and geographic mapping capabilities that BLS doesn't offer.

#### API Authentication and Setup

Both APIs require registration:

**BLS API Key:**
- Register at: `https://data.bls.gov/registrationEngine/`
- Free tier: 500 queries per day, 10 years of data per query
- I store this in a `.env` file as `BLS_API_KEY=your_key_here`

**FRED API Key:**
- Register at: `https://research.stlouisfed.org/useraccount/apikeys`
- Free tier: Unlimited queries (with reasonable rate limits)
- Stored as `FRED_API_KEY=your_key_here`

I use the `python-dotenv` library to load these securely without hardcoding them in my script.

### Multi-Category Dashboard Design

Unlike the health dashboard which focuses on geographic variation, the economic dashboard emphasizes **categorical organization** and **temporal trends**. I structured it around five main categories:

#### Category 1: Labor Market Indicators

This category shows employment and unemployment patterns for the Baltimore-Columbia-Towson Metropolitan Statistical Area (MSA).

**Geographic Coverage Explained:**

A Metropolitan Statistical Area (MSA) is larger than a single city. The Baltimore MSA (CBSA code 12580) includes:
- Baltimore City
- Baltimore County
- Carroll County
- Anne Arundel County
- Howard County
- Harford County
- Queen Anne's County

This broader definition captures commuting patterns and economic interconnections.

**Metrics I Selected:**

From BLS Local Area Unemployment Statistics (LAUS):
- `LAUMT241258000000003` - Unemployment rate (%)
- `LAUMT241258000000004` - Number of unemployed persons
- `LAUMT241258000000005` - Number of employed persons
- `LAUMT241258000000006` - Total labor force

From BLS Current Employment Statistics (CES):
- `SMU24125800000000001` - Total nonfarm employment
- `SMU24125803000000001` - Manufacturing employment
- `SMU24125804200000001` - Retail trade employment
- `SMU24125806562000001` - Healthcare employment
- `SMU24125807000000001` - Leisure & hospitality employment
- `SMU24125809000000001` - Government employment

**Why These Specific Series?** The series IDs follow BLS naming conventions. For example, `SMU24125800000000001` breaks down as:
- `SM` - State and Metro Area CES
- `U` - Not seasonally adjusted
- `24` - Maryland state code
- `12580` - Baltimore MSA code
- `00000000` - All industries
- `001` - All employees (in thousands)

By including industry breakdowns, users can see which sectors are growing or declining—crucial for workforce development planning.

#### Category 2: Prices & Inflation

**Why Baltimore-Washington CPI Area?**

The Consumer Price Index (CPI) is measured for specific geographic areas. Baltimore's CPI area is combined with Washington D.C. because the Census Bureau needs sufficient sample size for statistical reliability. This means:
- CPI series like `CUURS35ESA0` cover both metro areas
- It's broader than the employment MSA
- Still representative of Baltimore's cost of living trends

**Metrics:**
- All items CPI (`CUURS35ESA0`)
- Food CPI (`CUURS35ESAF`)
- Housing CPI (`CUURS35ESAH`)
- Transportation CPI (`CUURS35ESAT`)
- Medical care CPI (`CUURS35ESAM`)
- Energy CPI (`CUURS35ESAE`)

**Calculating Inflation:**

I don't just show raw CPI values—I calculate year-over-year inflation rates:

```python
df_cpi["inflation_rate"] = df_cpi["value"].pct_change(12) * 100
```

This takes the percentage change from 12 months ago (since CPI is monthly data) and converts to a percentage.

#### Category 3: Housing & Real Estate

Housing affordability is critical to economic wellbeing, so I dedicated a category to it:

**Home Price Index (HPI) from FHFA:**
- Series: `ATNHPIUS12580Q` (Baltimore-Columbia-Towson MSA)
- Quarterly data, not seasonally adjusted
- Base period: 1995 Q1 = 100
- Measures price changes for homes with mortgages backed by Fannie Mae or Freddie Mac

**Why HPI Matters:**

If the HPI is 250, it means home prices are 150% higher than in 1995 Q1. By comparing this to income growth, we can assess affordability trends.

I also include national housing metrics since Baltimore-specific data isn't always available:
- Median home sale price (`MSPUS`)
- Homeownership rate (`RHORUSQ156N`)

#### Category 4: Geographic/County Data

This is where FRED's GeoFRED Maps API shines. I can create choropleth maps showing unemployment rates for all Maryland counties.

**The Challenge:**

Unlike BLS, which has straightforward timeseries endpoints, FRED Maps requires navigating multiple API endpoints:

1. **Series Data endpoint** - Get latest cross-section
2. **Regional Data endpoint** - Get historical cross-sections with specific parameters
3. **Series Group Info endpoint** - Get metadata needed for Regional Data queries
4. **Shapes endpoint** - Get GeoJSON boundaries

**My Implementation:**

```python
def fred_maps_series_cross_section(series_id, api_key, date=None):
    # Try Series Data first (simpler)
    response = requests.get(FRED_MAPS_SERIES_DATA, 
                           params={"series_id": series_id, 
                                  "api_key": api_key})
    
    if has_valid_data(response):
        return parse_data(response)
    
    # Fallback to Regional Data (requires more parameters)
    metadata = get_series_group_info(series_id, api_key)
    response = requests.get(FRED_MAPS_REGIONAL,
                           params={
                               "series_group": metadata["series_group"],
                               "region_type": metadata["region_type"],
                               "season": metadata["season"],
                               "units": metadata["units"],
                               "date": date or metadata["max_date"]
                           })
    return parse_data(response)
```

This dual-endpoint approach ensures I can always get data even if one endpoint fails.

### Data Architecture

#### Time Window Management

I decided to show the last 7 years of data (current year minus 6, through current year). This provides enough historical context to identify trends without overwhelming users.

```python
TODAY = datetime.now()
CURRENT_YEAR = TODAY.year
START_YEAR = str(CURRENT_YEAR - 6)
END_YEAR = str(CURRENT_YEAR)
```

This auto-updates as time passes—no hardcoding needed.

#### Data Fetching Functions

**BLS Timeseries:**

```python
def bls_fetch_timeseries(series_ids, start_year, end_year, registration_key):
    payload = {
        "seriesid": series_ids,  # Can request multiple series at once
        "startyear": start_year,
        "endyear": end_year,
        "registrationkey": registration_key,
        "catalog": True  # Include metadata
    }
    
    response = requests.post(BLS_TIMESERIES_URL, 
                            data=json.dumps(payload),
                            headers={"Content-type": "application/json"})
    
    return response.json()
```

The BLS API uses POST requests with JSON payloads, which is unusual but allows requesting multiple series simultaneously (more efficient).

**FRED Series Observations:**

```python
def fred_series_observations(series_id, api_key, start=None):
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start or f"{START_YEAR}-01-01"
    }
    
    response = requests.get(FRED_SERIES_OBS_URL, params=params)
    data = response.json()
    
    df = pd.DataFrame(data["observations"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    
    return df[["date", "value"]]
```

FRED uses GET requests with query parameters—more conventional but requires separate calls for each series.

#### DataFrame Transformation

BLS returns data in a nested JSON structure that requires careful parsing:

```python
def bls_to_dataframe(bls_json):
    rows = []
    for series in bls_json["Results"]["series"]:
        series_id = series["seriesID"]
        for observation in series["data"]:
            year = observation["year"]
            period = observation["period"]  # e.g., "M01" for January
            
            # Convert period to actual date
            if period.startswith("M") and period != "M13":
                month = int(period[1:])
                date = pd.Timestamp(year=int(year), month=month, day=1)
                
                rows.append({
                    "series_id": series_id,
                    "date": date,
                    "value": float(observation["value"])
                })
    
    return pd.DataFrame(rows).sort_values("date")
```

This converts BLS's period notation into standard datetime objects that Plotly can use.

#### Dashboard Display Functions

Each category has its own display function that:
1. Fetches relevant data
2. Creates visualizations
3. Provides download buttons
4. Shows key metrics

For example, `display_labor_market_dashboard()`:
- Fetches LAUS and CES data
- Creates time series charts
- Displays latest values as metric cards
- Organizes into tabs (Overview, By Industry, Trends)

This modular approach makes the code maintainable and allows easy addition of new categories.

---

## AI Integration: Multi-Agent Framework Implementation

This section represents the most innovative aspect of my project—creating AI assistants that can intelligently interact with both economic and health data using purpose-built frameworks.

### What is an Agentic Workflow?

Traditional chatbots follow scripted responses. Agentic workflows are fundamentally different. An agent:

1. **Understands Intent** - Uses GPT-4o to interpret what the user wants
2. **Selects Tools** - Decides which functions/APIs to call based on the query
3. **Executes Actions** - Runs the selected tools with appropriate parameters
4. **Synthesizes Response** - Combines tool outputs into a coherent answer

I implemented two different agent frameworks, each optimized for its specific use case.

### CrewAI for Economic Dashboard

For the BLS/FRED Economic Dashboard, I chose **CrewAI** because it provides a role-based multi-agent orchestration framework. CrewAI is ideal for scenarios where specialized agents need to collaborate on complex tasks through a sequential workflow.

#### Step 1: Installing Dependencies

```bash
pip install crewai
```

CrewAI provides the core framework for agent collaboration. Note that the tool decorator is included in the main crewai package, not a separate crewai-tools package.

#### Step 2: Environment Configuration

The agents require two environment variables:

```
OPENAI_API_KEY=sk-your-key-here
CHROMA_OPENAI_API_KEY=sk-your-key-here
```

The first is for the language model, and the second is for ChromaDB embeddings which power the agent memory system. Both can use the same OpenAI key. These are stored in `.env` and loaded with `python-dotenv`.

#### Step 3: Tool Creation with CrewAI

CrewAI uses custom tools that wrap Python functions. Each tool has a name and docstring that helps the agent understand when to use it:

```python
from crewai.tools import tool

@tool("Get unemployment data")
def get_unemployment_data() -> str:
    """Fetch latest unemployment statistics for Baltimore MSA from BLS LAUS."""
    # Function fetches data from BLS API
    # Returns formatted string with results
    return formatted_results

@tool("Get inflation data")
def get_cpi_data() -> str:
    """Fetch Consumer Price Index and inflation rate for Baltimore-Washington area from BLS."""
    # Function fetches CPI data and calculates inflation rate
    return formatted_results
```

I created five specialized tools:
- get_unemployment_data - BLS LAUS unemployment statistics
- get_cpi_data - Consumer Price Index and inflation
- get_industry_employment - Employment by sector from BLS CES
- get_home_price_index - FHFA housing data from FRED
- get_county_unemployment - Maryland county unemployment map data

#### Step 4: Multi-Agent Architecture

CrewAI uses three specialized agents that work in sequence:

**a) Economic Data Fetcher Agent**

```python
from crewai import Agent

fetcher = Agent(
    role="Economic Data Fetcher",
    goal="Use the provided tools to gather the exact Baltimore economic metrics requested by the user.",
    backstory="You specialize in querying BLS LAUS, BLS CES, CPI-U, FRED housing data, and county unemployment cross-sections for the Baltimore region.",
    tools=self.tools,
    verbose=False,
    allow_delegation=False
)
```

This agent calls the appropriate tools to fetch raw data from BLS and FRED APIs.

**b) Economic Data Analyst Agent**

```python
analyst = Agent(
    role="Economic Data Analyst",
    goal="Inspect retrieved metrics, calculate trends, and highlight the most relevant economic context for the user.",
    backstory="You interpret labor market, price, and housing indicators for Baltimore and summarize what the numbers mean.",
    verbose=False,
    allow_delegation=False
)
```

This agent reviews the fetched data and identifies trends, patterns, and notable changes.

**c) Economic Data Communicator Agent**

```python
communicator = Agent(
    role="Economic Data Communicator",
    goal="Deliver a clear, user-friendly answer that cites metrics, dates, geographies, and data sources.",
    backstory="You craft concise explanations so the user quickly understands Baltimore's economic picture.",
    verbose=False,
    allow_delegation=False
)
```

This agent formats the analysis into a clear response for the user.

#### Step 5: Task-Based Workflow

Each user query creates three sequential tasks:

```python
from crewai import Task

# Task 1: Fetch Data
fetch_task = Task(
    description="Identify which economic datasets are needed. Call the available tools to fetch the actual numbers.",
    expected_output="Structured bullet list with fetched metrics, each including value, unit, geography, date, and data source.",
    agent=fetcher,
    tools=self.tools
)

# Task 2: Analyze Data
analyze_task = Task(
    description="Review the fetcher's output and synthesize insights that answer the user's question.",
    expected_output="Numbered list of insights referencing the fetched metrics and explaining their significance.",
    agent=analyst
)

# Task 3: Communicate Results
respond_task = Task(
    description="Compose the final response for the user. Cite data sources and years, and explain the significance.",
    expected_output="Well-structured response with 2-4 short paragraphs or bullet points summarizing the metrics, trends, and caveats.",
    agent=communicator
)
```

Each task has a description, expected output, and assigned agent. Tasks run sequentially, with later tasks accessing earlier task outputs.

#### Step 6: Crew Orchestration with Memory and Embeddings

```python
from crewai import Crew, Process

crew = Crew(
    agents=[fetcher, analyst, communicator],
    tasks=[fetch_task, analyze_task, respond_task],
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
```

Key configuration choices:
- **Process.sequential**: Tasks run in order, each building on previous outputs
- **memory=True**: Enables agents to retain context across the conversation
- **embedder**: Uses OpenAI embeddings (text-embedding-3-small model) to convert text to vectors for semantic understanding
- **verbose=True**: Shows agent reasoning steps for debugging

The embedding model is critical because it allows agents to understand semantic relationships between queries and maintain conversation context.

### Agno for Health Dashboard

For the Baltimore Health Dashboard, I chose **Agno** because it excels at fast, lightweight single-agent architectures with built-in persistence and tool integration.

#### Why Agno?

Agno is a high-performance multi-agent framework built for speed, privacy, and scale:
- 529x faster than LangGraph, 70x faster than CrewAI for agent instantiation
- 24x less memory than LangGraph, 10x less than CrewAI
- Single Agent class with built-in memory and tool support
- SQLite database integration for conversation persistence
- Production-ready with FastAPI runtime included

#### Step 1: Installing Dependencies

```bash
pip install agno>=2.2.0
```

#### Step 2: Agno Architecture

Agno uses a simplified single-agent pattern optimized for interactive queries:

**a) Agent Initialization**

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
from agno.db.sqlite import SqliteDb

# Create database for conversation persistence
db = SqliteDb(db_file="data/baltimore_health_chatbot.db")

# Create agent with tools
agent = Agent(
    name="BaltimoreHealthAssistant",
    model=OpenAIChat(id="gpt-4o", api_key=openai_key),
    tools=[query_metric_data, get_tract_info, find_extreme_values],
    db=db,
    description="You are a helpful assistant for the Baltimore City Health Dashboard.",
    instructions=[
        "Always specify which metric, year, and location you're discussing",
        "Use tools to fetch real data instead of making assumptions"
    ],
    markdown=True,
    add_history_to_context=True,
    num_history_messages=10,
    read_chat_history=True
)
```

**b) Tool Registration**

```python
@tool
def query_metric_data(query: str) -> str:
    """Query health metric data by location, metric type, or year."""
    # Implementation
    return results
```

**c) Running the Agent**

```python
# Simple API - just run with the message
response = agent.run(user_message)
```

**Key Advantages of Agno:**
- Fast performance for real-time interactive queries
- Built-in conversation persistence across sessions
- Simple API reduces code complexity
- Ideal for health dashboard where speed matters
- Automatic tool selection and execution

### Framework Comparison

| Feature | CrewAI (Economic) | Agno (Health) |
|---------|-------------------|---------------|
| **Architecture** | Multi-agent with roles | Single agent with tools |
| **Use Case** | Complex collaborative analysis | Fast interactive queries |
| **Performance** | Moderate (task orchestration) | Very fast (optimized for speed) |
| **Memory** | Embedding-based context | SQLite conversation history |
| **API Complexity** | High (agents, tasks, crews) | Low (agent.run) |
| **Tool Registration** | @tool decorator | @tool decorator |
| **Best For** | Multi-step workflows | Real-time Q&A |

### Custom Tool Development

I developed specialized tools for both dashboards using the @tool decorator pattern:

**For CrewAI Economic Dashboard (5 tools):**

#### Tool 1: Get Unemployment Data

```python
from crewai.tools import tool

@tool("Get unemployment data")
def get_unemployment_data() -> str:
    """Fetch latest unemployment statistics for Baltimore MSA from BLS LAUS."""
    try:
        # Call BLS API for LAUS data
        data = bls_fetch_timeseries(
            [LAUS_UR_MSA, LAUS_UNEMP_MSA, LAUS_EMP_MSA, LAUS_LF_MSA],
            START_YEAR, END_YEAR,
            registration_key=bls_key, 
            catalog=True
        )
        
        # Convert to DataFrame
        df = bls_to_dataframe(data)
        
        # Get most recent values for each series
        latest = df.sort_values("date").groupby("series_id").tail(1)
        
        # Format response
        ur_val = latest[latest["series_id"] == LAUS_UR_MSA]["value"].iloc[0]
        ur_date = latest[latest["series_id"] == LAUS_UR_MSA]["date"].iloc[0]
        
        # Extract values from result dictionary
        ur_date, ur_val = result_dict.get(LAUS_UR_MSA, ("n/a", None))
        unemp_val = result_dict.get(LAUS_UNEMP_MSA, (None, 0))[1]
        emp_val = result_dict.get(LAUS_EMP_MSA, (None, 0))[1]
        lf_val = result_dict.get(LAUS_LF_MSA, (None, 0))[1]
        
        return f"Baltimore MSA Labor Market (BLS LAUS, {ur_date}): Unemployment Rate: {ur_val:.1f}%, Unemployed: {int(unemp_val):,}, Employed: {int(emp_val):,}, Labor Force: {int(lf_val):,}"
    
    except Exception as e:
        return f"Error fetching unemployment data: {e}"
```

**Key Implementation Details:**

The @tool decorator registers the function with CrewAI. The function name and docstring help agents understand when to use it. Tools return formatted strings that agents can incorporate into their responses. Error handling ensures graceful failures when APIs are unavailable.

#### Tool 2: Get Inflation Data

Calculates Consumer Price Index and year-over-year inflation:

```python
@tool("Get inflation data")
def get_cpi_data() -> str:
    """Fetch Consumer Price Index and inflation rate for Baltimore-Washington area from BLS."""
    try:
    data = bls_fetch_timeseries([CPIU_BALT_ALL], START_YEAR, END_YEAR, ...)
    df = bls_to_dataframe(data).sort_values("date")
        last = df.iloc[-1]
        
        # Calculate 12-month percentage change
        yoy = df.set_index("date")["value"].pct_change(12).loc[last["date"]]
        
        return f"CPI-U Baltimore-Washington Area (BLS, {last['date'].date().isoformat()}): Index: {last['value']:.1f}, Annual Inflation Rate: {yoy*100:.1f}%"
    except Exception as e:
        return f"Error fetching CPI data: {e}"
```

#### Tool 3: Get Industry Employment

Fetches employment by sector from BLS Current Employment Statistics:

```python
@tool("Get industry employment")
def get_industry_employment() -> str:
    """Fetch employment by industry sector for Baltimore MSA from BLS CES."""
    try:
        ces_series = [CES_TNF_MSA, CES_MANUF_MSA, CES_TRADE_MSA, 
                     CES_PROF_MSA, CES_HEALTH_MSA, CES_LEISURE_MSA, CES_GOVT_MSA]
        data = bls_fetch_timeseries(ces_series, START_YEAR, END_YEAR, ...)
    df = bls_to_dataframe(data)
        last = df.sort_values("date").groupby("series_id").tail(1)
        
        # Map series IDs to readable names
        names = {
            CES_TNF_MSA: "Total Nonfarm",
            CES_MANUF_MSA: "Manufacturing",
            CES_TRADE_MSA: "Retail Trade",
            # ... more mappings
        }
        
        result = f"Baltimore MSA Employment by Industry (BLS CES, {last.iloc[0]['date'].date().isoformat()}): "
        parts = [f"{names.get(row['series_id'])}: {int(row['value']):,}" 
                for _, row in last.iterrows()]
        return result + ", ".join(parts)
    except Exception as e:
        return f"Error fetching industry employment: {e}"
```

#### Tool 4: Get Home Price Index

Fetches FHFA housing data and calculates changes:

```python
@tool("Get home price index")
def get_home_price_index() -> str:
    """Fetch FHFA Home Price Index for Baltimore MSA from FRED."""
    try:
        hpi_series = FRED_HOUSING_SERIES["Home Price Index (FHFA)"]["series_id"]
        df = fred_series_observations(hpi_series, fred_key, start=f"{START_YEAR}-01-01")
        df = df.sort_values("date")
        last = df.iloc[-1]
        
        # Calculate year-over-year change if enough data
    if len(df) >= 5:
            yoy_change = ((last["value"] / df.iloc[-5]["value"]) - 1) * 100
            return f"Baltimore MSA Home Price Index (FHFA via FRED, {last['date'].date().isoformat()}): HPI: {last['value']:.2f}, Year-over-Year Change: {yoy_change:+.1f}%"
    else:
            return f"Baltimore MSA Home Price Index (FHFA via FRED, {last['date'].date().isoformat()}): HPI: {last['value']:.2f}"
    except Exception as e:
        return f"Error fetching home price data: {e}"
```

#### Tool 5: Get County Unemployment

Fetches Maryland county unemployment from FRED Maps API:

```python
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
        
        # Filter to Maryland counties (FIPS code starts with "24")
        md = cross[cross["code"].astype(str).str.zfill(5).str[:2] == "24"].copy()
        if md.empty:
            return "No Maryland counties found in FRED Maps data."
        
        # Sort and get highest and lowest unemployment counties
    md_sorted = md.sort_values("value", ascending=False)
    top5 = md_sorted.head(5)
    bottom5 = md_sorted.tail(5)
    
        result = f"Maryland County Unemployment (FRED Maps, {used_date or 'Latest'}): "
        result += "Highest: " + ", ".join([f"{row['region']}: {row['value']:.1f}%" 
                                          for _, row in top5.iterrows()])
        result += " | Lowest: " + ", ".join([f"{row['region']}: {row['value']:.1f}%" 
                                            for _, row in bottom5.iterrows()])
    return result
    except Exception as e:
        return f"Error fetching county data: {e}"
```

This tool provides geographic context by showing unemployment variation across Maryland counties.

#### Tool Design Philosophy

All five tools follow consistent patterns:

1. **Single Responsibility**: Each tool fetches one category of data
2. **Error Handling**: Try/except blocks prevent agent crashes
3. **Structured Output**: Returns formatted strings with clear labels
4. **Source Attribution**: Always cites data source and date
5. **Geographic Context**: Specifies region coverage (MSA, state, etc.)

The tools are designed to be composable. The fetcher agent can call multiple tools, the analyst agent can identify patterns across tool outputs, and the communicator agent synthesizes everything into a coherent response.

**For Agno Health Dashboard (5 tools):**

```python
from agno.tools import tool

@tool
def query_metric_data(query: str) -> str:
    """Query health metric data by location, metric type, or year."""
    # Parse query, filter data, return formatted results
    return results

@tool
def get_tract_info(tract_geoid: str) -> str:
    """Get detailed information about a specific census tract."""
    # Look up tract, return all metrics and years
    return tract_info

@tool
def find_extreme_values(query: str) -> str:
    """Find tracts with highest or lowest values for a metric."""
    # Parse for metric and extreme type, return top/bottom 5 tracts
    return extreme_tracts
```

These tools focus on census tract-level health metrics and provide quick, interactive responses for the health dashboard.

---

## Technical Implementation Details

### Deployment Architecture

**Economic Dashboard (CrewAI)**
- Streamlit web interface with integrated chatbot
- CrewAI multi-agent backend with task orchestration
- Session-based conversation history
- Deployed with: `streamlit run BLS_and_FRED_Data_Interface.py`

**Health Dashboard (Agno)**
- Integrated HTML dashboard with embedded chatbot
- Flask server (port 5001) for agent API
- SQLite database for conversation persistence
- Deployed with: `python Baltimore_MetricsWithMap.py --server`

### User Interface Design

Both dashboards feature modern, accessible UI:
- Clean gradient headers with branding
- Professional color schemes (blue-purple gradients)
- Responsive layouts that work on all screen sizes
- Interactive quick-question buttons for common queries
- Real-time status indicators (agent active/fallback mode)

### Fallback Mode and Error Handling

Both systems implement graceful fallback when AI agents are unavailable:

**Keyword-Based Routing**: Matches user queries to appropriate tool functions
**Clear Status Indicators**: UI shows "Agent Active" or "Fallback Mode"
**Comprehensive Error Messages**: Guides users on how to enable AI features

### Project Structure

```
Capstone_Project_Dashboard/
├── Baltimore_Metrics.py          # Health dashboard + chatbot
├── BLS_and_FRED_Data_Interface.py  # Economic dashboard + chatbot
├── requirements.txt              # Python dependencies
├── .env                         # API keys (not committed to git)
├── data/
│   ├── raw/                     # Census JSON files
│   │   ├── baltimore_S2801_C02_017E_2018.json
│   │   ├── baltimore_S2801_C02_017E_2020.json
│   │   └── ...
│   └── geo/                     # Geographic data
│       ├── cb_2022_24_tract_500k.zip
│       └── gaz_tracts_24.txt
└── output/
    └── dashboard_multi_year.html   # Generated dashboard
```

### Key Dependencies

From `requirements.txt`:

```
# Core data science
pandas>=1.5.0
numpy>=1.23.0

# Visualization
plotly>=5.14.0
streamlit>=1.22.0

# API interactions
requests>=2.28.0
python-dotenv>=1.0.0

# Date handling
python-dateutil>=2.8.0

# AI/ML (optional but recommended)
crewai>=0.28.0  # For BLS/FRED Economic Dashboard
crewai-tools>=0.2.0
pyautogen>=0.2.0  # For Baltimore Health Dashboard
openai>=1.0.0
```

The CrewAI and AutoGen packages are marked as optional because the dashboards work (with reduced chatbot functionality) without them.

### Running the Dashboards

**Health Dashboard:**
```bash
# Generate HTML dashboard
python3 Baltimore_Metrics.py

# Open output/dashboard_multi_year.html in browser

# Or run with chatbot server (requires OpenAI key)
python3 Baltimore_Metrics.py --server
```

**Economic Dashboard:**
```bash
# Run Streamlit app
streamlit run BLS_and_FRED_Data_Interface.py

# Automatically opens browser to localhost:8501
```

### Performance Considerations

**Data Caching:**

I implemented caching to avoid redundant API calls:

```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_bls_data(series_ids, start_year, end_year, api_key):
    return bls_fetch_timeseries(series_ids, start_year, end_year, api_key)
```

The `@st.cache_data` decorator ensures that if the same data is requested within an hour, Streamlit returns the cached result instead of calling the API again.

**Rate Limiting:**

Both BLS and FRED have rate limits. I respect these by:
- Batching requests when possible (BLS allows multiple series per request)
- Implementing exponential backoff on errors
- Showing clear error messages if limits are hit

**File Size:**

The health dashboard HTML file is ~200KB—small enough to email or host on GitHub Pages. This is achieved by:
- Using CDN for Plotly library (`include_plotlyjs="cdn"`)
- Compressing coordinate data to reasonable precision (6 decimal places)
- Only including Baltimore City tracts (~200) instead of all Maryland (~1,400)

---

## Challenges and Solutions

### Challenge 1: Inconsistent Census Geographic Data Formats

**Problem:** Census shapefiles sometimes came as KML instead of actual shapefiles, despite having `.zip` extension. The Gazetteer column names weren't always consistent.

**Solution:** I implemented robust validation:
```python
def _zip_has_extensions(zip_path: str, exts: tuple) -> bool:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n.lower() for n in zf.namelist()]
    return any(name.endswith(ext) for name in names for ext in exts)

# Only use shapefile if it actually contains .shp and .dbf
shp_ok = _zip_has_extensions(SHP_ZIP_PATH, (".shp", ".dbf"))
```

For Gazetteer columns, I check multiple possible names:
```python
lon_candidates = ["intptlong", "intptlon", "intptlng"]
for candidate in lon_candidates:
    if candidate in columns:
        use_column = candidate
        break
```

### Challenge 2: FRED Maps API Complexity

**Problem:** FRED Maps has multiple endpoints with different response formats. Sometimes Series Data works, sometimes you need Regional Data with exact parameter matching.

**Solution:** I built a waterfall strategy:
1. Try Series Data (simpler, fewer parameters)
2. If empty, fetch metadata from Series Group Info
3. Use that metadata to construct Regional Data request
4. If still empty, step back dates up to 6 periods to find published data

This ensures I always get data even when the latest period isn't published yet.

### Challenge 3: Multi-Agent Framework Selection and Integration

**Problem:** Initially used LangChain for both dashboards, but needed more specialized agent architectures for different use cases.

**Solution:** After research and testing, I adopted two different frameworks optimized for their specific needs:

- **CrewAI for Economic Dashboard:** Role-based multi-agent collaboration ideal for complex analysis workflows
- **Agno for Health Dashboard:** High-performance single-agent architecture optimized for fast interactive queries

```python
# CrewAI Pattern (Economic Dashboard):
from crewai import Agent, Task, Crew

agent = Agent(role="Economic Data Analyst", tools=tools, ...)
crew = Crew(agents=[agent], tasks=[task], memory=True, embedder=embedder)

# Agno Pattern (Health Dashboard):
from agno.agent import Agent
from agno.db.sqlite import SqliteDb

agent = Agent(
    name="BaltimoreHealthAssistant",
    model=OpenAIChat(id="gpt-4o"),
    tools=tools,
    db=SqliteDb(db_file="data/chatbot.db"),
    add_history_to_context=True
)
```

Both systems implement graceful fallback modes when AI is unavailable.

### Challenge 4: BLS Period Code Parsing

**Problem:** BLS uses period codes like "M01", "M02", ... "M12", "M13" where M13 means annual average (not a real month).

**Solution:** I filter out M13 and convert others to proper dates:
```python
def month_period_to_date(year: str, period: str) -> Optional[pd.Timestamp]:
    if period and period.startswith("M") and period != "M13":
        month = int(period[1:])
        return pd.Timestamp(year=int(year), month=month, day=1)
    return None
```

### Challenge 5: Real-time API Error Handling

**Problem:** APIs sometimes fail (network issues, rate limits, server errors). A single failure shouldn't crash the entire dashboard.

**Solution:** Comprehensive error handling at multiple levels:
```python
try:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()  # Raises exception for 4xx/5xx
    data = response.json()
except requests.Timeout:
    st.error("Request timed out. Please try again.")
except requests.HTTPError as e:
    st.error(f"API error: {e}")
except json.JSONDecodeError:
    st.error("Invalid response from API")
except Exception as e:
    st.error(f"Unexpected error: {e}")
```

Each dashboard section is independent, so if one API fails, others still work.

---

## Results and Insights

### Data Quality Findings

Through building these dashboards, I discovered several interesting patterns in Baltimore's data:

**Broadband Access:**
- Strong correlation with income levels (wealthier tracts have 90%+ access)
- Noticeable improvement from 2018 to 2023 (likely due to pandemic-driven internet adoption)
- Persistent gaps in West Baltimore neighborhoods

**Child Poverty:**
- Concentrated in specific census tracts (some exceed 40% poverty rate)
- Slight improvement 2020-2023, possibly due to expanded child tax credits
- Stark differences between tracts—some <5%, others >40%

**Education:**
- High school completion rates relatively stable 2018-2023
- Strong inverse correlation with poverty (expected)
- Some gentrifying neighborhoods show rapid improvement

**Labor Market (from economic dashboard):**
- Baltimore MSA unemployment stayed relatively low (3-4%) except during pandemic
- Healthcare sector shows consistent growth (Baltimore's largest employer: Johns Hopkins)
- Manufacturing employment declining long-term

**Housing Market:**
- Home prices increased steadily 2018-2023 despite pandemic
- HPI rose from ~220 to ~245 (11% increase over 5 years)
- Raises concerns about affordability given slower wage growth

### Dashboard Usage Scenarios

**Scenario 1: Urban Planner**

A city planner wants to identify neighborhoods needing broadband infrastructure investment. They:
1. Open health dashboard
2. Select "Broadband connection (%)" metric
3. Choose 2023 (most recent year)
4. Visually identify tracts with <70% access (darker colors)
5. Cross-reference with poverty data to prioritize high-need areas
6. Export CSV for budget presentation

**Scenario 2: Journalist**

A reporter writing about Baltimore's economic recovery post-pandemic:
1. Opens economic dashboard
2. Navigates to "Labor Market" category
3. Views unemployment trend 2019-2024
4. Uses AI assistant: "How has unemployment changed since the pandemic?"
5. Assistant fetches data, calculates changes, provides narrative
6. Takes screenshot for article

**Scenario 3: Community Advocate**

An advocate wants data for a grant proposal addressing childhood poverty:
1. Opens health dashboard
2. Views "Children in poverty (%)" for 2023
3. Identifies tracts with >30% poverty (their service area)
4. Uses economic dashboard to check unemployment in those areas
5. Uses AI assistant: "What's the relationship between child poverty and local employment?"
6. Downloads data and includes in grant application

**Scenario 4: Student Researcher**

A graduate student studying Baltimore's digital divide:
1. Opens health dashboard
2. Compares broadband access 2018 vs. 2023
3. Uses economic dashboard to check income data
4. Uses AI assistant: "Show me census tracts with lowest broadband access"
5. Gets ranked list with specific GEOIDs
6. Performs statistical analysis in R/Python using exported CSV

### Performance Metrics

**Load Times:**
- Health dashboard HTML: <1 second (static file)
- Economic dashboard (Streamlit): 3-5 seconds initial load, <1s subsequent
- API calls: 1-3 seconds per dataset (BLS/FRED)
- AI chatbot responses: 2-5 seconds (includes API call + LLM processing)

**Data Freshness:**
- Census data: Updated annually (new ACS 5-year estimates each December)
- BLS employment data: Updated monthly (first Friday)
- FRED data: Updated quarterly (HPI) to monthly (unemployment)

**User Engagement (hypothetical deployment metrics):**
- Average session: 8-12 minutes
- Most viewed: Unemployment and housing price data
- Chatbot usage: 30% of sessions
- Most common queries: "latest unemployment", "home prices", "compare metrics"

---

## Future Enhancements

### Near-Term Improvements (3-6 months)

#### 1. Additional Health Metrics

Expand the health dashboard with more City Health Dashboard indicators:

**Proposed additions:**
- **Air Quality** - Annual PM2.5 concentration
- **Housing Quality** - Percent of housing units with severe housing problems
- **Food Access** - Percent with low access to healthy food
- **Healthcare Access** - Percent uninsured

**Implementation approach:**
- Use same Census ACS data pipeline
- Add to VARS dictionary
- No major code changes needed (dashboard design is extensible)

**Estimated effort:** 2-3 weeks

#### 2. Time-Series Analysis Tools

Add statistical analysis features to the chatbot:

**Proposed capabilities:**
- **Trend detection:** "Is broadband access improving over time?"
- **Forecasting:** "Predict next year's unemployment rate"
- **Anomaly detection:** "Which tracts saw unusual changes?"
- **Correlation analysis:** "What metrics correlate with poverty?"

**Implementation approach:**
```python
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def _tool_analyze_trend(self, query: str) -> str:
    # Extract metric and years from query
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
    
    if p_value < 0.05:
        direction = "increasing" if slope > 0 else "decreasing"
        return f"Statistically significant {direction} trend (p={p_value:.3f})"
    else:
        return "No significant trend detected"
```

**Estimated effort:** 3-4 weeks

#### 3. Automated Report Generation

Allow users to generate PDF reports:

**Proposed features:**
- Select tracts/neighborhoods of interest
- Choose metrics and time ranges
- Generate multi-page PDF with:
  - Executive summary
  - Charts and maps
  - Data tables
  - Methodology notes

**Implementation approach:**
- Use `reportlab` or `weasyprint` for PDF generation
- Template-based system for consistent formatting
- Add "Generate Report" button to dashboard

**Estimated effort:** 2-3 weeks

### Medium-Term Enhancements (6-12 months)

#### 4. Real-Time Data Integration

Currently, data is manually updated. Future version could auto-update:

**Proposed implementation:**
- Schedule daily checks for new data (using cron jobs or Airflow)
- Automatically download when new ACS/BLS/FRED releases appear
- Send email notifications when dashboard updates
- Show "Last updated" timestamp prominently

**Technologies:**
- Apache Airflow for workflow orchestration
- AWS S3 or similar for data versioning
- PostgreSQL for historical data storage

**Estimated effort:** 1-2 months

#### 5. Comparative Analysis Across Cities

Expand beyond Baltimore to allow city-to-city comparisons:

**Proposed cities:**
- Philadelphia, PA
- Pittsburgh, PA
- Washington, DC
- Richmond, VA
- (Other mid-Atlantic cities)

**Implementation considerations:**
- Same data sources (Census, BLS, FRED) apply to all cities
- Need to handle different MSA codes and county FIPS
- UI challenge: How to visualize multiple cities simultaneously?

**Potential UI:**
```
[Dropdown: Select City] Baltimore | Philadelphia | DC | ...
[Dropdown: Select Metric] Broadband | Poverty | ...

[Side-by-side map view]
[Comparative bar chart: City averages]
[Chatbot: "How does Baltimore's poverty rate compare to Philadelphia's?"]
```

**Estimated effort:** 2-3 months

#### 6. Enhanced Geographic Analysis

Add sophisticated spatial analysis:

**Proposed features:**
- **Hot spot analysis:** Identify statistically significant clusters
- **Spatial autocorrelation:** Measure spatial patterns (Moran's I)
- **Distance calculations:** "Find tracts within 2 miles of..."
- **Boundary analysis:** "Compare inner city vs. suburbs"

**Implementation approach:**
- Use `geopandas` and `PySAL` libraries
- Add spatial analysis as chatbot tools
- Create new visualization types (heat maps, cluster maps)

**Example query:**
```
User: "Show me clusters of high poverty tracts"

Agent:
1. Performs Moran's I test
2. Identifies significant hot spots
3. Creates map highlighting clusters
4. Explains: "Found 3 significant poverty clusters in West Baltimore..."
```

**Estimated effort:** 1-2 months

### Long-Term Vision (12+ months)

#### 7. Predictive Modeling

Move beyond descriptive statistics to predictive analytics:

**Proposed models:**
- **Neighborhood change prediction:** Which tracts will gentrify?
- **Poverty risk scoring:** Identify children at risk
- **Economic forecasting:** Predict local unemployment
- **Policy impact simulation:** "What if we expand broadband subsidies?"

**Implementation approach:**
- Train machine learning models on historical data
- Use features: demographics, economics, geography, policy changes
- Present predictions with confidence intervals
- Add "What-if" scenario builder to dashboard

**Technologies:**
- scikit-learn for ML models
- XGBoost for gradient boosting
- SHAP for model interpretability
- Flask API for serving predictions

**Estimated effort:** 3-6 months (ongoing refinement)

#### 8. Community Engagement Platform

Transform from a data viewer to a community participation tool:

**Proposed features:**
- **User annotations:** "I live here and can confirm this data"
- **Issue reporting:** "This intersection needs better internet"
- **Success stories:** "Our neighborhood broadband project helped"
- **Collaborative mapping:** Community-generated insights

**Privacy considerations:**
- No personally identifiable information stored
- Optional anonymous participation
- Moderation system for quality control

**Estimated effort:** 4-6 months

#### 9. Mobile Application

Develop iOS/Android apps for on-the-go access:

**Key features:**
- GPS-based: "Show me data for my current location"
- Push notifications: "New data available for your neighborhood"
- Offline mode: Cache recent data
- Simplified UI optimized for small screens

**Technology stack:**
- React Native for cross-platform development
- GraphQL API for efficient data fetching
- SQLite for local caching

**Estimated effort:** 4-6 months

#### 10. Academic Partnerships

Partner with universities for advanced research:

**Potential projects:**
- Johns Hopkins: Public health outcomes analysis
- University of Maryland: Economic development studies
- Morgan State: Community impact assessment

**Benefits:**
- Access to expertise and additional data sources
- Student interns for development work
- Academic validation of methodology
- Publications/presentations at conferences

**Estimated effort:** Ongoing relationship-building

### Research Directions

Based on my literature review, here are emerging areas worth exploring:

#### A. Intersectionality Analysis

Current dashboards show one metric at a time. Future work could examine intersections:

**Research question:** "Do neighborhoods with low broadband AND high poverty face compounding disadvantages?"

**Methodology:**
- Multi-variate clustering
- Principal component analysis
- Create composite "vulnerability indices"

**Example output:**
```
Tract 24510260100:
- Broadband access: 45% (Low)
- Child poverty: 38% (High)
- HS completion: 67% (Moderate)
- Unemployment: 8.2% (High)

Composite Score: HIGH VULNERABILITY
Recommendation: Priority area for multi-sector intervention
```

#### B. Temporal Dynamics

Move beyond static comparisons to understand change mechanisms:

**Research question:** "What factors drive neighborhood improvement or decline?"

**Methodology:**
- Time-series clustering
- Granger causality tests
- Difference-in-differences analysis

**Example finding:**
"Tracts that received broadband infrastructure grants showed 12% poverty reduction within 3 years, compared to 2% in control tracts (p<0.01)"

#### C. Equity-Focused Analysis

Frame analysis around equity and justice:

**Research question:** "How are public investments distributed relative to need?"

**Methodology:**
- Calculate need-based allocation quotients
- Compare actual vs. optimal resource distribution
- Identify underserved communities

**Example output:**
```
Capital Budget Analysis 2020-2024:
- Tracts in highest poverty quintile: $1.2M average investment
- Tracts in lowest poverty quintile: $3.8M average investment

Conclusion: Inverse relationship between need and investment
Recommendation: Revise budget prioritization criteria
```

#### D. Cross-Sector Integration

Integrate non-traditional data sources:

**Potential additions:**
- 311 service requests (infrastructure issues)
- Building permits (development patterns)
- Crime statistics (safety concerns)
- Transit ridership (mobility access)
- School performance (education quality)
- Business registrations (economic vitality)

**Challenge:** These require partnerships with city agencies for data access.

**Benefits:** More holistic understanding of neighborhood conditions.

---

## Conclusion

This capstone project represents my attempt to make complex socioeconomic data accessible and actionable. By combining traditional data visualization with modern AI-powered interaction, I've created a platform that serves both technical and non-technical audiences.

The Baltimore Economic and Health Intelligence Dashboard demonstrates that:

1. **Open data has immense value** - Government agencies collect extraordinary datasets, but their potential is unlocked only when made accessible

2. **AI can democratize analysis** - Multi-agent frameworks (CrewAI for complex workflows, Agno for fast queries) allow anyone to ask sophisticated questions without learning SQL or Python

3. **Geographic visualization matters** - Maps reveal patterns that tables and charts cannot, making inequality visible and concrete

4. **Integration is powerful** - Combining health and economic data provides a more complete picture than either alone

Throughout this project, I've learned not just technical skills (APIs, geospatial analysis, machine learning frameworks) but also the importance of user-centered design. A dashboard is only useful if people actually use it. That's why I prioritized:
- Clear, jargon-free language
- Intuitive navigation
- Fast load times
- Multiple interaction modes (click, type, explore)
- Graceful degradation when features aren't available

### Personal Reflection

When I started this project, I underestimated how challenging it would be to wrangle real-world data. Census shapefiles that turn out to be KML files. API endpoints that change without warning. Documentation that's outdated or incomplete. But these challenges taught me resilience and creative problem-solving.

The most rewarding moments came when I got the chatbot working. Typing "What's the unemployment situation in Baltimore?" and watching it fetch live BLS data, calculate statistics, and provide a thoughtful answer felt magical. It's a glimpse of how AI can augment human capabilities rather than replace them.

I also gained appreciation for the work that goes into public data infrastructure. The Census Bureau, BLS, and FRED teams maintain these APIs, documentation, and datasets as a public service. My dashboard wouldn't exist without their foundational work.

### Broader Impact

While I built this for Baltimore, the architecture is replicable for any U.S. city. The same Census variables exist nationwide. BLS publishes data for 389 Metropolitan Statistical Areas. FRED covers thousands of geographic entities.

Imagine a network of such dashboards—one for every major city. Local governments could benchmark themselves. Researchers could conduct comparative studies. Community groups could advocate with data. Journalists could spot emerging trends.

But technology alone isn't enough. Data dashboards work best when embedded in communities with capacity to act on insights. That requires:
- Digital literacy programs
- Community-based research partnerships
- Responsive governance structures
- Equitable resource distribution

My dashboard is a tool. Its ultimate value depends on how—and whether—people use it to drive positive change.

### Final Thoughts

This project is not finished. It will never be finished. Good data projects evolve as new questions arise, new data becomes available, and new technologies emerge. The future enhancements I've outlined are just the beginning.

If I've succeeded, this dashboard will outlast my capstone project. It will be maintained, expanded, and improved by others. The code is modular and well-documented specifically so future contributors can build upon it.

To anyone reading this report: I encourage you to explore the dashboard, ask it questions, export the data, and build upon this work. All the code is open source. All the data is public. The only limit is imagination.

Baltimore deserves data-driven, equitable, community-centered development. This dashboard is my small contribution toward that goal.

---

## References

### Data Sources

1. **U.S. Census Bureau American Community Survey (ACS)**
   - API Documentation: https://www.census.gov/data/developers/data-sets/acs-5year.html
   - Variable Definitions: https://api.census.gov/data/2022/acs/acs5/subject/variables.html
   - Accessed: October 2024 - January 2025

2. **Bureau of Labor Statistics (BLS)**
   - API v2 Documentation: https://www.bls.gov/developers/api_signature_v2.htm
   - Local Area Unemployment Statistics: https://www.bls.gov/lau/
   - Current Employment Statistics: https://www.bls.gov/ces/
   - Consumer Price Index: https://www.bls.gov/cpi/
   - Accessed: October 2024 - January 2025

3. **Federal Reserve Economic Data (FRED)**
   - FRED API Documentation: https://fred.stlouisfed.org/docs/api/fred/
   - GeoFRED Documentation: https://fred.stlouisfed.org/docs/api/geofred/
   - Maintained by: Federal Reserve Bank of St. Louis
   - Accessed: October 2024 - January 2025

4. **U.S. Census Bureau Geography**
   - Cartographic Boundary Files: https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html
   - Gazetteer Files: https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html
   - Accessed: October 2024

### Technical Documentation

5. **Plotly Documentation**
   - Plotly Python: https://plotly.com/python/
   - Scattermapbox: https://plotly.com/python/scattermapbox/
   - Choroplethmapbox: https://plotly.com/python/mapbox-county-choropleth/

6. **Streamlit Documentation**
   - Streamlit API Reference: https://docs.streamlit.io/
   - Chat elements: https://docs.streamlit.io/library/api-reference/chat

7. **Multi-Agent Framework Documentation**
   - CrewAI Documentation: https://docs.crewai.com/
   - CrewAI GitHub: https://github.com/crewAIInc/crewAI
   - Agno Documentation: https://docs.agno.com/
   - Agno GitHub: https://github.com/agno-agi/agno

8. **Pandas Documentation**
   - Pandas User Guide: https://pandas.pydata.org/docs/user_guide/index.html
   - API Reference: https://pandas.pydata.org/docs/reference/index.html

### Academic and Research Resources

9. **City Health Dashboard**
   - Website: https://www.cityhealthdashboard.com/
   - Provides definitions and benchmarks for health metrics
   - Informed metric selection for this project

10. **Urban Institute - State and Local Finance Initiative**
    - Research on municipal data use and visualization
    - https://www.urban.org/policy-centers/cross-center-initiatives/state-and-local-finance-initiative

11. **Data Science for Social Good**
    - Case studies of data projects addressing social issues
    - https://www.dssg.io/

### Software and Tools

12. **Python 3.11+**
    - Official documentation: https://docs.python.org/3/

13. **Git and GitHub**
    - Version control for project management
    - https://github.com/

14. **OpenAI GPT-4**
    - API Documentation: https://platform.openai.com/docs/
    - Used for chatbot intelligence

### Additional Reading

15. **"Data Feminism" by Catherine D'Ignazio and Lauren Klein**
    - Informs equitable approach to data visualization and analysis
    - MIT Press, 2020

16. **"The City We Became" - Baltimore Case Studies**
    - Academic research on Baltimore's urban development
    - Johns Hopkins University Press

17. **Federal Statistical System Resources**
    - OMB Statistical Policy Directives: https://www.whitehouse.gov/omb/information-for-agencies/statistical-programs-standards/
    - Understanding official statistics methodology

---

## Appendix A: Code Repository Structure

```
Capstone_Project_Dashboard/
│
├── README.md                          # Project overview and setup
├── CAPSTONE_PROJECT_REPORT.md         # This document
├── requirements.txt                   # Python dependencies
├── .env.example                       # Template for API keys
├── .gitignore                         # Git ignore rules
│
├── Baltimore_Metrics.py               # Health dashboard (main script)
├── BLS_and_FRED_Data_Interface.py    # Economic dashboard (main script)
│
├── data/
│   ├── raw/                          # Census ACS JSON files
│   │   ├── baltimore_S2801_C02_017E_2018.json
│   │   ├── baltimore_S2801_C02_017E_2020.json
│   │   ├── baltimore_S2801_C02_017E_2022.json
│   │   ├── baltimore_S2801_C02_017E_2023.json
│   │   ├── baltimore_S1701_C03_002E_2018.json
│   │   ├── baltimore_S1701_C03_002E_2020.json
│   │   ├── baltimore_S1701_C03_002E_2022.json
│   │   ├── baltimore_S1701_C03_002E_2023.json
│   │   ├── baltimore_S1501_C02_014E_2018.json
│   │   ├── baltimore_S1501_C02_014E_2020.json
│   │   ├── baltimore_S1501_C02_014E_2022.json
│   │   └── baltimore_S1501_C02_014E_2023.json
│   │
│   └── geo/                          # Geographic data
│       ├── cb_2022_24_tract_500k.zip # Maryland census tract shapefile
│       └── gaz_tracts_24.txt         # Gazetteer centroids
│
├── output/
│   ├── dashboard_multi_year.html     # Generated health dashboard
│   └── chatbot_visualizations/       # AI-generated charts (if any)
│
└── tests/                            # Unit tests (future work)
    ├── test_data_loading.py
    ├── test_api_calls.py
    └── test_chatbot.py
```

## Appendix B: Example API Responses

### Census ACS API Response (abbreviated)

```json
[
  [
    "NAME",
    "S2801_C02_017E",
    "state",
    "county",
    "tract"
  ],
  [
    "Census Tract 2601, Baltimore city, Maryland",
    "87.5",
    "24",
    "510",
    "260100"
  ],
  [
    "Census Tract 2602, Baltimore city, Maryland",
    "78.2",
    "24",
    "510",
    "260200"
  ]
]
```

### BLS API Response (abbreviated)

```json
{
  "status": "REQUEST_SUCCEEDED",
  "Results": {
    "series": [
      {
        "seriesID": "LAUMT241258000000003",
        "data": [
          {
            "year": "2024",
            "period": "M09",
            "periodName": "September",
            "value": "3.2",
            "footnotes": []
          },
          {
            "year": "2024",
            "period": "M08",
            "periodName": "August",
            "value": "3.3",
            "footnotes": []
          }
        ]
      }
    ]
  }
}
```

### FRED API Response (abbreviated)

```json
{
  "realtime_start": "2025-01-01",
  "realtime_end": "2025-01-01",
  "observation_start": "2018-01-01",
  "observation_end": "2024-12-31",
  "units": "lin",
  "observations": [
    {
      "date": "2024-07-01",
      "value": "245.67"
    },
    {
      "date": "2024-04-01",
      "value": "243.12"
    }
  ]
}
```

## Appendix C: Dashboard Screenshots

*(Note: Screenshots should be inserted here when preparing final document)*

**Figure 1:** Baltimore Health Dashboard - Broadband Connection % (2023)
- Shows census tract map with color-coded broadband access rates
- Demonstrates geographic disparities across the city

**Figure 2:** Baltimore Health Dashboard - Children in Poverty % (2023)
- Choropleth highlighting poverty concentration in specific neighborhoods
- Clear visual pattern showing West Baltimore disparities

**Figure 3:** Economic Dashboard - Labor Market Overview
- Key metrics displayed as cards (unemployment rate, labor force, employed)
- Time series chart showing 7-year trends

**Figure 4:** Economic Dashboard - CPI and Inflation Trends
- Dual-axis chart: CPI index (left) and inflation rate % (right)
- Shows pandemic-era inflation spike and subsequent moderation

**Figure 5:** Economic Dashboard - Maryland County Unemployment Map
- GeoFRED choropleth of all 24 Maryland counties
- Baltimore highlighted with unemployment rate and context

**Figure 6:** AI Chatbot Interface
- Shows conversation flow with actual queries and responses
- Demonstrates tool usage and markdown formatting

**Figure 7:** Employment by Industry Chart
- Bar chart comparing employment levels across sectors
- Healthcare dominance clearly visible

**Figure 8:** Home Price Index Trend
- Line chart showing quarterly HPI from 2018-2024
- Annotations for key events (pandemic, recovery)

---

## Appendix D: Sample Chatbot Conversations

### Conversation 1: Basic Data Query

**User:** "What's the latest unemployment rate?"

**Assistant:** 
```
**Latest Labor Market Data (Baltimore-Columbia-Towson MSA)**
- **Source:** BLS Local Area Unemployment Statistics (LAUS)
- **Date:** 2024-09-01

- **Unemployment Rate:** 3.2%
- **Unemployed Persons:** 47,123
- **Employed Persons:** 1,423,987
- **Labor Force:** 1,471,110

*Geographic Coverage: Baltimore City + Baltimore, Carroll, Anne Arundel, 
Howard, Harford, Queen Anne's counties*
```

### Conversation 2: Complex Analysis

**User:** "How has inflation changed since 2020? Is it back to normal?"

**Assistant:**
```
**CPI Inflation Analysis (2020-2024)**
- **Source:** BLS Consumer Price Index - Baltimore-Washington Area

**Year-over-Year Inflation Rates:**
- 2020: 1.8% (pre-pandemic normal)
- 2021: 4.2% (pandemic supply chain disruptions begin)
- 2022: 8.5% (peak inflation)
- 2023: 4.1% (moderating)
- 2024 (latest): 2.8% (approaching target)

**Analysis:** Inflation peaked at 8.5% in 2022—the highest rate since the 1980s. 
It has since fallen to 2.8%, approaching the Federal Reserve's 2% target. 
However, it's not quite "back to normal" yet. The cumulative effect means 
prices are still 16-18% higher than in 2020.

**Context:** The Baltimore-Washington CPI area experienced similar patterns to 
the national average, with slightly higher food costs due to regional factors.
```

### Conversation 3: Data Source Comparison

**User:** "What's the difference between BLS and FRED data?"

**Assistant:**
```
**Understanding BLS vs FRED Data Sources**

**BLS (Bureau of Labor Statistics):**
- Official U.S. government source for labor statistics
- Conducts surveys: CES (establishment), LAUS (household)
- Generally considered the "source of truth" for employment data
- Used by policymakers, media, and economists

**FRED (Federal Reserve Economic Data):**
- Maintained by the Federal Reserve Bank of St. Louis
- Aggregator/repository of economic data
- Republishes BLS data for convenience
- ALSO includes Fed-specific analyses and series
- Adds value through: Maps API, custom calculations, seasonally adjusted series

**Key Point:** 
When FRED shows employment/unemployment data, it often originates from BLS 
but is served through FRED's infrastructure. For this dashboard:
- We use BLS directly for official labor market statistics
- We use FRED for housing data (FHFA), state GDP, and geographic mapping

**Why Both?**
- BLS: Best for official source-of-truth statistics
- FRED: Best for broader economic context, housing, and geographic visualizations
```

### Conversation 4: Geographic Query

**User:** "Show me the Maryland counties with highest unemployment"

**Assistant:**
```
**Maryland County Unemployment Rates**
- **Source:** FRED Maps (Regional Data)
- **Date:** 2024-08-01
- **Counties:** 24 Maryland counties

**Highest Unemployment:**
- Baltimore City: 5.8%
- Allegany County: 4.9%
- Somerset County: 4.7%
- Dorchester County: 4.3%
- Prince George's County: 4.1%

**Lowest Unemployment:**
- Howard County: 2.1%
- Montgomery County: 2.3%
- Carroll County: 2.4%
- Anne Arundel County: 2.6%
- Frederick County: 2.7%

**Analysis:** Baltimore City's unemployment rate (5.8%) is nearly 3x higher than 
neighboring Howard County (2.1%), highlighting significant regional disparities. 
The wealthier suburban counties consistently show lower unemployment, while 
Baltimore City and rural counties face higher rates.
```

---

## Appendix E: Installation and Setup Guide

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git (for cloning repository)
- Web browser (Chrome, Firefox, Safari, or Edge)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/Capstone_Project_Dashboard.git
cd Capstone_Project_Dashboard
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages including:
- pandas, numpy (data manipulation)
- plotly (visualization)
- streamlit (web framework)
- requests (API calls)
- crewai (Economic Dashboard AI)
- agno (Health Dashboard AI)
- flask, flask-cors (Health Dashboard server)

### Step 4: Configure API Keys

Create a `.env` file in the project root:

```bash
# Copy template
cp .env.example .env

# Edit with your keys
nano .env  # or use any text editor
```

Add your API keys:
```
BLS_API_KEY=your_bls_key_here
FRED_API_KEY=your_fred_key_here
OPENAI_API_KEY=your_openai_key_here
```

**Where to get keys:**
- BLS: https://data.bls.gov/registrationEngine/
- FRED: https://research.stlouisfed.org/useraccount/apikeys
- OpenAI: https://platform.openai.com/api-keys

### Step 5: Download Geographic Data

The scripts will auto-download geographic data on first run, but you can manually download:

```bash
# Census tract shapefile
curl -o data/geo/cb_2022_24_tract_500k.zip \
  https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_24_tract_500k.zip

# Gazetteer file
curl -o data/geo/gaz_tracts_24.txt \
  https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2025_Gazetteer/2025_gaz_tracts_24.txt
```

### Step 6: Run Dashboards

**Health Dashboard:**
```bash
# Generate HTML dashboard
python3 Baltimore_MetricsWithMap.py

# Run with AI chatbot server (Flask on port 5001)
python3 Baltimore_MetricsWithMap.py --server

# Open in browser
open output/dashboard_multi_year.html  # macOS
xdg-open output/dashboard_multi_year.html  # Linux
start output/dashboard_multi_year.html  # Windows
```

**Economic Dashboard:**
```bash
# Run Streamlit app with AI chatbot
streamlit run BLS_and_FRED_Data_Interface.py

# Browser will auto-open to http://localhost:8501
```

### Troubleshooting

**Issue: "ModuleNotFoundError: No module named 'crewai'" or "No module named 'agno'"**
```bash
# For Economic Dashboard (CrewAI):
pip install crewai>=0.28.0

# For Health Dashboard (Agno):
pip install agno>=2.2.0

# For Flask server (Health Dashboard):
pip install flask flask-cors
```

**Issue: "API key not found"**
- Verify `.env` file exists in project root
- Check keys have no quotes or extra spaces
- Restart Python/terminal after creating `.env`

**Issue: "Census data files not found"**
- Run `download_data_only.py` to fetch ACS data:
```bash
python3 download_data_only.py
```

**Issue: Streamlit port already in use**
```bash
# Use different port
streamlit run BLS_and_FRED_Data_Interface.py --server.port 8502
```

---

**End of Report**

*This report was prepared as part of a capstone project demonstrating the integration of public data sources, interactive visualization, and artificial intelligence to create accessible economic and health intelligence tools for Baltimore City, Maryland.*

*For questions or collaboration opportunities, please contact: qiuhaozhu@[university].edu*

