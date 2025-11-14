# Baltimore Economic Intelligence Dashboard - README

## 🚀 Quick Start (30 seconds)

```bash
# 1. Install dependencies
pip install streamlit pandas plotly requests python-dotenv python-dateutil

# 2. Get FREE API keys (no credit card needed):
#    - BLS: https://data.bls.gov/registrationEngine/
#    - FRED: https://fred.stlouisfed.org/docs/api/api_key.html

# 3. Create .env file
cat > .env << 'EOF'
BLS_API_KEY=paste_your_bls_key_here
FRED_API_KEY=paste_your_fred_key_here
EOF

# 4. Run it!
streamlit run BLS_and_FRED_Data_Interface.py
```

**That's it!** The dashboard will open in your browser. 🎉

**Optional:** For AI chatbot multi-agent mode, also install:
```bash
pip install crewai crewai-tools
# Then add OPENAI_API_KEY=sk-... to your .env file
```

---

## 🎯 Overview

## ✅ What Was Implemented

### 1. **Enhanced Organization with Categories**
The dashboard is now organized into 6 clear categories:
- 🏠 **Home & Overview** - Introduction and data source explanations
- 💼 **Labor Market** - Unemployment, employment, industry breakdowns
- 💰 **Prices & Inflation** - CPI and cost of living metrics
- 🏘️ **Housing & Real Estate** - Home prices and ownership rates
- 🗺️ **Geographic/County Data** - Maryland county unemployment maps
- 🤖 **AI Assistant** - Intelligent chatbot for data queries

### 2. **Comprehensive Data Source Documentation**
Every metric now includes:
- **Geographic Region** - Exact coverage (MSA, counties, state, national)
- **Data Source** - BLS LAUS, BLS CES, BLS CPI, FRED, etc.
- **Frequency** - Monthly, Quarterly, Annual
- **Seasonal Adjustment** - SA or NSA
- **Latest Date** - When the data was last updated

### 3. **Expanded Metrics**

#### Labor Market (BLS):
- ✅ Unemployment Rate (LAUS)
- ✅ Number of Unemployed Persons (LAUS)
- ✅ Number of Employed Persons (LAUS)
- ✅ Labor Force Size (LAUS)
- ✅ Total Nonfarm Employment (CES)
- ✅ Goods-Producing Employment (CES)
- ✅ Service-Providing Employment (CES)
- ✅ **Manufacturing Employment (CES)** - NEW
- ✅ **Retail Trade Employment (CES)** - NEW
- ✅ **Professional & Business Services (CES)** - NEW
- ✅ **Healthcare Employment (CES)** - NEW
- ✅ **Leisure & Hospitality Employment (CES)** - NEW
- ✅ **Government Employment (CES)** - NEW

#### Prices & Inflation (BLS):
- ✅ CPI-U All Items
- ✅ **CPI-U Food** - NEW
- ✅ **CPI-U Housing** - NEW
- ✅ **CPI-U Transportation** - NEW
- ✅ **CPI-U Medical Care** - NEW
- ✅ **CPI-U Energy** - NEW

#### Housing (FRED):
- ✅ Home Price Index (FHFA)
- ✅ Home Ownership Rate (National)

### 4. **Clear Explanation of BLS vs FRED**

Added comprehensive documentation explaining:
- **BLS** is the official U.S. government source (primary surveys)
- **FRED** is a Federal Reserve repository that republishes BLS data + adds Fed-specific analyses
- Why both sources might show similar unemployment data (FRED republishes BLS)
- When to use each source

Example from the code:
```python
# --- BLS LAUS (Local Area Unemployment Statistics) ---
# Source: BLS Current Population Survey (household survey)
# Region: Baltimore-Columbia-Towson, MD MSA (CBSA code 12580)
# Coverage: Baltimore City + Baltimore, Carroll, Anne Arundel, Howard, Harford, Queen Anne's counties
# Frequency: Monthly, Not Seasonally Adjusted (NSA)
# Note: LAUS provides official unemployment rates used by media and policymakers

LAUS_UR_MSA = "LAUMT241258000000003"  # Unemployment rate (%) - Baltimore MSA
```

### 5. **Multi-Agent AI Chatbot with CrewAI**

Implemented `BaltimoreEconomicChatbot` class with:
- **CrewAI Multi-Agent System** with 3 specialized agents that collaborate:
  1. **Economic Data Fetcher** - Retrieves data from BLS & FRED APIs
  2. **Economic Data Analyzer** - Interprets trends and calculates statistics
  3. **Economic Data Communicator** - Explains results in clear language

- **6 Data Tools:**
  1. `get_latest_unemployment` - Fetches BLS LAUS data
  2. `get_cpi_inflation` - Fetches BLS CPI data  
  3. `get_employment_by_industry` - Fetches BLS CES by sector
  4. `get_home_prices` - Fetches FRED HPI data
  5. `get_county_unemployment` - Fetches FRED Maps county data
  6. `compare_bls_fred_sources` - Explains data source differences

- **Intelligent Orchestration:** CrewAI assigns tasks to appropriate agents and coordinates their collaboration
- **Fallback Mode:** If OpenAI API key is not available, uses keyword-based routing
- **Regional Context:** Always includes geographic coverage in responses
- **Source Attribution:** Every response mentions BLS LAUS, FRED, etc.

### 6. **Improved User Interface**

- **Sidebar Navigation** - Easy category selection
- **Home Page** - Quick start guide and BLS vs FRED explanation
- **Sub-tabs within Categories** - e.g., Labor Market has "Overview", "By Industry", "Trends"
- **Interactive Metric Selection** - Users choose which metrics to visualize
- **Download Buttons** - CSV export for every dataset
- **Status Indicators** - Shows data source, date, region for every visualization

### 7. **Better Comments Throughout**

Every section now has:
```python
# ============================================================
# CATEGORY 1: LABOR MARKET - UNEMPLOYMENT & EMPLOYMENT
# ============================================================
# These metrics track job market health in Baltimore MSA
```

And every series has detailed comments:
```python
CES_HEALTH_MSA  = "SMU24125806562000001"  # Healthcare Employment - Baltimore MSA
```

## 📊 How to Use

### Running the Dashboard

```bash
# 1. Install dependencies (if not already installed)
pip install streamlit pandas plotly requests python-dotenv dateutil crewai crewai-tools

# 2. Create .env file with your API keys
echo "BLS_API_KEY=your_bls_key_here" > .env
echo "FRED_API_KEY=your_fred_key_here" >> .env
echo "OPENAI_API_KEY=your_openai_key_here" >> .env  # Optional for AI multi-agent chatbot

# 3. Run the dashboard
streamlit run BLS_and_FRED_Data_Interface.py
```

### Navigation

1. **Sidebar** - Select category (Labor Market, Inflation, Housing, etc.)
2. **Within Each Dashboard:**
   - Choose metrics to visualize
   - View latest values in metric cards
   - Download data as CSV
3. **AI Assistant:**
   - Ask questions like "What's the latest unemployment rate?"
   - Get responses with full regional context and source attribution

## 🔑 Understanding the Data

### Geographic Coverage by Category:

| Category | Region | Notes |
|----------|--------|-------|
| Labor Market (LAUS/CES) | Baltimore-Columbia-Towson MSA (12580) | Includes 6 counties |
| CPI | Baltimore-Washington Area | Broader than MSA |
| Housing (HPI) | Baltimore MSA | Same as labor market |
| County Maps | All Maryland Counties | 24 counties total |

### BLS vs FRED - Key Differences:

**BLS:**
- Official government statistics
- Primary data source
- ~145,000 businesses surveyed (CES)
- Use for: Official employment/unemployment/inflation

**FRED:**
- Fed repository/aggregator
- Republishes BLS + adds Fed data
- Provides mapping tools
- Use for: Housing, GDP, geographic viz, broader context

**Important:** When FRED shows unemployment data, it often **originates from BLS** but is served through FRED's infrastructure for convenience.

## 🤖 AI Chatbot Features

The chatbot can:
- ✅ Fetch live data from BLS and FRED APIs
- ✅ Explain regional coverage for each metric
- ✅ Compare data sources (BLS vs FRED)
- ✅ Answer questions about different metrics
- ✅ Provide geographic context

**Example Queries:**
- "What's the latest unemployment rate?"
- "Show me employment by industry"
- "What's the difference between BLS and FRED?"
- "How have home prices changed?"
- "Which Maryland counties have the highest unemployment?"

## 📝 Code Structure

```
BLS_and_FRED_Data_Interface.py
├── Lines 1-1190: Old commented code (kept for reference)
├── Lines 1191-1230: Imports & Config
├── Lines 1231-1260: API Endpoints
├── Lines 1261-1455: Series Definitions (organized by category)
│   ├── Category 1: Labor Market (LAUS, CES)
│   ├── Category 2: Prices & Inflation (CPI)
│   ├── Category 3: Housing (FHFA)
│   ├── Category 4: Economic Indicators
│   └── Category 5: Geographic/County Data
├── Lines 1456-1825: Helper Functions (BLS/FRED API calls)
├── Lines 1826-2214: Dashboard Display Functions
│   ├── display_labor_market_dashboard()
│   ├── display_prices_inflation_dashboard()
│   ├── display_housing_dashboard()
│   └── display_county_map_dashboard()
├── Lines 2215-2642: AI Chatbot Class
│   ├── BaltimoreEconomicChatbot
│   ├── 6 Tool Functions
│   └── display_chatbot_interface()
└── Lines 2643-2793: Main Application Entry Point
```

## 🎨 Visual Improvements

- **Metric Cards** - Show latest values prominently
- **Color-Coded Categories** - Each category has its own icon
- **Dual-Axis Charts** - E.g., CPI index + inflation rate on same chart
- **Choropleth Maps** - Maryland county unemployment with color scale
- **Interactive Legends** - Click to toggle series on/off
- **Responsive Layout** - Works on desktop and tablet

## 🔒 API Key Requirements

| API | Required? | Purpose |
|-----|-----------|---------|
| BLS_API_KEY | ✅ Yes | Fetch employment, unemployment, CPI data |
| FRED_API_KEY | ✅ Yes | Fetch housing, GDP, county map data |
| OPENAI_API_KEY | ⚠️ Optional | Enable AI multi-agent chatbot mode |

**Without OpenAI key:** Chatbot works in basic mode with keyword matching  
**With OpenAI key:** Chatbot uses CrewAI multi-agent system for intelligent analysis

## 🐛 Troubleshooting

### ✅ Multi-Agent System Status
**The dashboard now uses CrewAI for advanced multi-agent AI capabilities.**

**What happens:**
- ✅ **Without CrewAI:** Chatbot uses keyword-based routing (fully functional)
- ✅ **With CrewAI but no OpenAI key:** Same as above  
- ✅ **With CrewAI + OpenAI key:** Full multi-agent mode with specialized agent collaboration

### "Missing API keys" error
- Check your `.env` file is in the same directory as the script
- Make sure keys are in format: `BLS_API_KEY=your_key` (no quotes, no spaces)
- Create `.env` from template:
  ```bash
  cp env_config.txt .env
  # Then edit .env and add your actual keys
  ```

### "CrewAI not available" message
This is **NORMAL** if you haven't installed CrewAI. The dashboard will work fine!

**To add multi-agent AI capabilities:**
```bash
pip install crewai crewai-tools
```

Then add to your `.env`:
```
OPENAI_API_KEY=sk-your-key-here
```

**Without CrewAI:** Chatbot uses keyword matching (works great for common queries)  
**With CrewAI:** Chatbot uses multi-agent collaboration with GPT-4o (intelligently analyzes and explains complex data)

### FRED Maps not loading
- FRED Maps API can be slow or occasionally unavailable
- The code auto-retries with date back-off (up to 6 attempts)
- Check that FRED_API_KEY is valid
- Try again in a few minutes if it fails

### Empty charts
- **Most common:** API keys not set correctly in `.env`
- Verify your API keys are valid (test at bls.gov/developers and fred.stlouisfed.org)
- Check internet connection
- BLS API has rate limits:
  - Unregistered: 25 requests/day
  - Registered: 500 requests/day
  - Get a free key at: https://data.bls.gov/registrationEngine/

### Streamlit showing "Rerun" constantly
- This can happen if there's an error in data fetching
- Check the terminal/console for error messages
- Verify all API keys are set correctly
- Try clearing Streamlit cache: Press 'C' in the browser window

### Charts show but no data points
- The date range might be outside available data
- BLS/FRED may not have recent data published yet
- Try expanding the date range in the code (change `START_YEAR`)

### Performance is slow
- First load fetches data from BLS/FRED APIs (can take 10-30 seconds)
- Subsequent loads use Streamlit's caching (much faster)
- FRED Maps choropleth is the slowest component (30-60 seconds)
- Consider commenting out the county map if you don't need it

## 🚀 Next Steps / Future Enhancements

Possible additions:
- Add Maryland state GDP data (FRED has "MDRGSP")
- Include wage/earnings data by industry
- Add comparison between Baltimore MSA and other MSAs
- Historical recession indicators overlay
- Export to PDF/PowerPoint functionality

## 📞 Support

For issues with:
- **BLS API:** https://www.bls.gov/developers/
- **FRED API:** https://fred.stlouisfed.org/docs/api/
- **Dashboard Code:** Check comments in BLS_and_FRED_Data_Interface.py

## ✨ Summary

Your dashboard is now:
1. ✅ **Organized** - Clear categories and subcategories
2. ✅ **Documented** - Every metric has region, source, frequency
3. ✅ **Expanded** - More metrics (9 industry sectors, 6 CPI categories)
4. ✅ **Clear** - Explains BLS vs FRED distinction
5. ✅ **Intelligent** - AI chatbot with CrewAI multi-agent collaboration
6. ✅ **Professional** - Production-ready with proper error handling
7. ✅ **Advanced** - Specialized agents work together (fetcher, analyzer, communicator)

Enjoy your enhanced Baltimore Economic Intelligence Dashboard with CrewAI! 🎉

