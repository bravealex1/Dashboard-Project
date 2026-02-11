# from __future__ import annotations

# import os
# import sys
# import threading
# from pathlib import Path
# from typing import List, Dict, Optional, Tuple

# import pandas as pd
# import numpy as np
# import streamlit as st
# from dotenv import find_dotenv, load_dotenv
# from streamlit.components.v1 import html as st_html

# # Add parent directory to path to import from Baltimore_MetricsWithMap
# HERE = Path(__file__).resolve().parent
# sys.path.insert(0, str(HERE))

# # Import dashboard generation functions
# try:
#     from Baltimore_MetricsWithMap import (
#         YEARS, VARS, HERE as BALTIMORE_HERE,
#         combine_metrics, ensure_geo, centroids_from_gazetteer,
#         MAPBOX_STYLE, CITY_CENTER, DEFAULT_ZOOM, OUT_HTML
#     )
#     DASHBOARD_AVAILABLE = True
# except ImportError as e:
#     print(f"Warning: Could not import Baltimore_MetricsWithMap: {e}")
#     DASHBOARD_AVAILABLE = False
#     YEARS = [2018, 2020, 2022, 2023]
#     VARS = {}

# # Try to import Agno for chatbot
# try:
#     from agno.agent import Agent
#     from agno.models.openai import OpenAIChat
#     from agno.tools import tool
#     from agno.db.sqlite import SqliteDb
#     AGNO_AVAILABLE = True
# except ImportError:
#     AGNO_AVAILABLE = False
#     Agent = None
#     OpenAIChat = None
#     tool = None
#     SqliteDb = None

# # Try to import Flask for embedded chatbot server
# try:
#     from flask import Flask, jsonify, request
#     from flask_cors import CORS
#     from werkzeug.serving import make_server
#     FLASK_AVAILABLE = True
# except ImportError:
#     Flask = None
#     jsonify = None
#     request = None
#     CORS = None
#     make_server = None
#     FLASK_AVAILABLE = False

# # ================================================================================
# # APP CONFIGURATION
# # ================================================================================
# st.set_page_config(
#     page_title="Baltimore Health Dashboard",
#     page_icon="🏥",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # Load environment variables
# load_dotenv(find_dotenv(), override=False)

# DEFAULT_HTML_PATH = "dashboard_multi_year.html"
# HARDCODED_OPENAI_API_KEY = "REPLACE_WITH_YOUR_OPENAI_KEY"
# SERVER_HOST = "0.0.0.0"
# SERVER_PORT = 5001

# # ================================================================================
# # API KEY MANAGEMENT
# # ================================================================================
# def get_openai_api_key() -> Optional[str]:
#     """
#     Get OpenAI API key from Streamlit secrets (cloud) or environment variables (local).
    
#     Returns:
#         API key string or None if not found
#     """
#     # Use hardcoded key first for Streamlit Cloud deployment convenience
#     if HARDCODED_OPENAI_API_KEY and HARDCODED_OPENAI_API_KEY != "REPLACE_WITH_YOUR_OPENAI_KEY":
#         return HARDCODED_OPENAI_API_KEY.strip()
    
#     # Try Streamlit secrets first (for Streamlit Cloud deployment)
#     try:
#         if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
#             return st.secrets['OPENAI_API_KEY']
#     except Exception:
#         pass
    
#     # Fall back to environment variable (for local development)
#     return os.getenv('OPENAI_API_KEY', '').strip() or None

# # ================================================================================
# # BALTIMORE HEALTH CHATBOT (Integrated from Baltimore_MetricsWithMap.py)
# # ================================================================================
# class BaltimoreHealthChatbot:
#     """
#     Chatbot for Baltimore Health Dashboard integrated directly into Streamlit.
#     Uses Agno framework for intelligent AI conversations.
#     """

#     def __init__(self, openai_api_key: str = None):
#         """Initialize the chatbot with dashboard data."""
#         self.api_key = openai_api_key
#         self.agent = None
#         self.agent_init_error = None
        
#         # Load dashboard data
#         self.data = self._load_dashboard_data()
#         self.geo_data = self._load_geographic_data()

#         # Initialize the agent if Agno is available
#         if AGNO_AVAILABLE and DASHBOARD_AVAILABLE:
#             if not self.api_key:
#                 self.agent_init_error = "No OpenAI API key provided"
#             else:
#                 try:
#                     self.agent = self._create_agent()
#                 except Exception as e:
#                     self.agent_init_error = str(e)
#         else:
#             if not AGNO_AVAILABLE:
#                 self.agent_init_error = "Agno not available"
#             if not DASHBOARD_AVAILABLE:
#                 self.agent_init_error = "Dashboard module not available"

#     def _load_dashboard_data(self) -> pd.DataFrame:
#         """Load dashboard data using existing functions."""
#         if not DASHBOARD_AVAILABLE:
#             return pd.DataFrame()
        
#         try:
#             df_wide = combine_metrics(YEARS)
#             if df_wide.empty:
#                 return pd.DataFrame()

#             df_list = []
#             for metric_key, (var_code, description) in VARS.items():
#                 if var_code in df_wide.columns:
#                     metric_df = df_wide[['GEOID', 'year', var_code]].copy()
#                     metric_df['metric_key'] = metric_key
#                     metric_df['metric_name'] = description
#                     metric_df['var_code'] = var_code
#                     metric_df = metric_df.dropna(subset=[var_code])
#                     df_list.append(metric_df)

#             if not df_list:
#                 return pd.DataFrame()

#             return pd.concat(df_list, ignore_index=True)

#         except Exception as e:
#             print(f"Error loading dashboard data: {e}")
#             return pd.DataFrame()

#     def _load_geographic_data(self) -> pd.DataFrame:
#         """Load geographic data using existing functions."""
#         if not DASHBOARD_AVAILABLE:
#             return pd.DataFrame()
        
#         try:
#             geo_avail = ensure_geo()
#             if geo_avail.get("gaz_txt"):
#                 return centroids_from_gazetteer(geo_avail["gaz_txt"])
#         except Exception as e:
#             print(f"Error loading geographic data: {e}")
#         return pd.DataFrame()

#     def _create_tools(self):
#         """Create Agno tools for the agent."""
#         chatbot_self = self
        
#         @tool
#         def query_metric_data(query: str) -> str:
#             """Query health metric data by location, metric type, or year."""
#             return chatbot_self._query_metric_data(query)
        
#         @tool
#         def get_tract_info(tract_geoid: str) -> str:
#             """Get detailed information about a specific census tract."""
#             return chatbot_self._get_tract_info(tract_geoid)
        
#         @tool
#         def compare_metrics(query: str) -> str:
#             """Compare different metrics or time periods."""
#             return chatbot_self._compare_metrics(query)
        
#         @tool
#         def find_extreme_values(query: str) -> str:
#             """Find tracts with highest or lowest values for a metric."""
#             return chatbot_self._find_extreme_values(query)
        
#         @tool
#         def get_summary_stats(query: str) -> str:
#             """Get summary statistics for metrics."""
#             return chatbot_self._get_summary_stats(query)
        
#         return [query_metric_data, get_tract_info, compare_metrics, find_extreme_values, get_summary_stats]
    
#     def _create_agent(self) -> Agent:
#         """Create Agno agent with custom tools."""
#         if not AGNO_AVAILABLE:
#             return None
        
#         tools = self._create_tools()
#         db_file = HERE / "data" / "baltimore_health_chatbot.db"
#         db_file.parent.mkdir(parents=True, exist_ok=True)
#         db = SqliteDb(db_file=str(db_file))
        
#         agent = Agent(
#             name="BaltimoreHealthAssistant",
#             model=OpenAIChat(id="gpt-4o", api_key=self.api_key),
#             tools=tools,
#             db=db,
#             description="You are a helpful assistant for the Baltimore City Health Dashboard. You can help users understand health metrics including broadband connection, children in poverty, and high school completion for years 2018, 2020, 2022, and 2023.",
#             instructions=[
#                 "Always specify which metric, year, and location you're discussing",
#                 "Use tools to fetch real data instead of making assumptions",
#                 "Provide clear, concise answers with specific numbers when available",
#                 "If you don't have data for a query, say so clearly"
#             ],
#             markdown=True,
#             add_history_to_context=True,
#             num_history_messages=10,
#             read_chat_history=True
#         )
        
#         return agent

#     def _query_metric_data(self, query: str) -> str:
#         """Query health metric data based on user specifications."""
#         query_lower = query.lower()
#         metric_key = None
#         for key, (_, description) in VARS.items():
#             if any(word in query_lower for word in description.lower().split()):
#                 metric_key = key
#                 break

#         year = None
#         for y in YEARS:
#             if str(y) in query:
#                 year = y
#                 break

#         filtered_data = self.data.copy()
#         if metric_key:
#             filtered_data = filtered_data[filtered_data['metric_key'] == metric_key]
#         if year:
#             filtered_data = filtered_data[filtered_data['year'] == year]

#         if filtered_data.empty:
#             return "No data found matching your query."

#         result = "Query Results:\n\n"
#         for (mk, yr), group in filtered_data.groupby(['metric_key', 'year']):
#             var_code = group.iloc[0]['var_code']
#             values = group[var_code].dropna()
#             if len(values) > 0:
#                 metric_name = VARS.get(mk, ("", "Unknown"))[1]
#                 result += f"{metric_name} ({yr}):\n"
#                 result += f"  - Tracts: {len(values)}\n"
#                 result += f"  - Average: {values.mean():.2f}%\n"
#                 result += f"  - Range: {values.min():.2f}% - {values.max():.2f}%\n\n"

#         return result

#     def _get_tract_info(self, tract_geoid: str) -> str:
#         """Get detailed information about a specific census tract."""
#         tract_data = self.data[self.data['GEOID'] == tract_geoid]
#         if tract_data.empty:
#             return f"Tract {tract_geoid} not found."

#         result = f"Census Tract {tract_geoid}:\n\n"
#         for _, row in tract_data.iterrows():
#             var_code = row['var_code']
#             value = row[var_code]
#             result += f"{row['metric_name']} ({row['year']}): {value:.2f}%\n"

#         return result

#     def _compare_metrics(self, query: str) -> str:
#         """Compare different metrics or time periods."""
#         recent_data = self.data[self.data['year'] == max(YEARS)]
#         result = "Comparison (Most Recent Year):\n\n"
        
#         for mk in recent_data['metric_key'].unique():
#             metric_data = recent_data[recent_data['metric_key'] == mk]
#             var_code = metric_data.iloc[0]['var_code']
#             avg_value = metric_data[var_code].mean()
#             metric_name = VARS.get(mk, ("", "Unknown"))[1]
#             result += f"{metric_name}: {avg_value:.2f}%\n"

#         return result

#     def _find_extreme_values(self, query: str) -> str:
#         """Find tracts with highest or lowest values."""
#         query_lower = query.lower()
#         metric_key = None
#         for key, (_, description) in VARS.items():
#             if any(word in query_lower for word in description.lower().split()):
#                 metric_key = key
#                 break

#         if not metric_key:
#             return "Please specify which metric to analyze."

#         is_highest = any(word in query_lower for word in ['highest', 'best', 'most', 'top'])
#         if not is_highest and not any(word in query_lower for word in ['lowest', 'worst', 'least', 'bottom']):
#             return "Please specify highest or lowest values."

#         year = max(YEARS)
#         for y in YEARS:
#             if str(y) in query:
#                 year = y
#                 break

#         metric_data = self.data[(self.data['metric_key'] == metric_key) & (self.data['year'] == year)]
#         if metric_data.empty:
#             return f"No data found for {VARS.get(metric_key, ('', 'metric'))[1]} in {year}."

#         var_code = metric_data.iloc[0]['var_code']
#         sorted_data = metric_data.sort_values(var_code, ascending=not is_highest).head(5)
        
#         extreme_word = "highest" if is_highest else "lowest"
#         metric_name = VARS.get(metric_key, ("", "Unknown"))[1]
#         result = f"Tracts with {extreme_word} {metric_name.lower()} ({year}):\n\n"
        
#         for _, row in sorted_data.iterrows():
#             result += f"Tract {row['GEOID']}: {row[var_code]:.2f}%\n"

#         return result

#     def _get_summary_stats(self, query: str) -> str:
#         """Get summary statistics for metrics."""
#         result = "Summary Statistics:\n\n"
#         for (mk, yr), group in self.data.groupby(['metric_key', 'year']):
#             var_code = group.iloc[0]['var_code']
#             values = group[var_code].dropna()
#             if len(values) > 0:
#                 metric_name = VARS.get(mk, ("", "Unknown"))[1]
#                 result += f"{metric_name} ({yr}):\n"
#                 result += f"  - Tracts: {len(values)}\n"
#                 result += f"  - Mean: {values.mean():.2f}%\n"
#                 result += f"  - Std Dev: {values.std():.2f}%\n"
#                 result += f"  - Range: {values.min():.2f}% - {values.max():.2f}%\n\n"

#         return result

#     def chat(self, message: str) -> str:
#         """Process a user message and return a response."""
#         if self.agent is None:
#             return self._simple_response(message)

#         try:
#             response = self.agent.run(message)
#             if hasattr(response, 'content'):
#                 return response.content
#             elif isinstance(response, str):
#                 return response
#             else:
#                 return str(response)
#         except Exception as e:
#             return f"I apologize, but I encountered an error: {str(e)}"

#     def _simple_response(self, message: str) -> str:
#         """Simple fallback response when agent is not available."""
#         message_lower = message.lower()
        
#         if any(word in message_lower for word in ['hello', 'hi', 'hey']):
#             return "Hello! I can help you explore Baltimore health metrics. However, the AI agent is not currently available. Please check that Agno is installed and an OpenAI API key is configured."
        
#         if 'help' in message_lower:
#             return """I can help you with Baltimore health data including:
# - Broadband connection rates
# - Children in poverty rates  
# - High school completion rates

# Available years: 2018, 2020, 2022, 2023

# Note: Full AI capabilities require Agno and an OpenAI API key."""
        
#         return "I'm running in limited mode. For full AI capabilities, please ensure Agno is installed and provide an OpenAI API key."

# # ================================================================================
# # MAIN APP
# # ================================================================================
# def load_dashboard_html(html_path: Path) -> str:
#     """Load the static Plotly dashboard HTML file."""
#     html_path = ensure_path(html_path)
#     if not html_path.exists():
#         return None
#     return html_path.read_text(encoding="utf-8")


# def ensure_path(path_candidate) -> Path:
#     """Coerce strings/Path-like values into a Path object safely."""
#     if isinstance(path_candidate, Path):
#         return path_candidate
#     try:
#         # Primary candidate: as provided (typically relative to cwd)
#         first = Path(path_candidate)
#         if first.exists():
#             return first
#         # Fallback: relative to the script directory
#         second = HERE / path_candidate
#         if second.exists():
#             return second
#         # Fallback: inside output directory next to script
#         third = HERE / "output" / Path(path_candidate).name
#         if third.exists():
#             return third
#         return first
#     except Exception:
#         return DEFAULT_HTML_PATH


# def create_embedded_flask_app(chatbot: BaltimoreHealthChatbot):
#     """
#     Create a lightweight Flask app that mirrors the endpoints expected by
#     dashboard_multi_year.html so we don't need to run a separate server.
#     """
#     if not FLASK_AVAILABLE:
#         raise RuntimeError("Flask is not available. Install with: pip install flask flask-cors")

#     app = Flask(__name__)
#     CORS(app)

#     @app.route("/status", methods=["GET"])
#     def status():
#         return jsonify({
#             "initialized": chatbot is not None,
#             "agent_available": bool(chatbot and chatbot.agent),
#             "agent_init_error": getattr(chatbot, "agent_init_error", None),
#             "agno_available": AGNO_AVAILABLE,
#             "api_key_present": bool(chatbot and getattr(chatbot, "api_key", None)),
#             "data_rows": len(chatbot.data) if chatbot is not None and hasattr(chatbot, "data") else 0,
#         })

#     @app.route("/chat", methods=["POST"])
#     def chat():
#         if chatbot is None:
#             return jsonify({"error": "Chatbot not initialized"}), 500

#         payload = request.get_json(silent=True) or {}
#         user_message = payload.get("message", "").strip()
#         if not user_message:
#             return jsonify({"error": "No message provided"}), 400

#         response = chatbot.chat(user_message)
#         return jsonify({
#             "response": response,
#             "using_agent": bool(chatbot.agent),
#             "agent_init_error": chatbot.agent_init_error,
#         })

#     return app


# def ensure_embedded_server(chatbot: BaltimoreHealthChatbot, host: str = SERVER_HOST, port: int = SERVER_PORT) -> Tuple[bool, Optional[str]]:
#     """
#     Start the embedded Flask server once per Streamlit session.
#     Returns (started, error_message).
#     """
#     server_state = st.session_state.setdefault(
#         "_chatbot_server",
#         {"thread": None, "server": None, "port": port, "error": None},
#     )

#     existing_thread = server_state.get("thread")
#     if existing_thread and existing_thread.is_alive() and server_state.get("port") == port:
#         return True, None

#     if not FLASK_AVAILABLE:
#         return False, "Flask is not installed (pip install flask flask-cors)"

#     try:
#         app = create_embedded_flask_app(chatbot)
#         server = make_server(host, port, app)
#         thread = threading.Thread(target=server.serve_forever, daemon=True)
#         thread.start()
#         server_state.update({"thread": thread, "server": server, "port": port, "error": None})
#         return True, None
#     except Exception as exc:
#         server_state.update({"thread": None, "server": None, "error": str(exc)})
#         return False, str(exc)


# def get_chatbot_instance(api_key: Optional[str]) -> BaltimoreHealthChatbot:
#     """Initialize or fetch the cached chatbot instance."""
#     if "chatbot" not in st.session_state:
#         with st.spinner("Initializing chatbot..."):
#             st.session_state.chatbot = BaltimoreHealthChatbot(openai_api_key=api_key)
#     return st.session_state.chatbot

# def main() -> None:
#     st.title("🏥 Baltimore City Health Dashboard")
#     st.caption("Interactive health metrics dashboard with AI-powered chatbot")

#     # Shared setup
#     html_path = ensure_path(DEFAULT_HTML_PATH)
#     html_exists = html_path.exists()
#     api_key = get_openai_api_key()
#     chatbot = get_chatbot_instance(api_key=api_key)
#     server_running = False
#     server_error = None

#     if html_exists:
#         server_running, server_error = ensure_embedded_server(chatbot, port=SERVER_PORT)

#     # Sidebar configuration
#     with st.sidebar:
#         st.header("Dashboard")
        
#         # Check for dashboard HTML
#         if html_exists:
#             st.success("Dashboard loaded")
#         else:
#             st.warning("Dashboard HTML not found")
#             st.info("Generate it by running:\n```bash\npython Baltimore_MetricsWithMap.py\n```")
        
#         if html_exists:
#             if server_running:
#                 st.success(f"Embedded chatbot server running on port {SERVER_PORT}")
#             else:
#                 st.warning("Chatbot server unavailable")
#                 if server_error:
#                     st.caption(server_error)
        
#         st.markdown("---")
#         st.header("AI Chatbot")
        
#         # API key status
#         if api_key:
#             st.success("OpenAI API key configured")
#         else:
#             st.warning("No API key found")
#             st.info("Set OPENAI_API_KEY in:\n- `.env` file (local)\n- Streamlit secrets (cloud)")
        
#         # Agent status
#         if AGNO_AVAILABLE:
#             st.success("Agno framework available")
#         else:
#             st.error("Agno not installed")
#             st.info("Install: `pip install agno`")

#     # Main content area - two tabs
#     tab1, tab2 = st.tabs(["📊 Dashboard", "💬 AI Assistant"])
    
#     with tab1:
#         st.subheader("Interactive Health Metrics Map")
        
#         if html_exists:
#             dashboard_html = load_dashboard_html(html_path)
#             st_html(dashboard_html, height=900, scrolling=True)
            
#             if not server_running:
#                 st.warning(
#                     "The dashboard chat widget may not respond because the embedded chatbot server is not running."
#                 )
#         else:
#             st.error("Dashboard HTML file not found. Please generate it first.")
#             st.code("python Baltimore_MetricsWithMap.py", language="bash")
    
#     with tab2:
#         st.subheader("Chat with Baltimore Health Data")
        
#         if "chat_history" not in st.session_state:
#             st.session_state.chat_history = []
        
#         # Display agent status
#         chatbot = st.session_state.chatbot
#         if chatbot.agent_init_error:
#             st.warning(f"Chatbot running in fallback mode: {chatbot.agent_init_error}")
#         elif chatbot.agent:
#             st.success("AI agent active and ready")
        
#         # Display chat history
#         for msg in st.session_state.chat_history:
#             with st.chat_message(msg["role"]):
#                 st.markdown(msg["content"])
        
#         # Chat input
#         user_input = st.chat_input("Ask about Baltimore health data...")
        
#         if user_input:
#             # Display user message
#             with st.chat_message("user"):
#                 st.markdown(user_input)
#             st.session_state.chat_history.append({"role": "user", "content": user_input})
            
#             # Get and display assistant response
#             with st.chat_message("assistant"):
#                 with st.spinner("Thinking..."):
#                     response = chatbot.chat(user_input)
#                 st.markdown(response)
#             st.session_state.chat_history.append({"role": "assistant", "content": response})

# if __name__ == "__main__":
#     main()

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st
from streamlit.components.v1 import html


HERE = Path(__file__).resolve().parent
DEFAULT_HTML_PATH = "output/dashboard_multi_year.html"


def load_dashboard_html(html_path: Path) -> str:
    """
    Load the static Plotly dashboard HTML file.

    Raises:
        FileNotFoundError: If the HTML file does not exist.
    """
    if not html_path.exists():
        raise FileNotFoundError(
            f"Dashboard HTML not found at {html_path}. "
            "Run `python Baltimore_MetricsWithMap.py` to regenerate it."
        )

    return html_path.read_text(encoding="utf-8")


def main() -> None:
    st.set_page_config(
        page_title="Baltimore Health Dashboard",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🗺️ Baltimore City Health Dashboard")
    st.caption(
        "Streamlit wrapper for the interactive Plotly dashboard generated by "
        "`Baltimore_MetricsWithMap.py`."
    )

    st.sidebar.header("Dashboard Controls")
    html_path_str = st.sidebar.text_input(
        "Dashboard HTML Path",
        value=str(DEFAULT_HTML_PATH),
        help="Absolute path to the generated dashboard HTML file.",
    )
    html_path = Path(html_path_str).expanduser().resolve()

    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
    refresh_interval = st.sidebar.slider(
        "Refresh interval (seconds)",
        min_value=5,
        max_value=60,
        value=15,
        disabled=not auto_refresh,
    )

    refresh_button = st.sidebar.button("Reload Dashboard")

    if st.sidebar.button("Regenerate Instructions"):
        st.sidebar.info(
            "To rebuild the dashboard HTML, run:\n\n"
            "```bash\npython Baltimore_MetricsWithMap.py\n```"
        )

    # Simple auto-refresh mechanism
    if auto_refresh:
        placeholder = st.empty()
        refresh_countdown = refresh_interval
        placeholder.caption(f"Auto-refreshing in {refresh_countdown} seconds...")
        time.sleep(1)
        for remaining in range(refresh_interval - 1, -1, -1):
            placeholder.caption(f"Auto-refreshing in {remaining} seconds...")
            time.sleep(1)
        placeholder.empty()
        refresh_button = True

    # Load and display dashboard
    try:
        dashboard_html = load_dashboard_html(html_path)
        if refresh_button:
            st.success(f"Dashboard reloaded from {html_path}")

        html(
            dashboard_html,
            height=900,
            scrolling=True,
        )
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Failed to load dashboard HTML: {exc}")
        st.stop()


if __name__ == "__main__":
    main()
