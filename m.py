"""
NIRNAY (निर्णय) - Unified Market Analysis | A Pragyam Product Family Member
Quantitative Signal + Regime Intelligence System

Combines:
- MSF (Market Strength Factor) - Price structure analysis  
- MMR (Macro-Micro Regime) - Macro correlation analysis
- HMM (Hidden Markov Model) - Regime state detection
- GARCH - Volatility regime analysis
- CUSUM - Change point detection
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import requests
import io
import urllib3

# Import production core engine (single source of truth)
from nirnay_core import (
    MathUtils, MSFCalculator, MMRCalculator, DivergenceDetector,
    AdaptiveKalmanFilter, AdaptiveHMM, GARCHDetector, CUSUMDetector,
    NirnayEngine, MarketRegime, SignalType, VolatilityRegime,
    run_full_analysis as _core_run_full_analysis
)

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NIRNAY | Unified Market Analysis",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

VERSION = "v2.0.0"
PRODUCT_NAME = "Nirnay"
COMPANY = "Hemrek Capital"

# ══════════════════════════════════════════════════════════════════════════════
# PRAGYAM DESIGN SYSTEM CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-color: #FFC300;
        --primary-rgb: 255, 195, 0;
        --background-color: #0F0F0F;
        --secondary-background-color: #1A1A1A;
        --bg-card: #1A1A1A;
        --bg-elevated: #2A2A2A;
        --text-primary: #EAEAEA;
        --text-secondary: #EAEAEA;
        --text-muted: #888888;
        --border-color: #2A2A2A;
        --border-light: #3A3A3A;
        --success-green: #10b981;
        --danger-red: #ef4444;
        --warning-amber: #f59e0b;
        --info-cyan: #06b6d4;
        --neutral: #888888;
    }
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .main, [data-testid="stSidebar"] { background-color: var(--background-color); color: var(--text-primary); }
    .stApp > header { background-color: transparent; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 3.5rem; max-width: 90%; padding-left: 2rem; padding-right: 2rem; }
    
    /* Sidebar toggle button - always visible */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background-color: var(--secondary-background-color) !important;
        border: 2px solid var(--primary-color) !important;
        border-radius: 8px !important;
        padding: 10px !important;
        margin: 12px !important;
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.4) !important;
        z-index: 999999 !important;
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        width: 40px !important;
        height: 40px !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    [data-testid="collapsedControl"]:hover {
        background-color: rgba(var(--primary-rgb), 0.2) !important;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.6) !important;
        transform: scale(1.05);
    }
    
    [data-testid="collapsedControl"] svg {
        stroke: var(--primary-color) !important;
        width: 20px !important;
        height: 20px !important;
    }
    
    /* Also style the sidebar close button */
    [data-testid="stSidebar"] button[kind="header"] {
        background-color: transparent !important;
        border: none !important;
    }
    
    [data-testid="stSidebar"] button[kind="header"] svg {
        stroke: var(--primary-color) !important;
    }
    
    /* Ensure sidebar button is always on top */
    button[kind="header"] {
        z-index: 999999 !important;
    }
    
    .premium-header {
        background: var(--secondary-background-color);
        padding: 1.25rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.1);
        border: 1px solid var(--border-color);
        position: relative;
        overflow: hidden;
        margin-top: 1rem;
    }
    
    .premium-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .premium-header h1 { margin: 0; font-size: 2rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.50px; position: relative; }
    .premium-header .tagline { color: var(--text-muted); font-size: 0.9rem; margin-top: 0.25rem; font-weight: 400; position: relative; }
    .premium-header .product-badge { display: inline-block; background: rgba(var(--primary-rgb), 0.15); color: var(--primary-color); padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
    
    .metric-card {
        background-color: var(--bg-card);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
        margin-bottom: 0.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); border-color: var(--border-light); }
    .metric-card h4 { color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { color: var(--text-primary); font-size: 1.75rem; font-weight: 700; margin: 0; line-height: 1; }
    .metric-card .sub-metric { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; font-weight: 500; }
    .metric-card.success h2 { color: var(--success-green); }
    .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.warning h2 { color: var(--warning-amber); }
    .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.neutral h2 { color: var(--neutral); }
    .metric-card.primary h2 { color: var(--primary-color); }
    
    .signal-card {
        background-color: var(--bg-card);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .signal-card::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; }
    .signal-card.buy::before { background: var(--success-green); }
    .signal-card.sell::before { background: var(--danger-red); }
    .signal-card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; }
    .signal-card-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); }
    
    .status-badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .status-badge.buy { background: rgba(16, 185, 129, 0.15); color: var(--success-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-badge.sell { background: rgba(239, 68, 68, 0.15); color: var(--danger-red); border: 1px solid rgba(239, 68, 68, 0.3); }
    .status-badge.oversold { background: rgba(6, 182, 212, 0.15); color: var(--info-cyan); border: 1px solid rgba(6, 182, 212, 0.3); }
    .status-badge.overbought { background: rgba(245, 158, 11, 0.15); color: var(--warning-amber); border: 1px solid rgba(245, 158, 11, 0.3); }
    .status-badge.neutral { background: rgba(136, 136, 136, 0.15); color: var(--neutral); border: 1px solid rgba(136, 136, 136, 0.3); }
    .status-badge.divergence { background: rgba(var(--primary-rgb), 0.15); color: var(--primary-color); border: 1px solid rgba(var(--primary-rgb), 0.3); }
    
    .info-box { background: var(--secondary-background-color); border: 1px solid var(--border-color); border-left: 0px solid var(--primary-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }
    
    .stButton>button { border: 2px solid var(--primary-color); background: transparent; color: var(--primary-color); font-weight: 700; border-radius: 12px; padding: 0.75rem 2rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); text-transform: uppercase; letter-spacing: 0.5px; }
    .stButton>button:hover { box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6); background: var(--primary-color); color: #1A1A1A; transform: translateY(-2px); }
    .stButton>button:active { transform: translateY(0); }
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background: transparent; }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); border-bottom: 2px solid transparent; transition: color 0.3s, border-bottom 0.3s; background: transparent; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: var(--primary-color); border-bottom: 2px solid var(--primary-color); background: transparent !important; }
    
    .stPlotlyChart { border-radius: 12px; background-color: var(--secondary-background-color); padding: 10px; border: 1px solid var(--border-color); box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.1); }
    .stDataFrame { border-radius: 12px; background-color: var(--secondary-background-color); border: 1px solid var(--border-color); }
    .section-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%); margin: 1.5rem 0; }
    
    .symbol-row { display: flex; align-items: center; justify-content: space-between; padding: 0.75rem 1rem; border-radius: 8px; background: var(--bg-elevated); margin-bottom: 0.5rem; transition: all 0.2s ease; }
    .symbol-row:hover { background: var(--border-light); }
    .symbol-name { font-weight: 700; color: var(--text-primary); font-size: 0.9rem; }
    .symbol-price { color: var(--text-muted); font-size: 0.85rem; }
    .symbol-score { font-weight: 700; font-size: 0.9rem; }
    
    .conviction-meter { height: 8px; background: var(--bg-elevated); border-radius: 4px; overflow: hidden; margin-top: 0.5rem; }
    .conviction-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
    
    .sidebar-title { font-size: 0.75rem; font-weight: 700; color: var(--primary-color); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; }
    
    [data-testid="stSidebar"] { background: var(--secondary-background-color); border-right: 1px solid var(--border-color); }
    
    .stTextInput > div > div > input { background: var(--bg-elevated) !important; border: 1px solid var(--border-color) !important; border-radius: 8px !important; color: var(--text-primary) !important; }
    .stTextInput > div > div > input:focus { border-color: var(--primary-color) !important; box-shadow: 0 0 0 2px rgba(var(--primary-rgb), 0.2) !important; }
    
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--background-color); }
    ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-light); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & SYMBOLS
# ══════════════════════════════════════════════════════════════════════════════

SCREENER_SYMBOLS = [
    "SENSEXIETF.NS", "NIFTYIETF.NS", "MON100.NS", "MAKEINDIA.NS", "SILVERIETF.NS",
    "HEALTHIETF.NS", "CONSUMIETF.NS", "GOLDIETF.NS", "INFRAIETF.NS", "CPSEETF.NS",
    "TNIDETF.NS", "COMMOIETF.NS", "MODEFENCE.NS", "MOREALTY.NS", "PSUBNKIETF.NS",
    "MASPTOP50.NS", "FMCGIETF.NS", "BANKIETF.NS", "ITIETF.NS", "EVINDIA.NS",
    "MNC.NS", "FINIETF.NS", "AUTOIETF.NS", "PVTBANIETF.NS", "MONIFTY500.NS",
    "ECAPINSURE.NS", "MIDCAPIETF.NS", "MOSMALL250.NS", "OILIETF.NS", "METALIETF.NS"
]

SYMBOL_NAMES = {
    "SENSEXIETF.NS": "SENSEX", "NIFTYIETF.NS": "NIFTY 50", "MON100.NS": "NIFTY 100",
    "MAKEINDIA.NS": "Make India", "SILVERIETF.NS": "Silver", "HEALTHIETF.NS": "Healthcare",
    "CONSUMIETF.NS": "Consumer", "GOLDIETF.NS": "Gold", "INFRAIETF.NS": "Infra",
    "CPSEETF.NS": "CPSE", "TNIDETF.NS": "TN Index", "COMMOIETF.NS": "Commodities",
    "MODEFENCE.NS": "Defence", "MOREALTY.NS": "Realty", "PSUBNKIETF.NS": "PSU Bank",
    "MASPTOP50.NS": "Top 50", "FMCGIETF.NS": "FMCG", "BANKIETF.NS": "Banking",
    "ITIETF.NS": "IT", "EVINDIA.NS": "EV India", "MNC.NS": "MNC",
    "FINIETF.NS": "Financial", "AUTOIETF.NS": "Auto", "PVTBANIETF.NS": "Pvt Bank",
    "MONIFTY500.NS": "NIFTY 500", "ECAPINSURE.NS": "Insurance", "MIDCAPIETF.NS": "Midcap",
    "MOSMALL250.NS": "Smallcap", "OILIETF.NS": "Oil & Gas", "METALIETF.NS": "Metal"
}

MACRO_SYMBOLS_STOOQ = {
    "India 10Y": "10YINY.B", "India 02Y": "2YINY.B",
    "US 30Y": "30YUSY.B", "US 10Y": "10YUSY.B", "US 05Y": "5YUSY.B", "US 02Y": "2YUSY.B",
    "UK 30Y": "30YUKY.B", "UK 10Y": "10YUKY.B", "UK 05Y": "5YUKY.B", "UK 02Y": "2YUKY.B",
    "EU (DE) 30Y": "30YDEY.B", "EU (DE) 10Y": "10YDEY.B", "EU (DE) 05Y": "5YDEY.B", "EU (DE) 02Y": "2YDEY.B",
    "China 10Y": "10YCNY.B", "China 02Y": "2YCNY.B",
    "Japan 30Y": "30YJPY.B", "Japan 10Y": "10YJPY.B", "Japan 02Y": "2YJPY.B",
    "Singapore 10Y": "10YSGY.B",
}

MACRO_SYMBOLS_YF = {
    "Dollar Index": "DX-Y.NYB", "Crude Oil": "CL=F", "Brent Crude": "BZ=F",
    "USD/INR": "INR=X", "GBP/INR": "GBPINR=X", "EUR/INR": "EURINR=X",
    "SGD/INR": "SGDINR=X", "JPY/INR": "JPYINR=X", "Gold": "GC=F", "Silver": "SI=F"
}

MACRO_SYMBOLS = {**MACRO_SYMBOLS_STOOQ, **MACRO_SYMBOLS_YF}

# ══════════════════════════════════════════════════════════════════════════════
# SPREAD SCREENER CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

INDEX_LIST = [
    "NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY 200", "NIFTY 500",
    "NIFTY MIDCAP 50", "NIFTY MIDCAP 100", "NIFTY SMLCAP 100", "NIFTY BANK",
    "NIFTY AUTO", "NIFTY FIN SERVICE", "NIFTY FMCG", "NIFTY IT",
    "NIFTY MEDIA", "NIFTY METAL", "NIFTY PHARMA",
    "S&P 500", "DOW JONES", "NASDAQ 100"
]

US_INDEX_LIST = ["S&P 500", "DOW JONES", "NASDAQ 100"]

BASE_URL = "https://www.niftyindices.com/IndexConstituent/"
INDEX_URL_MAP = {
    "NIFTY 50": f"{BASE_URL}ind_nifty50list.csv",
    "NIFTY NEXT 50": f"{BASE_URL}ind_niftynext50list.csv",
    "NIFTY 100": f"{BASE_URL}ind_nifty100list.csv",
    "NIFTY 200": f"{BASE_URL}ind_nifty200list.csv",
    "NIFTY 500": f"{BASE_URL}ind_nifty500list.csv",
    "NIFTY MIDCAP 50": f"{BASE_URL}ind_niftymidcap50list.csv",
    "NIFTY MIDCAP 100": f"{BASE_URL}ind_niftymidcap100list.csv",
    "NIFTY SMLCAP 100": f"{BASE_URL}ind_niftysmallcap100list.csv",
    "NIFTY BANK": f"{BASE_URL}ind_niftybanklist.csv",
    "NIFTY AUTO": f"{BASE_URL}ind_niftyautolist.csv",
    "NIFTY FIN SERVICE": f"{BASE_URL}ind_niftyfinancelist.csv",
    "NIFTY FMCG": f"{BASE_URL}ind_niftyfmcglist.csv",
    "NIFTY IT": f"{BASE_URL}ind_niftyitlist.csv",
    "NIFTY MEDIA": f"{BASE_URL}ind_niftymedialist.csv",
    "NIFTY METAL": f"{BASE_URL}ind_niftymetallist.csv",
    "NIFTY PHARMA": f"{BASE_URL}ind_niftypharmalist.csv"
}

ANALYSIS_UNIVERSE_OPTIONS = ["F&O Stocks", "Index Constituents"]

def get_display_name(symbol):
    return SYMBOL_NAMES.get(symbol, symbol.replace(".NS", ""))

def get_currency_symbol(symbol):
    """Return appropriate currency symbol based on ticker suffix"""
    if isinstance(symbol, str) and symbol.endswith('.NS'):
        return '₹'
    return '$'  # Default to USD for international/US tickers

# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE SELECTION FUNCTIONS (for Spread Screener)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def get_fno_stock_list():
    """Fetch F&O stock list from NSE"""
    # Try nsetools library first
    try:
        from nsetools import Nse
        nse = Nse()
        stock_data = nse.get_advances_declines()
        if isinstance(stock_data, list):
            stock_data = pd.DataFrame(stock_data)
        if not isinstance(stock_data, pd.DataFrame):
            raise ValueError(f"Unexpected type: {type(stock_data)}")
    except ImportError:
        # Fallback: fetch F&O lot sizes CSV from NSE (contains all F&O symbols)
        try:
            url = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, verify=False, timeout=15)
            if response.status_code == 200 and len(response.text) > 100:
                stock_data = pd.read_csv(io.StringIO(response.text))
                # Column is typically "SYMBOL" or second column
                if 'SYMBOL' in stock_data.columns:
                    symbols = stock_data['SYMBOL'].str.strip().dropna().unique().tolist()
                else:
                    # Try second column (common format)
                    symbols = stock_data.iloc[:, 1].str.strip().dropna().unique().tolist()
                symbols = [s for s in symbols if s and str(s).strip() and s.upper() != 'SYMBOL']
                symbols_ns = [str(s) + ".NS" for s in symbols if s and str(s).strip()]
                if symbols_ns:
                    return symbols_ns, f"✓ Fetched {len(symbols_ns)} F&O securities (from NSE archives)"
            return None, "Could not fetch F&O list from NSE archives"
        except Exception as e2:
            return None, f"nsetools not installed and NSE archive fallback failed: {e2}"
    except Exception as e:
        return None, f"Error: {e}"
    
    try:
        if not isinstance(stock_data, pd.DataFrame):
            return None, f"API returned unexpected type: {type(stock_data)}"
        
        symbols = None
        if 'SYMBOL' in stock_data.columns:
            symbols = stock_data['SYMBOL'].tolist()
        elif 'symbol' in stock_data.columns:
            symbols = stock_data['symbol'].tolist()
        elif stock_data.index.name in ['SYMBOL', 'symbol']:
            symbols = stock_data.index.tolist()
        else:
            if isinstance(stock_data.index, pd.RangeIndex):
                return None, f"Could not find SYMBOL column"
            elif len(stock_data.index) > 0:
                symbols = stock_data.index.tolist()

        if symbols is None:
             return None, f"Could not extract symbols"
            
        symbols_ns = [str(s) + ".NS" for s in symbols if s and str(s).strip()]
        
        if not symbols_ns:
            return None, "Symbol list empty after cleaning"

        return symbols_ns, f"✓ Fetched {len(symbols_ns)} F&O securities"
            
    except Exception as e:
        return None, f"Error: {e}"


@st.cache_data(ttl=3600, show_spinner=False)
def get_index_stock_list(index):
    """Fetch index constituents from NSE Indices or US Indices"""
    # Route US indices to separate handler
    if index in US_INDEX_LIST:
        return get_us_index_stock_list(index)
    
    url = INDEX_URL_MAP.get(index)
    if not url:
        return None, f"No URL for {index}"
        
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
        
        csv_file = io.StringIO(response.text)
        stock_df = pd.read_csv(csv_file)
        
        if 'Symbol' in stock_df.columns:
            symbols = stock_df['Symbol'].tolist()
            symbols_ns = [str(s) + ".NS" for s in symbols if s and str(s).strip()]
            return symbols_ns, f"✓ Fetched {len(symbols_ns)} constituents"
        else:
            return None, f"No Symbol column found"
            
    except Exception as e:
        return None, f"Error: {e}"


# Hardcoded Dow Jones 30 components (updated periodically — small stable list)
DOW_JONES_TICKERS = [
    "AMZN", "AMGN", "AAPL", "BA", "CAT", "CSCO", "CVX", "GS", "HD", "HON",
    "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG",
    "CRM", "SHW", "TRV", "UNH", "V", "VZ", "WMT", "DIS", "DOW", "NVDA"
]


@st.cache_data(ttl=3600, show_spinner=False)
def get_us_index_stock_list(index):
    """Fetch US index constituents from Wikipedia with hardcoded fallback for Dow Jones"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        if index == "S&P 500":
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            tables = pd.read_html(io.StringIO(response.text))
            if tables:
                df = tables[0]
                if 'Symbol' in df.columns:
                    symbols = df['Symbol'].str.strip().str.replace('.', '-', regex=False).tolist()
                    symbols = [s for s in symbols if s and str(s).strip()]
                    return symbols, f"✓ Fetched {len(symbols)} S&P 500 constituents"
            return None, "Could not parse S&P 500 table"
            
        elif index == "DOW JONES":
            # Dow Jones is only 30 stocks — use hardcoded list with Wikipedia refresh attempt
            try:
                url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                tables = pd.read_html(io.StringIO(response.text))
                # Find the table with ticker symbols (usually has 'Symbol' or 'Ticker' column)
                for tbl in tables:
                    for col in ['Symbol', 'Ticker', 'Ticker symbol']:
                        if col in tbl.columns:
                            symbols = tbl[col].str.strip().tolist()
                            symbols = [s for s in symbols if s and str(s).strip() and len(s) <= 5]
                            if 20 <= len(symbols) <= 35:
                                return symbols, f"✓ Fetched {len(symbols)} Dow Jones constituents"
            except Exception:
                pass
            # Fallback to hardcoded list
            return DOW_JONES_TICKERS.copy(), f"✓ Loaded {len(DOW_JONES_TICKERS)} Dow Jones constituents"
            
        elif index == "NASDAQ 100":
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            tables = pd.read_html(io.StringIO(response.text))
            for tbl in tables:
                if 'Ticker' in tbl.columns:
                    symbols = tbl['Ticker'].str.strip().str.replace('.', '-', regex=False).tolist()
                    symbols = [s for s in symbols if s and str(s).strip()]
                    if len(symbols) >= 90:
                        return symbols, f"✓ Fetched {len(symbols)} NASDAQ 100 constituents"
                elif 'Symbol' in tbl.columns:
                    symbols = tbl['Symbol'].str.strip().str.replace('.', '-', regex=False).tolist()
                    symbols = [s for s in symbols if s and str(s).strip()]
                    if len(symbols) >= 90:
                        return symbols, f"✓ Fetched {len(symbols)} NASDAQ 100 constituents"
            return None, "Could not parse NASDAQ 100 table"
        
        return None, f"Unknown US index: {index}"
        
    except Exception as e:
        return None, f"Error fetching {index}: {e}"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_batch_data(stock_list, end_date=None, days_back=100, include_live=True):
    """Batch download for spread screener with optional live data for current day"""
    if end_date is None:
        end_date = datetime.date.today()
    
    # Add buffer for end date to ensure we get the requested date
    download_end = end_date + datetime.timedelta(days=5)
    start_date = end_date - datetime.timedelta(days=days_back + 365)
    
    try:
        all_data = yf.download(
            stock_list,
            start=start_date,
            end=download_end,
            progress=False,
            auto_adjust=True,
            group_by='ticker'
        )
        
        if all_data.empty:
            return None, "No data returned"
            
        if isinstance(all_data, pd.DataFrame) and isinstance(all_data.columns, pd.MultiIndex):
            data_dict = {}
            for ticker in stock_list:
                try:
                    ticker_df = all_data.xs(ticker, level=0, axis=1)
                    if not ticker_df.empty and not ticker_df['Close'].isnull().all():
                        data_dict[ticker] = ticker_df.copy()
                except KeyError:
                    pass

        elif isinstance(all_data, dict):
            data_dict = {t:df.copy() for t,df in all_data.items() if not df.empty and not df['Close'].isnull().all()}

        else:
             return None, "Unexpected data structure"
        
        # Fetch live data for today if requested and end_date is today
        if include_live and end_date == datetime.date.today() and data_dict:
            today_ts = pd.Timestamp(datetime.date.today())
            
            # Check if today's data is missing from at least one ticker
            sample_df = list(data_dict.values())[0]
            sample_df.index = pd.to_datetime(sample_df.index)
            if sample_df.index.tz is not None:
                sample_df.index = sample_df.index.tz_localize(None)
            
            has_today = any(idx.date() == datetime.date.today() for idx in sample_df.index)
            
            if not has_today:
                # Fetch live data for all tickers
                try:
                    live_data = yf.download(
                        list(data_dict.keys()),
                        period="1d",
                        progress=False,
                        auto_adjust=True,
                        group_by='ticker'
                    )
                    
                    if not live_data.empty:
                        if isinstance(live_data, pd.DataFrame) and isinstance(live_data.columns, pd.MultiIndex):
                            for ticker in data_dict.keys():
                                try:
                                    live_ticker = live_data.xs(ticker, level=0, axis=1)
                                    if not live_ticker.empty and not live_ticker['Close'].isnull().all():
                                        # Append live data to historical
                                        hist_df = data_dict[ticker]
                                        hist_df.index = pd.to_datetime(hist_df.index)
                                        if hist_df.index.tz is not None:
                                            hist_df.index = hist_df.index.tz_localize(None)
                                        
                                        live_ticker.index = pd.to_datetime(live_ticker.index)
                                        if live_ticker.index.tz is not None:
                                            live_ticker.index = live_ticker.index.tz_localize(None)
                                        
                                        # Only append if not already present
                                        new_dates = live_ticker.index.difference(hist_df.index)
                                        if len(new_dates) > 0:
                                            data_dict[ticker] = pd.concat([hist_df, live_ticker.loc[new_dates]]).sort_index()
                                except KeyError:
                                    pass
                        
                        return data_dict, f"✓ Downloaded {len(data_dict)} tickers (with live data)"
                except Exception:
                    pass  # Fall through to return historical data only
            
        return data_dict, f"✓ Downloaded {len(data_dict)} tickers"

    except Exception as e:
        return None, f"Download error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING (Macro + Individual Tickers)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_stooq_symbol(symbol, start_date, end_date):
    """Fetch single symbol from Stooq via direct HTTP request"""
    try:
        url = f"https://stooq.com/q/d/l/?s={symbol}&d1={start_date.strftime('%Y%m%d')}&d2={end_date.strftime('%Y%m%d')}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and len(response.text) > 50:
            df = pd.read_csv(io.StringIO(response.text))
            if 'Date' in df.columns and 'Close' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
                return df['Close']
    except Exception:
        pass
    return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_macro_data(days_back=100):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_back + 365)

    # Fetch from Stooq via direct HTTP requests
    stooq_df = pd.DataFrame()
    for name, symbol in MACRO_SYMBOLS_STOOQ.items():
        series = fetch_stooq_symbol(symbol, start_date, end_date)
        if series is not None and len(series) > 0:
            stooq_df[symbol] = series

    if not stooq_df.empty:
        stooq_df = stooq_df.sort_index()

    yf_df = pd.DataFrame()
    try:
        yf_tickers = list(MACRO_SYMBOLS_YF.values())
        yf_raw = yf.download(yf_tickers, start=start_date, end=end_date, progress=False)
        if not yf_raw.empty:
            if isinstance(yf_raw.columns, pd.MultiIndex):
                if 'Close' in yf_raw.columns.get_level_values(0):
                    yf_df = yf_raw['Close']
                elif 'Adj Close' in yf_raw.columns.get_level_values(0):
                    yf_df = yf_raw['Adj Close']
            else:
                if 'Close' in yf_raw.columns:
                    yf_df = yf_raw[['Close']]
                else:
                    yf_df = yf_raw
            if yf_df.index.tz is not None:
                yf_df.index = yf_df.index.tz_localize(None)
            yf_df = yf_df.sort_index()

            # Fetch live data for today if missing
            has_today = any(idx.date() == datetime.date.today() for idx in yf_df.index)
            if not has_today:
                try:
                    live_yf = yf.download(yf_tickers, period="1d", progress=False)
                    if not live_yf.empty:
                        if isinstance(live_yf.columns, pd.MultiIndex):
                            if 'Close' in live_yf.columns.get_level_values(0):
                                live_yf = live_yf['Close']
                            elif 'Adj Close' in live_yf.columns.get_level_values(0):
                                live_yf = live_yf['Adj Close']
                        if live_yf.index.tz is not None:
                            live_yf.index = live_yf.index.tz_localize(None)
                        new_dates = live_yf.index.difference(yf_df.index)
                        if len(new_dates) > 0:
                            yf_df = pd.concat([yf_df, live_yf.loc[new_dates]]).sort_index()
                except Exception:
                    pass
    except Exception:
        pass

    if not stooq_df.empty and not yf_df.empty:
        combined_macro = pd.concat([stooq_df, yf_df], axis=1).sort_index()
    elif not stooq_df.empty:
        combined_macro = stooq_df
    elif not yf_df.empty:
        combined_macro = yf_df
    else:
        return pd.DataFrame()
    return combined_macro.ffill()


def fetch_ticker_data(target_ticker, macro_df, days_back=100, include_live=True):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_back + 365)
    try:
        target_df = yf.download(target_ticker, start=start_date, end=end_date, progress=False)
        if target_df.empty:
            return None
        if isinstance(target_df.columns, pd.MultiIndex):
            target_df.columns = target_df.columns.get_level_values(0)
        target_df = target_df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
        if target_df.index.tz is not None:
            target_df.index = target_df.index.tz_localize(None)

        if include_live:
            has_today = any(idx.date() == datetime.date.today() for idx in target_df.index)
            if not has_today:
                try:
                    live_df = yf.download(target_ticker, period="1d", progress=False)
                    if not live_df.empty:
                        if isinstance(live_df.columns, pd.MultiIndex):
                            live_df.columns = live_df.columns.get_level_values(0)
                        live_df = live_df[['Open', 'High', 'Low', 'Close', 'Volume']]
                        if live_df.index.tz is not None:
                            live_df.index = live_df.index.tz_localize(None)
                        new_dates = live_df.index.difference(target_df.index)
                        if len(new_dates) > 0:
                            target_df = pd.concat([target_df, live_df.loc[new_dates]]).sort_index()
                except Exception:
                    pass

        combined = target_df.join(macro_df, how='left').ffill()
        return combined
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL ENGINE (delegates to nirnay_core v2.0)
# ══════════════════════════════════════════════════════════════════════════════
# All mathematical logic lives in nirnay_core.py (single source of truth).
# These are thin wrappers for backward compatibility with the UI code.

# Re-export core utilities for any direct usage in app
sigmoid = MathUtils.sigmoid
zscore_clipped = MathUtils.zscore_clipped
calculate_atr = MathUtils.calculate_atr


def calculate_msf(df, length=20, roc_len=14, clip=3.0):
    """Wrapper: delegates to MSFCalculator in nirnay_core"""
    calc = MSFCalculator(length, roc_len)
    return calc.calculate(df)


def calculate_mmr(df, length=20, num_vars=5):
    """Wrapper: delegates to MMRCalculator in nirnay_core"""
    available_macros = [v for v in MACRO_SYMBOLS.values() if v in df.columns]
    calc = MMRCalculator(length, num_vars)
    return calc.calculate(df, available_macros)


def run_full_analysis(df, length, roc_len, regime_sensitivity, base_weight):
    """Wrapper: delegates to nirnay_core.run_full_analysis, enriches driver names"""
    result_df, drivers = _core_run_full_analysis(df, length, roc_len, regime_sensitivity, base_weight, MACRO_SYMBOLS)
    # Core returns {"Symbol": ticker, "Correlation": val} — enrich with human-readable "Name"
    reverse_macro = {v: k for k, v in MACRO_SYMBOLS.items()}
    for d in drivers:
        d["Name"] = reverse_macro.get(d["Symbol"], d["Symbol"])
    return result_df, drivers


# ══════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_price_chart(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color='#10b981', decreasing_line_color='#ef4444',
        increasing_fillcolor='rgba(16,185,129,0.3)', decreasing_fillcolor='rgba(239,68,68,0.3)', name='Price'
    ))
    ma20 = df['Close'].rolling(20).mean()
    fig.add_trace(go.Scatter(x=df.index, y=ma20, mode='lines', name='MA20', line=dict(color='#FFC300', width=1.5), opacity=0.8))
    ma50 = df['Close'].rolling(50).mean()
    fig.add_trace(go.Scatter(x=df.index, y=ma50, mode='lines', name='MA50', line=dict(color='#06b6d4', width=1.5), opacity=0.8))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=380,
        margin=dict(l=10, r=10, t=30, b=50),
        xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', side='right'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)', font=dict(size=10, color='#888888')),
        font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified'
    )
    return fig


def create_oscillator_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Unified_Osc'].clip(lower=0), fill='tozeroy', fillcolor='rgba(239,68,68,0.15)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Unified_Osc'].clip(upper=0), fill='tozeroy', fillcolor='rgba(16,185,129,0.15)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    
    trace_colors = np.where(df['Unified_Osc'] < -5, '#10b981', np.where(df['Unified_Osc'] > 5, '#ef4444', '#888888'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Unified_Osc'], mode='lines+markers', name='Unified Signal', line=dict(color='#EAEAEA', width=2), marker=dict(color=trace_colors, size=6, line=dict(width=0))))
    fig.add_trace(go.Scatter(x=df.index, y=df['MSF_Osc'], mode='lines', name='MSF (Internal)', line=dict(color='#FFC300', width=1.5, dash='dot'), opacity=0.6))
    fig.add_trace(go.Scatter(x=df.index, y=df['MMR_Osc'], mode='lines', name='MMR (Macro)', line=dict(color='#06b6d4', width=1.5, dash='dot'), opacity=0.6))
    
    buys = df[df['Buy_Signal']]
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys.index, y=buys['Unified_Osc'], mode='markers', name='Buy Signal', marker=dict(symbol='circle', color='#10b981', size=14, line=dict(color='white', width=2))))
    sells = df[df['Sell_Signal']]
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells.index, y=sells['Unified_Osc'], mode='markers', name='Sell Signal', marker=dict(symbol='circle', color='#ef4444', size=14, line=dict(color='white', width=2))))
    
    fig.add_hline(y=5, line=dict(color='rgba(239,68,68,0.5)', width=1, dash='dash'))
    fig.add_hline(y=-5, line=dict(color='rgba(16,185,129,0.5)', width=1, dash='dash'))
    fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.2)', width=1))
    fig.add_hrect(y0=5, y1=10, fillcolor='rgba(239,68,68,0.08)', line_width=0)
    fig.add_hrect(y0=-10, y1=-5, fillcolor='rgba(16,185,129,0.08)', line_width=0)
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=320,
        margin=dict(l=10, r=10, t=10, b=50),
        xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', range=[-12, 12], tickvals=[-10, -5, 0, 5, 10], side='right'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)', font=dict(size=10, color='#888888')),
        font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified'
    )
    return fig


def create_gauge_chart(value):
    color = '#10b981' if value < -5 else '#ef4444' if value > 5 else '#888888'
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number=dict(font=dict(size=32, color=color, family='Inter'), suffix=""),
        gauge=dict(
            axis=dict(range=[-10, 10], tickwidth=1, tickcolor='#3A3A3A', tickvals=[-10, -5, 0, 5, 10], tickfont=dict(size=10, color='#888888')),
            bar=dict(color=color, thickness=0.3), bgcolor='#1A1A1A', borderwidth=2, bordercolor='#2A2A2A',
            steps=[dict(range=[-10, -5], color='rgba(16,185,129,0.15)'), dict(range=[-5, 5], color='rgba(136,136,136,0.1)'), dict(range=[5, 10], color='rgba(239,68,68,0.15)')],
            threshold=dict(line=dict(color='white', width=2), thickness=0.8, value=value)
        )
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=200, margin=dict(l=20, r=20, t=30, b=20), font=dict(family='Inter', color='#EAEAEA'))
    return fig


def create_heatmap_chart(results_df):
    symbols = results_df['DisplayName'].tolist()
    scores = results_df['Signal'].tolist()
    n_cols = 6
    n_rows = int(np.ceil(len(symbols) / n_cols))
    while len(symbols) < n_cols * n_rows:
        symbols.append("")
        scores.append(0)
    symbols_grid = np.array(symbols).reshape(n_rows, n_cols)
    scores_grid = np.array(scores).reshape(n_rows, n_cols)
    colorscale = [[0, '#10b981'], [0.25, '#059669'], [0.5, '#1A1A1A'], [0.75, '#dc2626'], [1, '#ef4444']]
    normalized_scores = (scores_grid + 10) / 20
    
    fig = go.Figure(data=go.Heatmap(
        z=normalized_scores,
        text=[[f"{s}<br>{v:.1f}" if s else "" for s, v in zip(row_s, row_v)] for row_s, row_v in zip(symbols_grid, scores_grid)],
        texttemplate="%{text}", textfont=dict(size=11, color='white', family='Inter'),
        colorscale=colorscale, showscale=False, hovertemplate="<b>%{text}</b><extra></extra>", xgap=3, ygap=3
    ))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300,
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, autorange='reversed'),
        font=dict(family='Inter')
    )
    return fig


def create_distribution_chart(results_df):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=results_df['Signal'], nbinsx=20, marker=dict(color='#FFC300', line=dict(color='#2A2A2A', width=1)), opacity=0.8))
    fig.add_vline(x=-5, line=dict(color='#10b981', width=2, dash='dash'))
    fig.add_vline(x=5, line=dict(color='#ef4444', width=2, dash='dash'))
    fig.add_vline(x=0, line=dict(color='#888888', width=1))
    fig.add_vrect(x0=-10, x1=-5, fillcolor='rgba(16,185,129,0.1)', line_width=0)
    fig.add_vrect(x0=5, x1=10, fillcolor='rgba(239,68,68,0.1)', line_width=0)
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=200,
        margin=dict(l=0, r=0, t=10, b=30),
        xaxis=dict(title=dict(text='Signal Value', font=dict(size=10, color='#888888')), showgrid=True, gridcolor='rgba(42,42,42,0.5)', range=[-12, 12]),
        yaxis=dict(title=dict(text='Count', font=dict(size=10, color='#888888')), showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
        font=dict(family='Inter', color='#EAEAEA'), bargap=0.1
    )
    return fig


def create_sector_radar(results_df):
    sectors = {
        'Index': ['SENSEX', 'NIFTY 50', 'NIFTY 100', 'NIFTY 500', 'Top 50', 'Midcap', 'Smallcap'],
        'Banking': ['Banking', 'Pvt Bank', 'PSU Bank', 'Financial', 'Insurance'],
        'Commodities': ['Gold', 'Silver', 'Oil & Gas', 'Metal', 'Commodities'],
        'Defensive': ['Healthcare', 'Consumer', 'FMCG'],
        'Cyclical': ['Auto', 'Infra', 'Realty', 'IT'],
        'Thematic': ['Defence', 'EV India', 'Make India', 'MNC', 'CPSE']
    }
    sector_scores = {}
    for sector, symbols in sectors.items():
        matching = results_df[results_df['DisplayName'].isin(symbols)]
        sector_scores[sector] = matching['Signal'].mean() if not matching.empty else 0
    
    categories = list(sector_scores.keys())
    values = list(sector_scores.values())
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', fillcolor='rgba(255,195,0,0.2)', line=dict(color='#FFC300', width=2), marker=dict(size=8, color='#FFC300')))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-10, 10], tickvals=[-10, -5, 0, 5, 10], gridcolor='rgba(42,42,42,0.5)', linecolor='rgba(42,42,42,0.5)', tickfont=dict(size=9, color='#888888')),
            angularaxis=dict(gridcolor='rgba(42,42,42,0.5)', linecolor='rgba(42,42,42,0.5)', tickfont=dict(size=10, color='#EAEAEA')),
            bgcolor='#1A1A1A'
        ),
        paper_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=60, r=60, t=30, b=30), font=dict(family='Inter', color='#EAEAEA'), showlegend=False
    )
    return fig


def create_scatter_matrix(results_df):
    fig = go.Figure()
    colors = results_df['Zone'].map({'Oversold': '#10b981', 'Overbought': '#ef4444', 'Neutral': '#888888'})
    fig.add_trace(go.Scatter(
        x=results_df['MSF'], y=results_df['MMR'], mode='markers+text',
        marker=dict(size=12, color=colors, line=dict(color='#2A2A2A', width=1), opacity=0.8),
        text=results_df['DisplayName'], textposition='top center', textfont=dict(size=8, color='#888888'),
        hovertemplate="<b>%{text}</b><br>MSF: %{x:.2f}<br>MMR: %{y:.2f}<extra></extra>"
    ))
    fig.add_hline(y=0, line=dict(color='rgba(255,195,0,0.3)', width=1, dash='dash'))
    fig.add_vline(x=0, line=dict(color='rgba(255,195,0,0.3)', width=1, dash='dash'))
    fig.add_annotation(x=6, y=6, text="SELL ZONE", font=dict(size=10, color='rgba(239,68,68,0.5)'), showarrow=False)
    fig.add_annotation(x=-6, y=-6, text="BUY ZONE", font=dict(size=10, color='rgba(16,185,129,0.5)'), showarrow=False)
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=350,
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis=dict(title=dict(text='MSF (Internal)', font=dict(size=11, color='#888888')), showgrid=True, gridcolor='rgba(42,42,42,0.5)', range=[-12, 12], zeroline=False),
        yaxis=dict(title=dict(text='MMR (Macro)', font=dict(size=11, color='#888888')), showgrid=True, gridcolor='rgba(42,42,42,0.5)', range=[-12, 12], zeroline=False),
        font=dict(family='Inter', color='#EAEAEA')
    )
    return fig


def create_ranking_chart(results_df, top_n=10):
    sorted_df = results_df.sort_values('Signal')
    bottom = sorted_df.head(top_n//2)
    top = sorted_df.tail(top_n//2)
    combined = pd.concat([bottom, top])
    colors = ['#10b981' if v < 0 else '#ef4444' for v in combined['Signal']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=combined['DisplayName'], x=combined['Signal'], orientation='h',
        marker=dict(color=colors, line=dict(color='#2A2A2A', width=1)),
        text=[f"{v:.1f}" for v in combined['Signal']], textposition='outside', textfont=dict(size=10, color='#888888')
    ))
    fig.add_vline(x=0, line=dict(color='#FFC300', width=1))
    fig.add_vline(x=-5, line=dict(color='rgba(16,185,129,0.5)', width=1, dash='dash'))
    fig.add_vline(x=5, line=dict(color='rgba(239,68,68,0.5)', width=1, dash='dash'))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=300,
        margin=dict(l=80, r=50, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', range=[-12, 12], tickvals=[-10, -5, 0, 5, 10]),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        font=dict(family='Inter', color='#EAEAEA')
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS & MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown("""
    <div class="premium-header">
        <h1>NIRNAY : Unified Market Analysis</h1>
        <div class="tagline">Quantitative Signal + Regime Intelligence System</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">NIRNAY</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">निर्णय | Decision Intelligence</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        mode = st.radio("Analysis Mode", ["🏠 Home", "🏦 ETF Screener", "📊 Market Screener", "📈 Chart Analysis"], label_visibility="collapsed")
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # ETF Screener specific options (fixed ETF universe)
        etf_mode = None
        etf_date = None
        etf_start_date = None
        etf_end_date = None
        
        if "ETF" in mode:
            st.markdown('<div class="sidebar-title">📊 Analysis Type</div>', unsafe_allow_html=True)
            etf_mode = st.radio(
                "Select ETF Mode",
                ["📅 Single Day", "📈 Time Series"],
                label_visibility="collapsed",
                help="Single Day: Analyze one date | Time Series: Track signals over a date range"
            )
            
            if "Single" in etf_mode:
                st.markdown('<div class="sidebar-title">📅 Analysis Date</div>', unsafe_allow_html=True)
                etf_date = st.date_input(
                    "ETF Analysis Date",
                    datetime.date.today(),
                    max_value=datetime.date.today(),
                    help="Select the date for signal analysis (defaults to today)"
                )
            else:
                st.markdown('<div class="sidebar-title">📅 Date Range</div>', unsafe_allow_html=True)
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    etf_start_date = st.date_input(
                        "ETF Start Date",
                        datetime.date.today() - datetime.timedelta(days=30),
                        max_value=datetime.date.today(),
                        help="Start of analysis period"
                    )
                with col_e2:
                    etf_end_date = st.date_input(
                        "ETF End Date",
                        datetime.date.today(),
                        max_value=datetime.date.today(),
                        help="End of analysis period"
                    )
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Market Screener specific options (F&O / Index universe)
        spread_universe = None
        spread_index = None
        spread_date = None
        spread_mode = None
        spread_start_date = None
        spread_end_date = None
        
        if "Market" in mode:
            st.markdown('<div class="sidebar-title">🎯 Universe Selection</div>', unsafe_allow_html=True)
            spread_universe = st.selectbox(
                "Analysis Universe",
                ANALYSIS_UNIVERSE_OPTIONS,
                help="Choose between F&O stocks or specific index constituents"
            )
            if spread_universe == "Index Constituents":
                spread_index = st.selectbox(
                    "Select Index",
                    INDEX_LIST,
                    index=INDEX_LIST.index("NIFTY 500"),
                    help="Select the index for constituent analysis"
                )
            
            st.markdown('<div class="sidebar-title">📊 Analysis Type</div>', unsafe_allow_html=True)
            spread_mode = st.radio(
                "Select Mode",
                ["📅 Single Day", "📈 Time Series"],
                label_visibility="collapsed",
                help="Single Day: Analyze one date | Time Series: Track signals over a date range"
            )
            
            if "Single" in spread_mode:
                st.markdown('<div class="sidebar-title">📅 Analysis Date</div>', unsafe_allow_html=True)
                spread_date = st.date_input(
                    "Select Date",
                    datetime.date.today(),
                    max_value=datetime.date.today(),
                    help="Select the date for signal analysis (defaults to today)"
                )
            else:
                st.markdown('<div class="sidebar-title">📅 Date Range</div>', unsafe_allow_html=True)
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    spread_start_date = st.date_input(
                        "Start Date",
                        datetime.date.today() - datetime.timedelta(days=30),
                        max_value=datetime.date.today(),
                        help="Start of analysis period"
                    )
                with col_d2:
                    spread_end_date = st.date_input(
                        "End Date",
                        datetime.date.today(),
                        max_value=datetime.date.today(),
                        help="End of analysis period"
                    )
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-title">⚙️ Parameters</div>', unsafe_allow_html=True)
        with st.expander("Indicator Settings", expanded=False):
            length = st.slider("Lookback Period", 10, 50, 20)
            roc_len = st.slider("ROC Length", 5, 30, 14)
            regime_sensitivity = st.slider("Regime Sensitivity", 0.5, 3.0, 1.5, 0.1)
            base_weight = st.slider("Base MSF Weight", 0.0, 1.0, 0.5, 0.05)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> MSF + MMR + HMM<br>
                <strong>Data:</strong> Live Market Feed
            </p>
        </div>
        """, unsafe_allow_html=True)
        return mode, length, roc_len, regime_sensitivity, base_weight, spread_universe, spread_index, spread_date, spread_mode, spread_start_date, spread_end_date, etf_mode, etf_date, etf_start_date, etf_end_date


def run_home_page():
    """Landing page with overview and quick navigation"""
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card primary' style='min-height: 280px;'>
            <h3 style='color: var(--primary-color); margin-bottom: 1rem;'>🏦 ETF Screener</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Full MSF + MMR + Regime analysis across a curated universe of 30 ETFs covering major indices and sectors.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Features:</strong><br>
                • Single Day Analysis<br>
                • Time Series Tracking<br>
                • Macro Correlation (MMR)<br>
                • HMM Regime Detection
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card success' style='min-height: 280px;'>
            <h3 style='color: var(--success-green); margin-bottom: 1rem;'>📊 Market Screener</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                MSF-based signal analysis across F&O stocks or index constituents. Scan 200-500 stocks efficiently.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Features:</strong><br>
                • F&O Stocks Universe<br>
                • 19 Index Constituents (incl. S&amp;P 500, Dow Jones, NASDAQ 100)<br>
                • Time Series Analysis<br>
                • Volatility Regime (GARCH)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card info' style='min-height: 280px;'>
            <h3 style='color: var(--info-cyan); margin-bottom: 1rem;'>📈 Chart Analysis</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Deep dive into individual securities with unified analysis including regime context and macro drivers.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Features:</strong><br>
                • Any NSE Symbol<br>
                • Price & Oscillator Charts<br>
                • HMM State Probabilities<br>
                • CUSUM Change Detection
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Analysis methodology section
    st.markdown("### 📐 Analysis Methodology")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.markdown("""
        <div class='signal-card buy' style='padding: 1.5rem;'>
            <h4 style='color: var(--success-green); margin-bottom: 1rem;'>MSF - Market Structure & Flow</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem; line-height: 1.7;'>
                Internal price-based indicator:
            </p>
            <ul style='color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8; margin-top: 0.5rem;'>
                <li><strong>Momentum</strong> - ROC dynamics</li>
                <li><strong>Microstructure</strong> - Efficiency</li>
                <li><strong>Flow</strong> - Volume-weighted</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown("""
        <div class='signal-card sell' style='padding: 1.5rem;'>
            <h4 style='color: var(--danger-red); margin-bottom: 1rem;'>MMR - Macro Regression</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem; line-height: 1.7;'>
                External macro correlation:
            </p>
            <ul style='color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8; margin-top: 0.5rem;'>
                <li><strong>Bonds</strong> - US/IN 10Y yields</li>
                <li><strong>Currencies</strong> - DXY, INR</li>
                <li><strong>Commodities</strong> - Gold, Crude</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        st.markdown("""
        <div class='metric-card info' style='padding: 1.5rem;'>
            <h4 style='color: var(--info-cyan); margin-bottom: 1rem;'>Regime Intelligence</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem; line-height: 1.7;'>
                Adaptive state detection:
            </p>
            <ul style='color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8; margin-top: 0.5rem;'>
                <li><strong>HMM</strong> - State discovery</li>
                <li><strong>GARCH</strong> - Volatility regime</li>
                <li><strong>CUSUM</strong> - Change points</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Signal interpretation
    st.markdown("### 🎯 Signal Interpretation")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown("""
        <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid var(--success-green); border-radius: 12px; padding: 1.25rem;'>
            <h4 style='color: var(--success-green); margin-bottom: 0.75rem;'>🟢 Oversold Zone</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem;'>Signal &lt; -5</p>
            <p style='color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;'>
                Potential buying opportunity. Look for confirmation with divergences and macro support.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_s2:
        st.markdown("""
        <div style='background: rgba(136, 136, 136, 0.1); border: 1px solid var(--neutral); border-radius: 12px; padding: 1.25rem;'>
            <h4 style='color: var(--neutral); margin-bottom: 0.75rem;'>⚪ Neutral Zone</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem;'>Signal -5 to +5</p>
            <p style='color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;'>
                No clear directional bias. Wait for breakout or use other confluence factors.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_s3:
        st.markdown("""
        <div style='background: rgba(239, 68, 68, 0.1); border: 1px solid var(--danger-red); border-radius: 12px; padding: 1.25rem;'>
            <h4 style='color: var(--danger-red); margin-bottom: 0.75rem;'>🔴 Overbought Zone</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem;'>Signal &gt; +5</p>
            <p style='color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;'>
                Potential selling opportunity. Watch for bearish divergences and macro headwinds.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("### 📊 System Coverage")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card neutral"><h4>ETF Universe</h4><h2>{len(SCREENER_SYMBOLS)}</h2><div class="sub-metric">Curated ETFs</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card neutral"><h4>Index Options</h4><h2>{len(INDEX_LIST)}</h2><div class="sub-metric">NSE Indices</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card neutral"><h4>Macro Factors</h4><h2>{len(MACRO_SYMBOLS)}</h2><div class="sub-metric">Correlation Drivers</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card neutral"><h4>Analysis Modes</h4><h2>3</h2><div class="sub-metric">Screener Types</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Getting started
    st.markdown("""
    <div class='info-box'>
        <h4>🚀 Getting Started</h4>
        <p style='color: var(--text-muted); line-height: 1.7;'>
            Select an analysis mode from the sidebar to begin. Each mode offers both <strong>Single Day</strong> analysis 
            for current signals and <strong>Time Series</strong> analysis for tracking signal evolution over time.
            Adjust indicator parameters in the sidebar's <em>Indicator Settings</em> expander for fine-tuning.
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    mode, length, roc_len, regime_sensitivity, base_weight, spread_universe, spread_index, spread_date, spread_mode, spread_start_date, spread_end_date, etf_mode, etf_date, etf_start_date, etf_end_date = render_sidebar()
    
    # Only show main header on Home page
    if "Home" in mode:
        render_header()
    
    if "Home" in mode:
        run_home_page()
    elif "ETF" in mode:
        # ETF Screener (fixed ETF universe)
        if etf_mode and "Time Series" in etf_mode:
            run_etf_timeseries_mode(length, roc_len, regime_sensitivity, base_weight, etf_start_date, etf_end_date)
        else:
            run_etf_screener_mode(length, roc_len, regime_sensitivity, base_weight, etf_date)
    elif "Market" in mode:
        # Market Screener (F&O / Index universe)
        if spread_mode and "Time Series" in spread_mode:
            run_market_timeseries_mode(length, roc_len, regime_sensitivity, base_weight, spread_universe, spread_index, spread_start_date, spread_end_date)
        else:
            run_market_screener_mode(length, roc_len, regime_sensitivity, base_weight, spread_universe, spread_index, spread_date)
    elif "Chart" in mode:
        run_chart_mode(length, roc_len, regime_sensitivity, base_weight)
    
    # Dynamic footer with current IST time
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    ist_now = utc_now + datetime.timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© 2026 {PRODUCT_NAME} | {COMPANY} | {VERSION} | {current_time_ist}")


def run_chart_mode(length, roc_len, regime_sensitivity, base_weight):
    col1, col2 = st.columns([3, 1])
    with col1:
        target_symbol = st.text_input("Symbol", value="SPY", placeholder="Enter ticker (e.g., SPY, AAPL, NIFTYIETF.NS)", label_visibility="collapsed")
    with col2:
        analyze_btn = st.button("◈ ANALYZE", type="primary", width="stretch")
    
    if analyze_btn and target_symbol:
        with st.spinner(""):
            st.toast("Fetching market data...", icon="⏳")
            macro_df = fetch_macro_data(days_back=100)
            df = fetch_ticker_data(target_symbol, macro_df, days_back=100)
            
            if df is not None and not df.empty:
                try:
                    df, drivers = run_full_analysis(df, length, roc_len, regime_sensitivity, base_weight)
                    display_df = df.iloc[-100:].copy()
                    
                    curr_unified = display_df['Unified_Osc'].iloc[-1]
                    curr_msf = display_df['MSF_Osc'].iloc[-1]
                    curr_mmr = display_df['MMR_Osc'].iloc[-1]
                    curr_condition = display_df['Condition'].iloc[-1]
                    curr_price = display_df['Close'].iloc[-1]
                    prev_price = display_df['Close'].iloc[-2]
                    price_change = ((curr_price - prev_price) / prev_price) * 100
                    has_buy = display_df['Buy_Signal'].iloc[-1]
                    has_sell = display_df['Sell_Signal'].iloc[-1]
                    
                    # NEW: Get regime intelligence data
                    curr_regime = display_df['Regime'].iloc[-1]
                    curr_hmm_bull = display_df['HMM_Bull'].iloc[-1]
                    curr_hmm_bear = display_df['HMM_Bear'].iloc[-1]
                    curr_vol_regime = display_df['Vol_Regime'].iloc[-1]
                    curr_change_point = display_df['Change_Point'].iloc[-1]
                    curr_confidence = display_df['Confidence'].iloc[-1]
                    
                    st.toast("Analysis Complete!", icon="✅")
                    
                    # Row 1: Signal metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        color_class = "success" if curr_condition == "Oversold" else "danger" if curr_condition == "Overbought" else "neutral"
                        st.markdown(f'<div class="metric-card {color_class}"><h4>Unified Signal</h4><h2>{curr_unified:.2f}</h2><div class="sub-metric">{curr_condition}</div></div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="metric-card primary"><h4>MSF (Internal)</h4><h2>{curr_msf:.2f}</h2><div class="sub-metric">Structure & Flow</div></div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown(f'<div class="metric-card info"><h4>MMR (Macro)</h4><h2>{curr_mmr:.2f}</h2><div class="sub-metric">Macro Regression</div></div>', unsafe_allow_html=True)
                    with col4:
                        price_color = "success" if price_change >= 0 else "danger"
                        curr_sym = get_currency_symbol(target_symbol)
                        st.markdown(f'<div class="metric-card {price_color}"><h4>Price</h4><h2>{curr_sym}{curr_price:,.2f}</h2><div class="sub-metric">{"▲" if price_change >= 0 else "▼"} {abs(price_change):.2f}%</div></div>', unsafe_allow_html=True)
                    
                    # Row 2: Regime Intelligence metrics (NEW)
                    st.markdown("<br>", unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        regime_color = "success" if "BULL" in curr_regime else "danger" if "BEAR" in curr_regime else "warning" if curr_regime == "TRANSITION" else "neutral"
                        st.markdown(f'<div class="metric-card {regime_color}"><h4>HMM Regime</h4><h2 style="font-size: 1.3rem;">{curr_regime}</h2><div class="sub-metric">Confidence: {curr_confidence:.0%}</div></div>', unsafe_allow_html=True)
                    with col2:
                        vol_color = "danger" if curr_vol_regime in ["HIGH", "EXTREME"] else "success" if curr_vol_regime == "LOW" else "neutral"
                        st.markdown(f'<div class="metric-card {vol_color}"><h4>Volatility Regime</h4><h2 style="font-size: 1.3rem;">{curr_vol_regime}</h2><div class="sub-metric">GARCH-based</div></div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown(f'<div class="metric-card success"><h4>P(Bull)</h4><h2>{curr_hmm_bull:.0%}</h2><div class="sub-metric">HMM State</div></div>', unsafe_allow_html=True)
                    with col4:
                        st.markdown(f'<div class="metric-card danger"><h4>P(Bear)</h4><h2>{curr_hmm_bear:.0%}</h2><div class="sub-metric">HMM State</div></div>', unsafe_allow_html=True)
                    
                    # Change point alert (NEW)
                    if curr_change_point:
                        st.markdown('<span class="status-badge regime">⚠️ CHANGE POINT DETECTED - Regime Transition</span>', unsafe_allow_html=True)
                    
                    if has_buy or has_sell:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if has_buy:
                            st.markdown('<span class="status-badge buy">◉ CONFIRMED BUY SIGNAL</span>', unsafe_allow_html=True)
                        if has_sell:
                            st.markdown('<span class="status-badge sell">◉ CONFIRMED SELL SIGNAL</span>', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    tab3, tab1, tab2 = st.tabs(["**🎯 Regime Intelligence**", "**📈 Price & Oscillator**", "**📊 Signal Components**"])
                    
                    with tab3:
                        # Regime Intelligence Tab (FIRST)
                        st.markdown("##### HMM State Probabilities Over Time")
                        
                        # Create HMM probability chart
                        fig_hmm = go.Figure()
                        fig_hmm.add_trace(go.Scatter(x=display_df.index, y=display_df['HMM_Bull'], mode='lines', name='P(Bull)', line=dict(color='#10b981', width=2), fill='tozeroy', fillcolor='rgba(16,185,129,0.2)'))
                        fig_hmm.add_trace(go.Scatter(x=display_df.index, y=display_df['HMM_Bear'], mode='lines', name='P(Bear)', line=dict(color='#ef4444', width=2), fill='tozeroy', fillcolor='rgba(239,68,68,0.2)'))
                        fig_hmm.add_hline(y=0.5, line_dash='dash', line_color='#888888')
                        fig_hmm.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=250, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1], showgrid=True, gridcolor='#2A2A2A'), xaxis=dict(showgrid=True, gridcolor='#2A2A2A'), legend=dict(orientation='h', y=1.1), hovermode='x unified')
                        st.plotly_chart(fig_hmm, width="stretch", config={'displayModeBar': False})
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("##### Regime Timeline")
                            # Create regime color mapping
                            regime_colors = {'STRONG_BULL': '#059669', 'BULL': '#10b981', 'WEAK_BULL': '#34d399', 'NEUTRAL': '#888888', 'WEAK_BEAR': '#fbbf24', 'BEAR': '#ef4444', 'CRISIS': '#dc2626', 'TRANSITION': '#a855f7'}
                            regime_vals = display_df['Regime'].map(lambda x: regime_colors.get(x, '#888888'))
                            
                            # Count regimes
                            regime_counts = display_df['Regime'].value_counts()
                            fig_regime = go.Figure(go.Pie(
                                labels=regime_counts.index, values=regime_counts.values, hole=0.5,
                                marker=dict(colors=[regime_colors.get(r, '#888888') for r in regime_counts.index], line=dict(color='#1A1A1A', width=2)),
                                textinfo='label+percent', textfont=dict(size=10, color='white')
                            ))
                            fig_regime.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#EAEAEA'), height=250, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
                            st.plotly_chart(fig_regime, width="stretch", config={'displayModeBar': False})
                        
                        with c2:
                            st.markdown("##### Volatility Regime Distribution")
                            vol_counts = display_df['Vol_Regime'].value_counts()
                            vol_colors = {'LOW': '#10b981', 'NORMAL': '#888888', 'HIGH': '#f59e0b', 'EXTREME': '#ef4444'}
                            fig_vol = go.Figure(go.Pie(
                                labels=vol_counts.index, values=vol_counts.values, hole=0.5,
                                marker=dict(colors=[vol_colors.get(v, '#888888') for v in vol_counts.index], line=dict(color='#1A1A1A', width=2)),
                                textinfo='label+percent', textfont=dict(size=10, color='white')
                            ))
                            fig_vol.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#EAEAEA'), height=250, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
                            st.plotly_chart(fig_vol, width="stretch", config={'displayModeBar': False})
                        
                        # Change points
                        change_points = display_df[display_df['Change_Point'] == True]
                        if len(change_points) > 0:
                            st.markdown(f"##### Change Points Detected: {len(change_points)}")
                            st.dataframe(change_points[['Close', 'Unified_Osc', 'Regime', 'Vol_Regime']].tail(10), width="stretch")
                        else:
                            st.info("No change points detected in this period")
                    
                    with tab1:
                        st.plotly_chart(create_price_chart(display_df, target_symbol), width="stretch", config={'displayModeBar': False})
                        st.plotly_chart(create_oscillator_chart(display_df), width="stretch", config={'displayModeBar': False})
                    
                    with tab2:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("##### Signal Gauge")
                            st.plotly_chart(create_gauge_chart(curr_unified), width="stretch", config={'displayModeBar': False})
                        with c2:
                            st.markdown("##### Top Macro Drivers")
                            if drivers:
                                for d in sorted(drivers, key=lambda x: abs(x['Correlation']), reverse=True):
                                    corr = d['Correlation']
                                    color = '#10b981' if corr > 0 else '#ef4444'
                                    pct = abs(corr) * 100
                                    st.markdown(f'<div style="margin-bottom: 0.75rem;"><div style="display: flex; justify-content: space-between; font-size: 0.85rem;"><span style="color: #EAEAEA;">{d["Name"]}</span><span style="color: {color}; font-weight: 600;">{corr:+.3f}</span></div><div class="conviction-meter"><div class="conviction-fill" style="width: {pct}%; background: {color};"></div></div></div>', unsafe_allow_html=True)
                            else:
                                st.info("No macro correlations available")
                        
                except Exception as e:
                    st.error(f"Analysis Error: {str(e)}")
            else:
                st.warning("No data found. Please verify the ticker symbol.")


def run_etf_screener_mode(length, roc_len, regime_sensitivity, base_weight, analysis_date):
    """ETF Screener: NIRNAY analysis on fixed ETF universe with date selection"""
    
    # Format analysis date
    if analysis_date is None:
        analysis_date = datetime.date.today()
    analysis_date_str = analysis_date.strftime("%d %b %Y")
    is_today = analysis_date == datetime.date.today()
    
    st.markdown(f"""
    <div class='info-box'>
        <h4>🏦 ETF Screener - Fixed Universe</h4>
        <p>Full NIRNAY (MSF + MMR) analysis across {len(SCREENER_SYMBOLS)} ETFs.<br>
        <strong>Analysis Date:</strong> {analysis_date_str} {"(Today)" if is_today else ""}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Validate analysis date
    if analysis_date > datetime.date.today():
        st.error("⚠️ Analysis date cannot be in the future.")
        return
    
    if st.button("◈ RUN ETF SCREENER", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.markdown("**⏳ Fetching global macro data...**")
        
        # Fetch macro data with buffer for historical analysis
        days_back = 100 + (datetime.date.today() - analysis_date).days
        macro_df = fetch_macro_data(days_back=days_back)
        
        results = []
        errors = []
        total = len(SCREENER_SYMBOLS)
        
        for i, symbol in enumerate(SCREENER_SYMBOLS):
            status_text.markdown(f"**⏳ Scanning {get_display_name(symbol)} ({i+1}/{total})**")
            progress_bar.progress((i + 1) / total)
            df = fetch_ticker_data(symbol, macro_df, days_back=days_back)
            
            if df is not None and len(df) > length + 5:
                try:
                    df, _ = run_full_analysis(df, length, roc_len, regime_sensitivity, base_weight)
                    
                    # Find the row for the analysis date
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    
                    # Get the closest date on or before analysis_date
                    analysis_datetime = pd.Timestamp(analysis_date)
                    valid_dates = df.index[df.index <= analysis_datetime]
                    
                    if len(valid_dates) == 0:
                        continue
                    
                    target_date = valid_dates[-1]
                    target_idx = df.index.get_loc(target_date)
                    
                    if target_idx < 1:
                        continue
                    
                    last_row = df.iloc[target_idx]
                    prev_row = df.iloc[target_idx - 1]
                    price_change = ((last_row['Close'] - prev_row['Close']) / prev_row['Close']) * 100
                    
                    signal_str = "BUY" if last_row['Buy_Signal'] else "SELL" if last_row['Sell_Signal'] else "-"
                    div_str = "BULL" if last_row['Bullish_Div'] else "BEAR" if last_row['Bearish_Div'] else "-"
                    
                    results.append({
                        "Symbol": symbol, "DisplayName": get_display_name(symbol),
                        "Price": round(last_row['Close'], 2),
                        "Change": round(price_change, 2),
                        "Signal": round(last_row['Unified_Osc'], 2),
                        "MSF": round(last_row['MSF_Osc'], 2),
                        "MMR": round(last_row['MMR_Osc'], 2),
                        "Zone": last_row['Condition'],
                        "Trigger": signal_str,
                        "Divergence": div_str,
                        "Agreement": round(last_row['Agreement'], 3),
                        # NEW: Regime Intelligence columns
                        "Regime": last_row['Regime'],
                        "HMM_Bull": round(last_row['HMM_Bull'], 2),
                        "HMM_Bear": round(last_row['HMM_Bear'], 2),
                        "Vol_Regime": last_row['Vol_Regime'],
                        "Confidence": round(last_row['Confidence'], 2),
                        "Change_Point": last_row['Change_Point']
                    })
                except Exception as e:
                    errors.append(f"{symbol}: {str(e)[:80]}")
        
        progress_bar.empty()
        status_text.empty()
        
        if errors:
            st.warning(f"⚠️ {len(errors)}/{total} tickers failed analysis. Sample: {errors[0]}")
        
        if results:
            st.toast(f"ETF Scan Complete! Analyzed {len(results)}/{total} ETFs", icon="✅")
            results_df = pd.DataFrame(results)
            
            # Calculate summary stats
            n_oversold = len(results_df[results_df['Zone'] == 'Oversold'])
            n_overbought = len(results_df[results_df['Zone'] == 'Overbought'])
            n_neutral = len(results_df[results_df['Zone'] == 'Neutral'])
            n_buys = len(results_df[results_df['Trigger'] == 'BUY'])
            n_sells = len(results_df[results_df['Trigger'] == 'SELL'])
            avg_signal = results_df['Signal'].mean()
            
            # NEW: Calculate HMM regime distribution
            n_bull = len(results_df[results_df['Regime'].str.contains('BULL', na=False)])
            n_bear = len(results_df[results_df['Regime'].str.contains('BEAR', na=False)])
            n_transition = len(results_df[results_df['Regime'] == 'TRANSITION'])
            dominant_regime = (results_df['Regime'].mode().iloc[0] if not results_df['Regime'].mode().empty else 'NEUTRAL') if len(results_df) > 0 else 'NEUTRAL'
            regime_color = "success" if "BULL" in dominant_regime else "danger" if "BEAR" in dominant_regime else "warning" if dominant_regime == "TRANSITION" else "neutral"
            
            # Metrics row
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.markdown(f'<div class="metric-card info"><h4>Universe</h4><h2>{len(results)}</h2><div class="sub-metric">ETFs Analyzed</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card success"><h4>Oversold</h4><h2>{n_oversold}</h2><div class="sub-metric">Buy Zone</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card danger"><h4>Overbought</h4><h2>{n_overbought}</h2><div class="sub-metric">Sell Zone</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card primary"><h4>Buy Signals</h4><h2>{n_buys}</h2><div class="sub-metric">Confirmed</div></div>', unsafe_allow_html=True)
            with c5:
                st.markdown(f'<div class="metric-card warning"><h4>Sell Signals</h4><h2>{n_sells}</h2><div class="sub-metric">Confirmed</div></div>', unsafe_allow_html=True)
            with c6:
                st.markdown(f'<div class="metric-card {regime_color}"><h4>HMM Regime</h4><h2 style="font-size: 1.1rem;">{dominant_regime}</h2><div class="sub-metric">Bull: {n_bull} | Bear: {n_bear}</div></div>', unsafe_allow_html=True)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
#            # Tabs for ETF Screener - Regime Analysis first
            tab4, tab1, tab2, tab3, tab5 = st.tabs(["**🎯 Regime Analysis**", "**📊 Signal Dashboard**", "**📈 Top Signals**", "**📉 Distribution**", "**📋 Full Data**"])
            
            with tab1:
                col_buy, col_sell = st.columns(2)
                
                with col_buy:
                    st.markdown('<div class="signal-card buy"><div class="signal-card-header"><span class="signal-card-title">🟢 Buy Opportunities</span></div>', unsafe_allow_html=True)
                    
                    confirmed_buys = results_df[results_df['Trigger'] == 'BUY'].sort_values('Signal').head(15)
                    if not confirmed_buys.empty:
                        st.markdown('<span class="status-badge buy">CONFIRMED BUY SIGNALS</span>', unsafe_allow_html=True)
                        for _, row in confirmed_buys.iterrows():
                            st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • ₹{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #10b981;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                    
                    oversold = results_df[(results_df['Zone'] == 'Oversold') & (results_df['Trigger'] != 'BUY')].sort_values('Signal').head(15)
                    if not oversold.empty:
                        st.markdown('<span class="status-badge oversold">OVERSOLD ZONE</span>', unsafe_allow_html=True)
                        for _, row in oversold.iterrows():
                            st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • ₹{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #06b6d4;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
                    
                    if confirmed_buys.empty and oversold.empty:
                        st.markdown('<p style="color: #888888; padding: 1rem;">No buy opportunities detected</p>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col_sell:
                    st.markdown('<div class="signal-card sell"><div class="signal-card-header"><span class="signal-card-title">🔴 Sell Opportunities</span></div>', unsafe_allow_html=True)
                    
                    confirmed_sells = results_df[results_df['Trigger'] == 'SELL'].sort_values('Signal', ascending=False).head(15)
                    if not confirmed_sells.empty:
                        st.markdown('<span class="status-badge sell">CONFIRMED SELL SIGNALS</span>', unsafe_allow_html=True)
                        for _, row in confirmed_sells.iterrows():
                            st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • ₹{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #ef4444;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                    
                    overbought = results_df[(results_df['Zone'] == 'Overbought') & (results_df['Trigger'] != 'SELL')].sort_values('Signal', ascending=False).head(15)
                    if not overbought.empty:
                        st.markdown('<span class="status-badge overbought">OVERBOUGHT ZONE</span>', unsafe_allow_html=True)
                        for _, row in overbought.iterrows():
                            st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • ₹{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #f59e0b;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
                    
                    if confirmed_sells.empty and overbought.empty:
                        st.markdown('<p style="color: #888888; padding: 1rem;">No sell opportunities detected</p>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Divergence alerts
                st.markdown("<br>", unsafe_allow_html=True)
                bull_divs = results_df[results_df['Divergence'] == 'BULL']
                bear_divs = results_df[results_df['Divergence'] == 'BEAR']
                
                if not bull_divs.empty or not bear_divs.empty:
                    st.markdown("##### 📊 Divergence Alerts")
                    div_cols = st.columns(2)
                    with div_cols[0]:
                        if not bull_divs.empty:
                            st.markdown('<span class="status-badge divergence">BULLISH DIVERGENCES</span>', unsafe_allow_html=True)
                            for _, row in bull_divs.head(10).iterrows():
                                st.markdown(f'<div class="symbol-row"><span class="symbol-name">{row["DisplayName"]}</span><span style="color: #FFC300;">Price ▼ | Signal ▲</span></div>', unsafe_allow_html=True)
                    with div_cols[1]:
                        if not bear_divs.empty:
                            st.markdown('<span class="status-badge divergence">BEARISH DIVERGENCES</span>', unsafe_allow_html=True)
                            for _, row in bear_divs.head(10).iterrows():
                                st.markdown(f'<div class="symbol-row"><span class="symbol-name">{row["DisplayName"]}</span><span style="color: #FFC300;">Price ▲ | Signal ▼</span></div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown("##### 🏆 Top Oversold ETFs")
                top_oversold = results_df.nsmallest(15, 'Signal')
                cols_o = ['DisplayName', 'Price', 'Change', 'Signal', 'MSF', 'MMR', 'Zone', 'Trigger']
                st.dataframe(top_oversold[cols_o].rename(columns={'DisplayName': 'ETF', 'Change': 'Chg %'}), width="stretch", hide_index=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### 🔻 Top Overbought ETFs")
                top_overbought = results_df.nlargest(15, 'Signal')
                st.dataframe(top_overbought[cols_o].rename(columns={'DisplayName': 'ETF', 'Change': 'Chg %'}), width="stretch", hide_index=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### 📊 Signal Ranking Chart")
                st.plotly_chart(create_ranking_chart(results_df, 15), width="stretch", config={'displayModeBar': False})
            
            with tab3:
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    st.markdown("##### Signal Distribution")
                    st.plotly_chart(create_distribution_chart(results_df), width="stretch", config={'displayModeBar': False})
                    
                    st.markdown("##### Zone Breakdown")
                    zone_data = {
                        "Zone": ["Oversold (< -5)", "Neutral (-5 to +5)", "Overbought (> +5)"],
                        "Count": [n_oversold, n_neutral, n_overbought],
                        "Percentage": [f"{n_oversold/len(results_df)*100:.1f}%", f"{n_neutral/len(results_df)*100:.1f}%", f"{n_overbought/len(results_df)*100:.1f}%"]
                    }
                    st.dataframe(pd.DataFrame(zone_data), width="stretch", hide_index=True)
                
                with col_d2:
                    st.markdown("##### Statistical Summary")
                    stats_data = {
                        "Metric": ["Total ETFs", "Mean Signal", "Median Signal", "Std Dev", "Min Signal", "Max Signal", "Buy/Sell Ratio"],
                        "Value": [
                            f"{len(results_df)}",
                            f"{results_df['Signal'].mean():.2f}",
                            f"{results_df['Signal'].median():.2f}",
                            f"{results_df['Signal'].std():.2f}",
                            f"{results_df['Signal'].min():.2f}",
                            f"{results_df['Signal'].max():.2f}",
                            f"{n_buys}:{n_sells}" if n_sells > 0 else f"{n_buys}:0"
                        ]
                    }
                    st.dataframe(pd.DataFrame(stats_data), width="stretch", hide_index=True)
                    
                    st.markdown("##### Top Gainers Today")
                    top_gainers = results_df.nlargest(10, 'Change')[['DisplayName', 'Price', 'Change', 'Signal']]
                    top_gainers.columns = ['ETF', 'Price', 'Chg %', 'Signal']
                    st.dataframe(top_gainers, width="stretch", hide_index=True)
                    
                    st.markdown("##### Top Losers Today")
                    top_losers = results_df.nsmallest(10, 'Change')[['DisplayName', 'Price', 'Change', 'Signal']]
                    top_losers.columns = ['ETF', 'Price', 'Chg %', 'Signal']
                    st.dataframe(top_losers, width="stretch", hide_index=True)
            
            with tab4:
                # NEW: Regime Intelligence Analysis Tab
                st.markdown("##### HMM Regime Distribution")
                c1, c2 = st.columns(2)
                
                with c1:
                    regime_counts = results_df['Regime'].value_counts()
                    regime_colors = {'STRONG_BULL': '#059669', 'BULL': '#10b981', 'WEAK_BULL': '#34d399', 'NEUTRAL': '#888888', 'WEAK_BEAR': '#fbbf24', 'BEAR': '#ef4444', 'CRISIS': '#dc2626', 'TRANSITION': '#a855f7'}
                    fig_regime = go.Figure(go.Pie(
                        labels=regime_counts.index, values=regime_counts.values, hole=0.5,
                        marker=dict(colors=[regime_colors.get(r, '#888888') for r in regime_counts.index], line=dict(color='#1A1A1A', width=2)),
                        textinfo='label+percent', textfont=dict(size=11, color='white')
                    ))
                    fig_regime.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#EAEAEA'), height=300, margin=dict(l=20, r=20, t=30, b=20), showlegend=False, title=dict(text='HMM Regime', font=dict(size=14, color='#888888')))
                    st.plotly_chart(fig_regime, width="stretch", config={'displayModeBar': False})
                
                with c2:
                    vol_counts = results_df['Vol_Regime'].value_counts()
                    vol_colors = {'LOW': '#10b981', 'NORMAL': '#888888', 'HIGH': '#f59e0b', 'EXTREME': '#ef4444'}
                    fig_vol = go.Figure(go.Pie(
                        labels=vol_counts.index, values=vol_counts.values, hole=0.5,
                        marker=dict(colors=[vol_colors.get(v, '#888888') for v in vol_counts.index], line=dict(color='#1A1A1A', width=2)),
                        textinfo='label+percent', textfont=dict(size=11, color='white')
                    ))
                    fig_vol.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#EAEAEA'), height=300, margin=dict(l=20, r=20, t=30, b=20), showlegend=False, title=dict(text='Volatility Regime (GARCH)', font=dict(size=14, color='#888888')))
                    st.plotly_chart(fig_vol, width="stretch", config={'displayModeBar': False})
                
                st.markdown("##### Regime Breakdown by ETF")
                
                # Bull regime ETFs
                col_bull, col_bear = st.columns(2)
                with col_bull:
                    st.markdown('<span class="status-badge buy">BULLISH REGIME ETFs</span>', unsafe_allow_html=True)
                    bull_etfs = results_df[results_df['Regime'].str.contains('BULL', na=False)].sort_values('HMM_Bull', ascending=False)
                    if len(bull_etfs) > 0:
                        for _, row in bull_etfs.head(10).iterrows():
                            st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • {row["Regime"]}</span></div><span class="symbol-score" style="color: #10b981;">P(Bull): {row["HMM_Bull"]:.0%}</span></div>', unsafe_allow_html=True)
                    else:
                        st.info("No ETFs in bullish regime")
                
                with col_bear:
                    st.markdown('<span class="status-badge sell">BEARISH REGIME ETFs</span>', unsafe_allow_html=True)
                    bear_etfs = results_df[results_df['Regime'].str.contains('BEAR', na=False)].sort_values('HMM_Bear', ascending=False)
                    if len(bear_etfs) > 0:
                        for _, row in bear_etfs.head(10).iterrows():
                            st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • {row["Regime"]}</span></div><span class="symbol-score" style="color: #ef4444;">P(Bear): {row["HMM_Bear"]:.0%}</span></div>', unsafe_allow_html=True)
                    else:
                        st.info("No ETFs in bearish regime")
                
                # Change points
                change_point_etfs = results_df[results_df['Change_Point'] == True]
                if len(change_point_etfs) > 0:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<span class="status-badge regime">⚠️ CHANGE POINTS DETECTED</span>', unsafe_allow_html=True)
                    for _, row in change_point_etfs.iterrows():
                        st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • Regime Transition</span></div><span class="symbol-score" style="color: #a855f7;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
            
            with tab5:
                st.markdown(f"##### Complete ETF Scan Results ({len(results_df)} ETFs) - {analysis_date_str}")
                
                # Filter options
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                with filter_col1:
                    zone_filter = st.multiselect("Filter by Zone", ["Oversold", "Neutral", "Overbought"], default=["Oversold", "Neutral", "Overbought"], key="etf_zone_filter")
                with filter_col2:
                    signal_filter = st.multiselect("Filter by Trigger", ["BUY", "SELL", "-"], default=["BUY", "SELL", "-"], key="etf_signal_filter")
                with filter_col3:
                    sort_by = st.selectbox("Sort by", ["Signal", "Change", "Price", "DisplayName", "Regime", "Confidence"], index=0, key="etf_sort_by")
                
                # Apply filters
                filtered_df = results_df[
                    (results_df['Zone'].isin(zone_filter)) & 
                    (results_df['Trigger'].isin(signal_filter))
                ].sort_values(sort_by, ascending=(sort_by == 'DisplayName'))
                
                # Updated display columns with Regime Intelligence
                display_cols = ['DisplayName', 'Price', 'Change', 'Signal', 'Zone', 'Trigger', 'Regime', 'Vol_Regime', 'Confidence']
                display_df = filtered_df[display_cols].copy()
                display_df.columns = ['ETF', 'Price', 'Chg %', 'Signal', 'Zone', 'Trigger', 'HMM Regime', 'Vol Regime', 'Conf']
                
                st.dataframe(display_df, width="stretch", hide_index=True, height=400)
                
                st.markdown("<br>", unsafe_allow_html=True)
                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Report (CSV)",
                    data=csv_data,
                    file_name=f"nirnay_etf_screener_{analysis_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No data retrieved. Please check your internet connection.")


def run_market_screener_mode(length, roc_len, regime_sensitivity, base_weight, spread_universe, spread_index, spread_date):
    """Market Screener: Full NIRNAY analysis (MSF + MMR + Regime) on F&O / Index stocks"""
    
    # Format analysis date
    analysis_date = spread_date if spread_date else datetime.date.today()
    analysis_date_str = analysis_date.strftime("%d %b %Y")
    is_today = analysis_date == datetime.date.today()
    
    # Display universe info
    if spread_universe == "F&O Stocks":
        universe_title = "F&O Stocks"
        st.markdown(f"""
        <div class='info-box'>
            <h4>📊 Market Screener - {universe_title}</h4>
            <p>Full NIRNAY (MSF + MMR + Regime) analysis across all F&O securities from NSE.<br>
            <strong>Analysis Date:</strong> {analysis_date_str} {"(Today)" if is_today else ""}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        universe_title = spread_index if spread_index else "Index"
        st.markdown(f"""
        <div class='info-box'>
            <h4>📊 Market Screener - {universe_title}</h4>
            <p>Full NIRNAY (MSF + MMR + Regime) analysis across all constituents of {universe_title}.<br>
            <strong>Analysis Date:</strong> {analysis_date_str} {"(Today)" if is_today else ""}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Validate analysis date
    if analysis_date > datetime.date.today():
        st.error("⚠️ Analysis date cannot be in the future.")
        return
    
    if st.button("◈ RUN MARKET SCREENER", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch stock list based on universe selection
        status_text.markdown(f"**⏳ Fetching {universe_title} stock list...**")
        
        if spread_universe == "F&O Stocks":
            stock_list, fetch_msg = get_fno_stock_list()
        else:
            stock_list, fetch_msg = get_index_stock_list(spread_index)
        
        if not stock_list:
            st.error(f"Failed to fetch stock list: {fetch_msg}")
            progress_bar.empty()
            status_text.empty()
            return
        
        st.toast(fetch_msg, icon="✅")
        total_stocks = len(stock_list)
        
        # Batch download data
        status_text.markdown(f"**⏳ Downloading data for {total_stocks} stocks...**")
        progress_bar.progress(0.05)
        
        data_dict, batch_msg = fetch_batch_data(stock_list, end_date=analysis_date, days_back=100)
        
        if data_dict is None:
            st.error(f"Failed to download data: {batch_msg}")
            progress_bar.empty()
            status_text.empty()
            return
        
        st.toast(batch_msg, icon="📥")
        
        # Fetch macro data ONCE for all stocks (VIX, DXY, rates are market-wide)
        status_text.markdown("**⏳ Fetching macro data for MMR analysis...**")
        progress_bar.progress(0.1)
        macro_df = fetch_macro_data(days_back=100)
        
        # Process each stock
        results = []
        errors = []
        valid_tickers = list(data_dict.keys())
        total_valid = len(valid_tickers)
        
        for i, ticker in enumerate(valid_tickers):
            status_text.markdown(f"**⏳ Analyzing {ticker.replace('.NS', '')} ({i+1}/{total_valid})**")
            progress_bar.progress(0.1 + (0.9 * (i + 1) / total_valid))
            
            df = data_dict[ticker]
            
            if df is not None and len(df) > length + 5:
                try:
                    # Merge macro data with stock data for MMR
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df = df.join(macro_df, how='left').ffill()
                    
                    # Run FULL analysis (MSF + MMR + Regime Intelligence)
                    df, _ = run_full_analysis(df, length, roc_len, regime_sensitivity, base_weight)
                    
                    # Get the closest date on or before analysis_date
                    analysis_datetime = pd.Timestamp(analysis_date)
                    valid_dates = df.index[df.index <= analysis_datetime]
                    
                    if len(valid_dates) == 0:
                        continue  # No data for this date
                    
                    target_date = valid_dates[-1]
                    target_idx = df.index.get_loc(target_date)
                    
                    if target_idx < 1:
                        continue  # Need at least one previous row for change calculation
                    
                    last_row = df.iloc[target_idx]
                    prev_row = df.iloc[target_idx - 1]
                    price_change = ((last_row['Close'] - prev_row['Close']) / prev_row['Close']) * 100
                    
                    signal_str = "BUY" if last_row['Buy_Signal'] else "SELL" if last_row['Sell_Signal'] else "-"
                    div_str = "BULL" if last_row['Bullish_Div'] else "BEAR" if last_row['Bearish_Div'] else "-"
                    
                    results.append({
                        "Symbol": ticker,
                        "DisplayName": ticker.replace(".NS", ""),
                        "Price": round(last_row['Close'], 2),
                        "Change": round(price_change, 2),
                        "Signal": round(last_row['Unified_Osc'], 2),
                        "MSF": round(last_row['MSF_Osc'], 2),
                        "MMR": round(last_row['MMR_Osc'], 2),  # Now populated with actual MMR
                        "Zone": last_row['Condition'],
                        "Trigger": signal_str,
                        "Divergence": div_str,
                        "Agreement": round(last_row['Agreement'], 3),
                        # Regime Intelligence columns
                        "Regime": last_row['Regime'],
                        "HMM_Bull": round(last_row['HMM_Bull'], 2),
                        "HMM_Bear": round(last_row['HMM_Bear'], 2),
                        "Vol_Regime": last_row['Vol_Regime'],
                        "Confidence": round(last_row['Confidence'], 2),
                        "Change_Point": last_row['Change_Point']
                    })
                except Exception as e:
                    errors.append(f"{ticker}: {str(e)[:80]}")
        
        progress_bar.empty()
        status_text.empty()
        
        if errors:
            st.warning(f"⚠️ {len(errors)}/{total_valid} tickers failed analysis. Sample: {errors[0]}")
        
        if results:
            st.toast(f"Market Scan Complete! Analyzed {len(results)}/{total_stocks} stocks", icon="✅")
            results_df = pd.DataFrame(results)
            
            # Calculate summary stats
            n_oversold = len(results_df[results_df['Zone'] == 'Oversold'])
            n_overbought = len(results_df[results_df['Zone'] == 'Overbought'])
            n_neutral = len(results_df[results_df['Zone'] == 'Neutral'])
            n_buys = len(results_df[results_df['Trigger'] == 'BUY'])
            n_sells = len(results_df[results_df['Trigger'] == 'SELL'])
            avg_signal = results_df['Signal'].mean()
            
            # NEW: Calculate HMM regime distribution
            n_bull = len(results_df[results_df['Regime'].str.contains('BULL', na=False)])
            n_bear = len(results_df[results_df['Regime'].str.contains('BEAR', na=False)])
            n_transition = len(results_df[results_df['Regime'] == 'TRANSITION'])
            dominant_regime = (results_df['Regime'].mode().iloc[0] if not results_df['Regime'].mode().empty else 'NEUTRAL') if len(results_df) > 0 else 'NEUTRAL'
            regime_color = "success" if "BULL" in dominant_regime else "danger" if "BEAR" in dominant_regime else "warning" if dominant_regime == "TRANSITION" else "neutral"
            
            # Metrics row
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.markdown(f'<div class="metric-card info"><h4>Universe</h4><h2>{len(results)}</h2><div class="sub-metric">Stocks Analyzed</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card success"><h4>Oversold</h4><h2>{n_oversold}</h2><div class="sub-metric">Buy Zone</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card danger"><h4>Overbought</h4><h2>{n_overbought}</h2><div class="sub-metric">Sell Zone</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card primary"><h4>Buy Signals</h4><h2>{n_buys}</h2><div class="sub-metric">Confirmed</div></div>', unsafe_allow_html=True)
            with c5:
                st.markdown(f'<div class="metric-card warning"><h4>Sell Signals</h4><h2>{n_sells}</h2><div class="sub-metric">Confirmed</div></div>', unsafe_allow_html=True)
            with c6:
                st.markdown(f'<div class="metric-card {regime_color}"><h4>HMM Regime</h4><h2 style="font-size: 1.1rem;">{dominant_regime}</h2><div class="sub-metric">Bull: {n_bull} | Bear: {n_bear}</div></div>', unsafe_allow_html=True)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
#            # Tabs for Market Screener Single Day - Regime Analysis first
            tab4, tab1, tab2, tab3, tab5 = st.tabs(["**🎯 Regime Analysis**", "**📊 Signal Dashboard**", "**📈 Top Signals**", "**📉 Distribution**", "**📋 Full Data**"])
            
            with tab1:
                col_buy, col_sell = st.columns(2)
                
                with col_buy:
                    st.markdown('<div class="signal-card buy"><div class="signal-card-header"><span class="signal-card-title">🟢 Buy Opportunities</span></div>', unsafe_allow_html=True)
                    
                    confirmed_buys = results_df[results_df['Trigger'] == 'BUY'].sort_values('Signal').head(15)
                    if not confirmed_buys.empty:
                        st.markdown('<span class="status-badge buy">CONFIRMED BUY SIGNALS</span>', unsafe_allow_html=True)
                        for _, row in confirmed_buys.iterrows():
                            st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • {get_currency_symbol(row["Symbol"])}{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #10b981;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                    
                    oversold = results_df[(results_df['Zone'] == 'Oversold') & (results_df['Trigger'] != 'BUY')].sort_values('Signal').head(15)
                    if not oversold.empty:
                        st.markdown('<span class="status-badge oversold">OVERSOLD ZONE</span>', unsafe_allow_html=True)
                        for _, row in oversold.iterrows():
                            st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • {get_currency_symbol(row["Symbol"])}{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #06b6d4;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
                    
                    if confirmed_buys.empty and oversold.empty:
                        st.markdown('<p style="color: #888888; padding: 1rem;">No buy opportunities detected</p>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col_sell:
                    st.markdown('<div class="signal-card sell"><div class="signal-card-header"><span class="signal-card-title">🔴 Sell Opportunities</span></div>', unsafe_allow_html=True)
                    
                    confirmed_sells = results_df[results_df['Trigger'] == 'SELL'].sort_values('Signal', ascending=False).head(15)
                    if not confirmed_sells.empty:
                        st.markdown('<span class="status-badge sell">CONFIRMED SELL SIGNALS</span>', unsafe_allow_html=True)
                        for _, row in confirmed_sells.iterrows():
                            st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • {get_currency_symbol(row["Symbol"])}{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #ef4444;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                    
                    overbought = results_df[(results_df['Zone'] == 'Overbought') & (results_df['Trigger'] != 'SELL')].sort_values('Signal', ascending=False).head(15)
                    if not overbought.empty:
                        st.markdown('<span class="status-badge overbought">OVERBOUGHT ZONE</span>', unsafe_allow_html=True)
                        for _, row in overbought.iterrows():
                            st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • {get_currency_symbol(row["Symbol"])}{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #f59e0b;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
                    
                    if confirmed_sells.empty and overbought.empty:
                        st.markdown('<p style="color: #888888; padding: 1rem;">No sell opportunities detected</p>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Divergence alerts
                st.markdown("<br>", unsafe_allow_html=True)
                bull_divs = results_df[results_df['Divergence'] == 'BULL']
                bear_divs = results_df[results_df['Divergence'] == 'BEAR']
                
                if not bull_divs.empty or not bear_divs.empty:
                    st.markdown("##### 📊 Divergence Alerts")
                    div_cols = st.columns(2)
                    with div_cols[0]:
                        if not bull_divs.empty:
                            st.markdown('<span class="status-badge divergence">BULLISH DIVERGENCES</span>', unsafe_allow_html=True)
                            for _, row in bull_divs.head(10).iterrows():
                                st.markdown(f'<div class="symbol-row"><span class="symbol-name">{row["DisplayName"]}</span><span style="color: #FFC300;">Price ▼ | Signal ▲</span></div>', unsafe_allow_html=True)
                    with div_cols[1]:
                        if not bear_divs.empty:
                            st.markdown('<span class="status-badge divergence">BEARISH DIVERGENCES</span>', unsafe_allow_html=True)
                            for _, row in bear_divs.head(10).iterrows():
                                st.markdown(f'<div class="symbol-row"><span class="symbol-name">{row["DisplayName"]}</span><span style="color: #FFC300;">Price ▲ | Signal ▼</span></div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown("##### 🏆 Top 20 Most Oversold")
                top_oversold = results_df.nsmallest(20, 'Signal')
                cols_o = ['DisplayName', 'Price', 'Change', 'Signal', 'Zone', 'Trigger']
                st.dataframe(top_oversold[cols_o].rename(columns={'DisplayName': 'Symbol', 'Change': 'Chg %'}), width="stretch", hide_index=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### 🔻 Top 20 Most Overbought")
                top_overbought = results_df.nlargest(20, 'Signal')
                st.dataframe(top_overbought[cols_o].rename(columns={'DisplayName': 'Symbol', 'Change': 'Chg %'}), width="stretch", hide_index=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### 📊 Extreme Signals Chart")
                st.plotly_chart(create_ranking_chart(results_df, 20), width="stretch", config={'displayModeBar': False})
            
            with tab3:
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    st.markdown("##### Signal Distribution")
                    st.plotly_chart(create_distribution_chart(results_df), width="stretch", config={'displayModeBar': False})
                    
                    st.markdown("##### Zone Breakdown")
                    zone_data = {
                        "Zone": ["Oversold (< -5)", "Neutral (-5 to +5)", "Overbought (> +5)"],
                        "Count": [n_oversold, n_neutral, n_overbought],
                        "Percentage": [f"{n_oversold/len(results_df)*100:.1f}%", f"{n_neutral/len(results_df)*100:.1f}%", f"{n_overbought/len(results_df)*100:.1f}%"]
                    }
                    st.dataframe(pd.DataFrame(zone_data), width="stretch", hide_index=True)
                
                with col_d2:
                    st.markdown("##### Statistical Summary")
                    stats_data = {
                        "Metric": ["Total Stocks", "Mean Signal", "Median Signal", "Std Dev", "Min Signal", "Max Signal", "Buy/Sell Ratio"],
                        "Value": [
                            f"{len(results_df)}",
                            f"{results_df['Signal'].mean():.2f}",
                            f"{results_df['Signal'].median():.2f}",
                            f"{results_df['Signal'].std():.2f}",
                            f"{results_df['Signal'].min():.2f}",
                            f"{results_df['Signal'].max():.2f}",
                            f"{n_buys}:{n_sells}" if n_sells > 0 else f"{n_buys}:0"
                        ]
                    }
                    st.dataframe(pd.DataFrame(stats_data), width="stretch", hide_index=True)
                    
                    st.markdown("##### Top Gainers Today")
                    top_gainers = results_df.nlargest(10, 'Change')[['DisplayName', 'Price', 'Change', 'Signal']]
                    top_gainers.columns = ['Symbol', 'Price', 'Chg %', 'Signal']
                    st.dataframe(top_gainers, width="stretch", hide_index=True)
                    
                    st.markdown("##### Top Losers Today")
                    top_losers = results_df.nsmallest(10, 'Change')[['DisplayName', 'Price', 'Change', 'Signal']]
                    top_losers.columns = ['Symbol', 'Price', 'Chg %', 'Signal']
                    st.dataframe(top_losers, width="stretch", hide_index=True)
            
            with tab4:
                # NEW: Regime Intelligence Analysis Tab
                st.markdown("##### HMM Regime Distribution")
                c1, c2 = st.columns(2)
                
                with c1:
                    regime_counts = results_df['Regime'].value_counts()
                    regime_colors = {'STRONG_BULL': '#059669', 'BULL': '#10b981', 'WEAK_BULL': '#34d399', 'NEUTRAL': '#888888', 'WEAK_BEAR': '#fbbf24', 'BEAR': '#ef4444', 'CRISIS': '#dc2626', 'TRANSITION': '#a855f7'}
                    fig_regime = go.Figure(go.Pie(
                        labels=regime_counts.index, values=regime_counts.values, hole=0.5,
                        marker=dict(colors=[regime_colors.get(r, '#888888') for r in regime_counts.index], line=dict(color='#1A1A1A', width=2)),
                        textinfo='label+percent', textfont=dict(size=11, color='white')
                    ))
                    fig_regime.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#EAEAEA'), height=300, margin=dict(l=20, r=20, t=30, b=20), showlegend=False, title=dict(text='HMM Regime', font=dict(size=14, color='#888888')))
                    st.plotly_chart(fig_regime, width="stretch", config={'displayModeBar': False})
                
                with c2:
                    vol_counts = results_df['Vol_Regime'].value_counts()
                    vol_colors = {'LOW': '#10b981', 'NORMAL': '#888888', 'HIGH': '#f59e0b', 'EXTREME': '#ef4444'}
                    fig_vol = go.Figure(go.Pie(
                        labels=vol_counts.index, values=vol_counts.values, hole=0.5,
                        marker=dict(colors=[vol_colors.get(v, '#888888') for v in vol_counts.index], line=dict(color='#1A1A1A', width=2)),
                        textinfo='label+percent', textfont=dict(size=11, color='white')
                    ))
                    fig_vol.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#EAEAEA'), height=300, margin=dict(l=20, r=20, t=30, b=20), showlegend=False, title=dict(text='Volatility Regime (GARCH)', font=dict(size=14, color='#888888')))
                    st.plotly_chart(fig_vol, width="stretch", config={'displayModeBar': False})
                
                st.markdown("##### Regime Breakdown by Stock")
                
                # Bull regime stocks
                col_bull, col_bear = st.columns(2)
                with col_bull:
                    st.markdown('<span class="status-badge buy">BULLISH REGIME STOCKS</span>', unsafe_allow_html=True)
                    bull_stocks = results_df[results_df['Regime'].str.contains('BULL', na=False)].sort_values('HMM_Bull', ascending=False)
                    if len(bull_stocks) > 0:
                        for _, row in bull_stocks.head(15).iterrows():
                            st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • {row["Regime"]}</span></div><span class="symbol-score" style="color: #10b981;">P(Bull): {row["HMM_Bull"]:.0%}</span></div>', unsafe_allow_html=True)
                    else:
                        st.info("No stocks in bullish regime")
                
                with col_bear:
                    st.markdown('<span class="status-badge sell">BEARISH REGIME STOCKS</span>', unsafe_allow_html=True)
                    bear_stocks = results_df[results_df['Regime'].str.contains('BEAR', na=False)].sort_values('HMM_Bear', ascending=False)
                    if len(bear_stocks) > 0:
                        for _, row in bear_stocks.head(15).iterrows():
                            st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • {row["Regime"]}</span></div><span class="symbol-score" style="color: #ef4444;">P(Bear): {row["HMM_Bear"]:.0%}</span></div>', unsafe_allow_html=True)
                    else:
                        st.info("No stocks in bearish regime")
                
                # Change points
                change_point_stocks = results_df[results_df['Change_Point'] == True]
                if len(change_point_stocks) > 0:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<span class="status-badge regime">⚠️ CHANGE POINTS DETECTED</span>', unsafe_allow_html=True)
                    for _, row in change_point_stocks.head(10).iterrows():
                        st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • Regime Transition</span></div><span class="symbol-score" style="color: #a855f7;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
            
            with tab5:
                st.markdown(f"##### Complete Market Scan Results ({len(results_df)} stocks) - {analysis_date_str}")
                
                # Filter options
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                with filter_col1:
                    zone_filter = st.multiselect("Filter by Zone", ["Oversold", "Neutral", "Overbought"], default=["Oversold", "Neutral", "Overbought"])
                with filter_col2:
                    signal_filter = st.multiselect("Filter by Trigger", ["BUY", "SELL", "-"], default=["BUY", "SELL", "-"])
                with filter_col3:
                    sort_by = st.selectbox("Sort by", ["Signal", "Change", "Price", "DisplayName", "Regime", "Confidence"], index=0)
                
                # Apply filters
                filtered_df = results_df[
                    (results_df['Zone'].isin(zone_filter)) & 
                    (results_df['Trigger'].isin(signal_filter))
                ].sort_values(sort_by, ascending=(sort_by == 'DisplayName'))
                
                display_cols = ['DisplayName', 'Price', 'Change', 'Signal', 'MSF', 'Zone', 'Trigger', 'Regime', 'Vol_Regime', 'Confidence']
                display_df = filtered_df[display_cols].copy()
                display_df.columns = ['Symbol', 'Price', 'Chg %', 'Signal', 'MSF', 'Zone', 'Trigger', 'HMM Regime', 'Vol Regime', 'Conf']
                
                st.dataframe(display_df, width="stretch", hide_index=True, height=500)
                
                st.markdown("<br>", unsafe_allow_html=True)
                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Report (CSV)",
                    data=csv_data,
                    file_name=f"nirnay_market_{universe_title.replace(' ', '_')}_{analysis_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No data retrieved. Please check your internet connection or try a different universe.")


def run_market_timeseries_mode(length, roc_len, regime_sensitivity, base_weight, spread_universe, spread_index, start_date, end_date):
    """Market Time Series Analysis: Full NIRNAY (MSF + MMR + Regime) tracking over time"""
    
    # Validate dates
    if start_date is None or end_date is None:
        st.error("Please select both start and end dates.")
        return
    
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return
    
    if end_date > datetime.date.today():
        st.error("End date cannot be in the future.")
        return
    
    # Calculate date range
    date_range_days = (end_date - start_date).days
    
    # Display info
    universe_title = spread_index if spread_universe == "Index Constituents" and spread_index else "F&O Stocks"
    st.markdown(f"""
    <div class='info-box'>
        <h4>📈 Time Series Analysis - {universe_title}</h4>
        <p>Full NIRNAY (MSF + MMR + Regime) signal distribution over time.<br>
        <strong>Period:</strong> {start_date.strftime("%d %b %Y")} to {end_date.strftime("%d %b %Y")} ({date_range_days} days)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("◈ RUN MARKET TIME SERIES", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch stock list
        status_text.markdown(f"**⏳ Fetching {universe_title} stock list...**")
        
        if spread_universe == "F&O Stocks":
            stock_list, fetch_msg = get_fno_stock_list()
        else:
            stock_list, fetch_msg = get_index_stock_list(spread_index)
        
        if not stock_list:
            st.error(f"Failed to fetch stock list: {fetch_msg}")
            progress_bar.empty()
            status_text.empty()
            return
        
        st.toast(fetch_msg, icon="✅")
        total_stocks = len(stock_list)
        
        # Batch download data for entire period
        status_text.markdown(f"**⏳ Downloading historical data for {total_stocks} stocks...**")
        progress_bar.progress(0.05)
        
        data_dict, batch_msg = fetch_batch_data(stock_list, end_date=end_date, days_back=100 + date_range_days)
        
        if data_dict is None:
            st.error(f"Failed to download data: {batch_msg}")
            progress_bar.empty()
            status_text.empty()
            return
        
        st.toast(batch_msg, icon="📥")
        
        # Fetch macro data ONCE for all stocks (VIX, DXY, rates are market-wide)
        status_text.markdown("**⏳ Fetching macro data for MMR analysis...**")
        progress_bar.progress(0.1)
        macro_df = fetch_macro_data(days_back=100 + date_range_days)
        
        # Generate list of trading days to analyze
        status_text.markdown("**⏳ Identifying trading days...**")
        progress_bar.progress(0.12)
        
        # Use one of the stocks to identify trading days
        sample_ticker = list(data_dict.keys())[0]
        sample_df = data_dict[sample_ticker]
        sample_df.index = pd.to_datetime(sample_df.index)
        if sample_df.index.tz is not None:
            sample_df.index = sample_df.index.tz_localize(None)
        
        # Get trading days in range
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        trading_days = sample_df.index[(sample_df.index >= start_ts) & (sample_df.index <= end_ts)].tolist()
        
        if len(trading_days) == 0:
            st.error("No trading days found in the selected date range.")
            progress_bar.empty()
            status_text.empty()
            return
        
        # Check if requested end date data is available
        actual_last_date = trading_days[-1].date() if trading_days else None
        is_today_included = actual_last_date == datetime.date.today() if actual_last_date else False
        
        if end_date == datetime.date.today():
            if is_today_included:
                st.toast(f"Live Data Included - {actual_last_date.strftime('%d %b %Y')}", icon="🔴")
            else:
                st.toast(f"Data through {actual_last_date.strftime('%d %b %Y')}", icon="⚠️")
        elif actual_last_date and actual_last_date < end_date:
            st.toast(f"Data through {actual_last_date.strftime('%d %b %Y')}", icon="⚠️")
        
        st.toast(f"Found {len(trading_days)} trading days", icon="📅")
        
        # Process FULL analysis for all stocks (MSF + MMR + Regime)
        status_text.markdown("**⏳ Computing MSF + MMR + Regime signals for all stocks...**")
        progress_bar.progress(0.15)
        
        processed_data = {}
        errors = []
        valid_tickers = list(data_dict.keys())
        
        for i, ticker in enumerate(valid_tickers):
            df = data_dict[ticker]
            if df is not None and len(df) > length + 5:
                try:
                    # Merge macro data with stock data for MMR
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df = df.join(macro_df, how='left').ffill()
                    
                    # Run FULL analysis (MSF + MMR + Regime Intelligence)
                    df, _ = run_full_analysis(df, length, roc_len, regime_sensitivity, base_weight)
                    processed_data[ticker] = df
                except Exception as e:
                    errors.append(f"{ticker}: {str(e)[:80]}")
            
            if (i + 1) % 50 == 0:
                progress_bar.progress(0.15 + 0.35 * (i + 1) / len(valid_tickers))
        
        status_text.markdown(f"**⏳ Analyzing {len(trading_days)} trading days...**")
        
        # Analyze each trading day
        timeseries_results = []
        
        for day_idx, trading_day in enumerate(trading_days):
            progress_bar.progress(0.5 + 0.45 * (day_idx + 1) / len(trading_days))
            
            day_stats = {
                "Date": trading_day.date(),
                "Oversold": 0,
                "Overbought": 0,
                "Neutral": 0,
                "Buy_Signals": 0,
                "Sell_Signals": 0,
                "Total_Analyzed": 0,
                "Avg_Signal": 0,
                "Signal_Sum": 0,
                "Bull_Div": 0,
                "Bear_Div": 0,
                # NEW: Regime Intelligence stats
                "Regime_Bull": 0,
                "Regime_Bear": 0,
                "Regime_Neutral": 0,
                "Regime_Transition": 0,
                "Vol_High": 0,
                "Vol_Low": 0,
                "Change_Points": 0
            }
            
            for ticker, df in processed_data.items():
                try:
                    # Get data for this trading day
                    if trading_day not in df.index:
                        continue
                    
                    row = df.loc[trading_day]
                    
                    day_stats["Total_Analyzed"] += 1
                    day_stats["Signal_Sum"] += row['Unified_Osc']
                    
                    if row['Condition'] == 'Oversold':
                        day_stats["Oversold"] += 1
                    elif row['Condition'] == 'Overbought':
                        day_stats["Overbought"] += 1
                    else:
                        day_stats["Neutral"] += 1
                    
                    if row['Buy_Signal']:
                        day_stats["Buy_Signals"] += 1
                    if row['Sell_Signal']:
                        day_stats["Sell_Signals"] += 1
                    if row['Bullish_Div']:
                        day_stats["Bull_Div"] += 1
                    if row['Bearish_Div']:
                        day_stats["Bear_Div"] += 1
                    
                    # NEW: Regime Intelligence stats
                    regime = row['Regime']
                    if 'BULL' in regime:
                        day_stats["Regime_Bull"] += 1
                    elif 'BEAR' in regime:
                        day_stats["Regime_Bear"] += 1
                    elif regime == 'TRANSITION':
                        day_stats["Regime_Transition"] += 1
                    else:
                        day_stats["Regime_Neutral"] += 1
                    
                    vol_regime = row['Vol_Regime']
                    if vol_regime in ['HIGH', 'EXTREME']:
                        day_stats["Vol_High"] += 1
                    elif vol_regime == 'LOW':
                        day_stats["Vol_Low"] += 1
                    
                    if row['Change_Point']:
                        day_stats["Change_Points"] += 1
                        
                except Exception as e:
                    errors.append(f"{ticker}: {str(e)[:80]}")
            
            if day_stats["Total_Analyzed"] > 0:
                day_stats["Avg_Signal"] = day_stats["Signal_Sum"] / day_stats["Total_Analyzed"]
                day_stats["Oversold_Pct"] = (day_stats["Oversold"] / day_stats["Total_Analyzed"]) * 100
                day_stats["Overbought_Pct"] = (day_stats["Overbought"] / day_stats["Total_Analyzed"]) * 100
                day_stats["Neutral_Pct"] = (day_stats["Neutral"] / day_stats["Total_Analyzed"]) * 100
                # NEW: Regime percentages
                day_stats["Regime_Bull_Pct"] = (day_stats["Regime_Bull"] / day_stats["Total_Analyzed"]) * 100
                day_stats["Regime_Bear_Pct"] = (day_stats["Regime_Bear"] / day_stats["Total_Analyzed"]) * 100
                day_stats["Vol_High_Pct"] = (day_stats["Vol_High"] / day_stats["Total_Analyzed"]) * 100
            else:
                day_stats["Oversold_Pct"] = 0
                day_stats["Overbought_Pct"] = 0
                day_stats["Neutral_Pct"] = 0
                day_stats["Regime_Bull_Pct"] = 0
                day_stats["Regime_Bear_Pct"] = 0
                day_stats["Vol_High_Pct"] = 0
            
            timeseries_results.append(day_stats)
        
        progress_bar.empty()
        status_text.empty()
        
        if not timeseries_results:
            st.warning("No data could be analyzed for the selected period.")
            return
        
        ts_df = pd.DataFrame(timeseries_results)
        ts_df['Date'] = pd.to_datetime(ts_df['Date'])
        ts_df = ts_df.sort_values('Date')
        
        # Show actual analyzed date range
        actual_start = ts_df['Date'].min().strftime('%d %b %Y')
        actual_end = ts_df['Date'].max().strftime('%d %b %Y')
        st.toast(f"Time Series Complete! {len(ts_df)} days ({actual_start} to {actual_end})", icon="✅")
        
        # Summary metrics
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        
        with c1:
            avg_oversold = ts_df['Oversold_Pct'].mean()
            st.markdown(f'<div class="metric-card success"><h4>Avg Oversold</h4><h2>{avg_oversold:.1f}%</h2><div class="sub-metric">Daily Average</div></div>', unsafe_allow_html=True)
        with c2:
            avg_overbought = ts_df['Overbought_Pct'].mean()
            st.markdown(f'<div class="metric-card danger"><h4>Avg Overbought</h4><h2>{avg_overbought:.1f}%</h2><div class="sub-metric">Daily Average</div></div>', unsafe_allow_html=True)
        with c3:
            total_buys = ts_df['Buy_Signals'].sum()
            st.markdown(f'<div class="metric-card primary"><h4>Total Buy Signals</h4><h2>{total_buys:,}</h2><div class="sub-metric">Over Period</div></div>', unsafe_allow_html=True)
        with c4:
            total_sells = ts_df['Sell_Signals'].sum()
            st.markdown(f'<div class="metric-card warning"><h4>Total Sell Signals</h4><h2>{total_sells:,}</h2><div class="sub-metric">Over Period</div></div>', unsafe_allow_html=True)
        with c5:
            avg_signal = ts_df['Avg_Signal'].mean()
            regime = "BULLISH" if avg_signal < -1 else "BEARISH" if avg_signal > 1 else "NEUTRAL"
            regime_color = "success" if avg_signal < -1 else "danger" if avg_signal > 1 else "neutral"
            st.markdown(f'<div class="metric-card {regime_color}"><h4>Period Regime</h4><h2 style="font-size: 1.1rem;">{regime}</h2><div class="sub-metric">Avg: {avg_signal:.2f}</div></div>', unsafe_allow_html=True)
        with c6:
            st.markdown(f'<div class="metric-card info"><h4>Trading Days</h4><h2>{len(ts_df)}</h2><div class="sub-metric">Analyzed</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Tabs for different views
        tab3, tab1, tab2, tab4 = st.tabs(["**🎯 Regime Analysis**", "**📈 Zone Trends**", "**📊 Signal Trends**", "**📋 Data Table**"])
        
        with tab1:
            st.markdown("##### Overbought / Oversold Distribution Over Time")
            st.markdown('<p style="color: #888888; font-size: 0.85rem;">Shows the percentage of stocks in each zone daily</p>', unsafe_allow_html=True)
            
            # Stacked area chart for zones
            fig_zones = go.Figure()
            
            fig_zones.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Oversold_Pct'],
                mode='lines', name='Oversold %',
                fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.3)',
                line=dict(color='#10b981', width=2)
            ))
            
            fig_zones.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Overbought_Pct'],
                mode='lines', name='Overbought %',
                fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.3)',
                line=dict(color='#ef4444', width=2)
            ))
            
            fig_zones.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=400,
                margin=dict(l=10, r=10, t=30, b=50),
                xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', title='% of Stocks', range=[0, max(ts_df['Oversold_Pct'].max(), ts_df['Overbought_Pct'].max()) * 1.1]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
                font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified'
            )
            st.plotly_chart(fig_zones, width="stretch", config={'displayModeBar': False})
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Raw Counts Over Time")
            
            # Bar chart for raw counts
            fig_counts = go.Figure()
            
            fig_counts.add_trace(go.Bar(
                x=ts_df['Date'], y=ts_df['Oversold'],
                name='Oversold', 
                marker=dict(color='#10b981', line=dict(color='#10b981', width=1))
            ))
            
            fig_counts.add_trace(go.Bar(
                x=ts_df['Date'], y=ts_df['Overbought'],
                name='Overbought', 
                marker=dict(color='#ef4444', line=dict(color='#ef4444', width=1))
            ))
            
            fig_counts.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=350,
                margin=dict(l=10, r=10, t=10, b=50), barmode='group',
                xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', title='Stock Count'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
                font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified',
                colorway=['#10b981', '#ef4444']  # Ensure colors are applied
            )
            st.plotly_chart(fig_counts, width="stretch", config={'displayModeBar': False})
        
        with tab2:
            st.markdown("##### Buy / Sell Signal Counts Over Time")
            
            fig_signals = go.Figure()
            
            fig_signals.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Buy_Signals'],
                mode='lines+markers', name='Buy Signals',
                line=dict(color='#10b981', width=2),
                marker=dict(size=6, color='#10b981')
            ))
            
            fig_signals.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Sell_Signals'],
                mode='lines+markers', name='Sell Signals',
                line=dict(color='#ef4444', width=2),
                marker=dict(size=6, color='#ef4444')
            ))
            
            fig_signals.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=400,
                margin=dict(l=10, r=10, t=30, b=50),
                xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', title='Signal Count'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
                font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified'
            )
            st.plotly_chart(fig_signals, width="stretch", config={'displayModeBar': False})
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Divergence Signals Over Time")
            
            fig_div = go.Figure()
            
            fig_div.add_trace(go.Bar(
                x=ts_df['Date'], y=ts_df['Bull_Div'],
                name='Bullish Divergence', 
                marker=dict(color='#FFC300', line=dict(color='#FFC300', width=1))
            ))
            
            fig_div.add_trace(go.Bar(
                x=ts_df['Date'], y=-ts_df['Bear_Div'],
                name='Bearish Divergence', 
                marker=dict(color='#06b6d4', line=dict(color='#06b6d4', width=1))
            ))
            
            fig_div.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=300,
                margin=dict(l=10, r=10, t=10, b=50), barmode='relative',
                xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', title='Divergence Count'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
                font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified',
                colorway=['#FFC300', '#06b6d4']
            )
            st.plotly_chart(fig_div, width="stretch", config={'displayModeBar': False})
        
        with tab3:
            # ORIGINAL: Average Signal Value Over Time
            st.markdown("##### Average Signal Value Over Time")
            st.markdown('<p style="color: #888888; font-size: 0.85rem;">Negative = Bullish Bias | Positive = Bearish Bias</p>', unsafe_allow_html=True)
            
            fig_avg = go.Figure()
            
            colors = ['#10b981' if v < -2 else '#ef4444' if v > 2 else '#888888' for v in ts_df['Avg_Signal']]
            
            fig_avg.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Avg_Signal'].clip(lower=0),
                fill='tozeroy', fillcolor='rgba(239,68,68,0.15)',
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            
            fig_avg.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Avg_Signal'].clip(upper=0),
                fill='tozeroy', fillcolor='rgba(16,185,129,0.15)',
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            
            fig_avg.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Avg_Signal'],
                mode='lines+markers', name='Avg Signal',
                line=dict(color='#FFC300', width=2),
                marker=dict(size=6, color=colors)
            ))
            
            fig_avg.add_hline(y=2, line=dict(color='rgba(239,68,68,0.5)', width=1, dash='dash'))
            fig_avg.add_hline(y=-2, line=dict(color='rgba(16,185,129,0.5)', width=1, dash='dash'))
            fig_avg.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', width=1))
            
            fig_avg.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=350,
                margin=dict(l=10, r=10, t=30, b=50),
                xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', title='Average Signal', range=[-8, 8]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
                font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified'
            )
            st.plotly_chart(fig_avg, width="stretch", config={'displayModeBar': False})
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # NEW: HMM Regime Distribution Over Time
            st.markdown("##### HMM Regime Distribution Over Time")
            st.markdown('<p style="color: #888888; font-size: 0.85rem;">Percentage of stocks in each HMM regime daily</p>', unsafe_allow_html=True)
            
            # Regime trend chart
            fig_regime = go.Figure()
            
            fig_regime.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Regime_Bull_Pct'],
                mode='lines', name='Bull Regime %',
                fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.3)',
                line=dict(color='#10b981', width=2)
            ))
            
            fig_regime.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Regime_Bear_Pct'],
                mode='lines', name='Bear Regime %',
                fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.3)',
                line=dict(color='#ef4444', width=2)
            ))
            
            fig_regime.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=300,
                margin=dict(l=10, r=10, t=30, b=50),
                xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', title='% of Stocks', range=[0, 100]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
                font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified'
            )
            st.plotly_chart(fig_regime, width="stretch", config={'displayModeBar': False})
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Volatility Regime & Change Points Over Time")
            
            # Volatility regime chart
            fig_vol = go.Figure()
            
            fig_vol.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Vol_High_Pct'],
                mode='lines+markers', name='High Vol %',
                line=dict(color='#f59e0b', width=2),
                marker=dict(size=5)
            ))
            
            fig_vol.add_trace(go.Bar(
                x=ts_df['Date'], y=ts_df['Change_Points'],
                name='Change Points',
                marker=dict(color='#a855f7', opacity=0.7)
            ))
            
            fig_vol.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=250,
                margin=dict(l=10, r=10, t=10, b=50),
                xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', title='Count / %'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
                font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified'
            )
            st.plotly_chart(fig_vol, width="stretch", config={'displayModeBar': False})
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                st.markdown("##### HMM Regime Statistics")
                avg_bull = ts_df['Regime_Bull_Pct'].mean()
                avg_bear = ts_df['Regime_Bear_Pct'].mean()
                total_changes = ts_df['Change_Points'].sum()
                
                regime_stats = {
                    "Metric": ["Avg Bull Regime %", "Avg Bear Regime %", "Total Change Points", "Avg High Vol %"],
                    "Value": [f"{avg_bull:.1f}%", f"{avg_bear:.1f}%", f"{int(total_changes)}", f"{ts_df['Vol_High_Pct'].mean():.1f}%"]
                }
                st.dataframe(pd.DataFrame(regime_stats), width="stretch", hide_index=True)
            
            with col_r2:
                st.markdown("##### Signal Statistics")
                signal_stats = {
                    "Metric": ["Mean Signal", "Median Signal", "Min Signal", "Max Signal", "Std Dev"],
                    "Value": [
                        f"{ts_df['Avg_Signal'].mean():.2f}",
                        f"{ts_df['Avg_Signal'].median():.2f}",
                        f"{ts_df['Avg_Signal'].min():.2f}",
                        f"{ts_df['Avg_Signal'].max():.2f}",
                        f"{ts_df['Avg_Signal'].std():.2f}"
                    ]
                }
                st.dataframe(pd.DataFrame(signal_stats), width="stretch", hide_index=True)
        
        with tab4:
            st.markdown(f"##### Daily Time Series Data ({len(ts_df)} trading days)")
            
            # Include regime data in display
            display_ts = ts_df[['Date', 'Total_Analyzed', 'Oversold', 'Overbought', 
                               'Buy_Signals', 'Sell_Signals', 'Avg_Signal', 
                               'Regime_Bull', 'Regime_Bear', 'Change_Points']].copy()
            display_ts['Date'] = display_ts['Date'].dt.strftime('%Y-%m-%d')
            display_ts['Avg_Signal'] = display_ts['Avg_Signal'].round(2)
            display_ts.columns = ['Date', 'Stocks', 'Oversold', 'Overbought', 
                                 'Buy Sig', 'Sell Sig', 'Avg Sig', 'Bull Regime', 'Bear Regime', 'Changes']
            
            st.dataframe(display_ts, width="stretch", hide_index=True, height=500)
            
            st.markdown("<br>", unsafe_allow_html=True)
            csv_data = ts_df.to_csv(index=False).encode('utf-8')
            actual_start_str = ts_df['Date'].min().strftime('%Y%m%d')
            actual_end_str = ts_df['Date'].max().strftime('%Y%m%d')
            st.download_button(
                label="📥 Download Time Series Data (CSV)",
                data=csv_data,
                file_name=f"nirnay_market_timeseries_{universe_title.replace(' ', '_')}_{actual_start_str}_{actual_end_str}.csv",
                mime="text/csv"
            )


def run_etf_timeseries_mode(length, roc_len, regime_sensitivity, base_weight, start_date, end_date):
    """ETF Time Series Analysis: Track overbought/oversold signals over time for fixed ETF universe"""
    
    # Validate dates
    if start_date is None or end_date is None:
        st.error("Please select both start and end dates.")
        return
    
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return
    
    if end_date > datetime.date.today():
        st.error("End date cannot be in the future.")
        return
    
    # Calculate date range
    date_range_days = (end_date - start_date).days
    
    # Display info
    st.markdown(f"""
    <div class='info-box'>
        <h4>📈 ETF Time Series Analysis</h4>
        <p>Track overbought/oversold signal distribution across {len(SCREENER_SYMBOLS)} ETFs over time.<br>
        <strong>Period:</strong> {start_date.strftime("%d %b %Y")} to {end_date.strftime("%d %b %Y")} ({date_range_days} days)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("◈ RUN ETF TIME SERIES ANALYSIS", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch macro data
        status_text.markdown("**⏳ Fetching global macro data...**")
        progress_bar.progress(0.05)
        
        days_back = 100 + date_range_days + (datetime.date.today() - end_date).days
        macro_df = fetch_macro_data(days_back=days_back)
        
        # Process each ETF
        status_text.markdown("**⏳ Downloading ETF data...**")
        progress_bar.progress(0.1)
        
        processed_data = {}
        errors = []
        total = len(SCREENER_SYMBOLS)
        
        for i, symbol in enumerate(SCREENER_SYMBOLS):
            status_text.markdown(f"**⏳ Processing {get_display_name(symbol)} ({i+1}/{total})**")
            progress_bar.progress(0.1 + 0.4 * (i + 1) / total)
            
            df = fetch_ticker_data(symbol, macro_df, days_back=days_back)
            
            if df is not None and len(df) > length + 5:
                try:
                    df, _ = run_full_analysis(df, length, roc_len, regime_sensitivity, base_weight)
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    processed_data[symbol] = df
                except Exception as e:
                    errors.append(f"{symbol}: {str(e)[:80]}")
        
        if not processed_data:
            st.error("Failed to process ETF data.")
            progress_bar.empty()
            status_text.empty()
            return
        
        # Generate list of trading days
        status_text.markdown("**⏳ Identifying trading days...**")
        
        sample_ticker = list(processed_data.keys())[0]
        sample_df = processed_data[sample_ticker]
        
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        trading_days = sample_df.index[(sample_df.index >= start_ts) & (sample_df.index <= end_ts)].tolist()
        
        if len(trading_days) == 0:
            st.error("No trading days found in the selected date range.")
            progress_bar.empty()
            status_text.empty()
            return
        
        # Check if requested end date data is available
        actual_last_date = trading_days[-1].date() if trading_days else None
        is_today_included = actual_last_date == datetime.date.today() if actual_last_date else False
        
        if end_date == datetime.date.today():
            if is_today_included:
                st.toast(f"Live Data Included - {actual_last_date.strftime('%d %b %Y')}", icon="🔴")
            else:
                st.toast(f"Data through {actual_last_date.strftime('%d %b %Y')}", icon="⚠️")
        elif actual_last_date and actual_last_date < end_date:
            st.toast(f"Data through {actual_last_date.strftime('%d %b %Y')}", icon="⚠️")
        
        st.toast(f"Found {len(trading_days)} trading days", icon="📅")
        
        # Analyze each trading day
        status_text.markdown(f"**⏳ Analyzing {len(trading_days)} trading days...**")
        
        timeseries_results = []
        
        for day_idx, trading_day in enumerate(trading_days):
            progress_bar.progress(0.5 + 0.45 * (day_idx + 1) / len(trading_days))
            
            day_stats = {
                "Date": trading_day.date(),
                "Oversold": 0,
                "Overbought": 0,
                "Neutral": 0,
                "Buy_Signals": 0,
                "Sell_Signals": 0,
                "Total_Analyzed": 0,
                "Avg_Signal": 0,
                "Signal_Sum": 0,
                "Bull_Div": 0,
                "Bear_Div": 0,
                # Regime Intelligence stats
                "Regime_Bull": 0,
                "Regime_Bear": 0,
                "Regime_Neutral": 0,
                "Regime_Transition": 0,
                "Vol_High": 0,
                "Vol_Low": 0,
                "Change_Points": 0
            }
            
            for symbol, df in processed_data.items():
                try:
                    if trading_day not in df.index:
                        continue
                    
                    row = df.loc[trading_day]
                    
                    day_stats["Total_Analyzed"] += 1
                    day_stats["Signal_Sum"] += row['Unified_Osc']
                    
                    if row['Condition'] == 'Oversold':
                        day_stats["Oversold"] += 1
                    elif row['Condition'] == 'Overbought':
                        day_stats["Overbought"] += 1
                    else:
                        day_stats["Neutral"] += 1
                    
                    if row['Buy_Signal']:
                        day_stats["Buy_Signals"] += 1
                    if row['Sell_Signal']:
                        day_stats["Sell_Signals"] += 1
                    if row['Bullish_Div']:
                        day_stats["Bull_Div"] += 1
                    if row['Bearish_Div']:
                        day_stats["Bear_Div"] += 1
                    
                    # Regime Intelligence stats
                    regime = row['Regime']
                    if 'BULL' in regime:
                        day_stats["Regime_Bull"] += 1
                    elif 'BEAR' in regime:
                        day_stats["Regime_Bear"] += 1
                    elif regime == 'TRANSITION':
                        day_stats["Regime_Transition"] += 1
                    else:
                        day_stats["Regime_Neutral"] += 1
                    
                    vol_regime = row['Vol_Regime']
                    if vol_regime in ['HIGH', 'EXTREME']:
                        day_stats["Vol_High"] += 1
                    elif vol_regime == 'LOW':
                        day_stats["Vol_Low"] += 1
                    
                    if row['Change_Point']:
                        day_stats["Change_Points"] += 1
                        
                except Exception as e:
                    errors.append(f"{symbol}: {str(e)[:80]}")
            
            if day_stats["Total_Analyzed"] > 0:
                day_stats["Avg_Signal"] = day_stats["Signal_Sum"] / day_stats["Total_Analyzed"]
                day_stats["Oversold_Pct"] = (day_stats["Oversold"] / day_stats["Total_Analyzed"]) * 100
                day_stats["Overbought_Pct"] = (day_stats["Overbought"] / day_stats["Total_Analyzed"]) * 100
                day_stats["Neutral_Pct"] = (day_stats["Neutral"] / day_stats["Total_Analyzed"]) * 100
                # Regime percentages
                day_stats["Regime_Bull_Pct"] = (day_stats["Regime_Bull"] / day_stats["Total_Analyzed"]) * 100
                day_stats["Regime_Bear_Pct"] = (day_stats["Regime_Bear"] / day_stats["Total_Analyzed"]) * 100
                day_stats["Vol_High_Pct"] = (day_stats["Vol_High"] / day_stats["Total_Analyzed"]) * 100
            else:
                day_stats["Oversold_Pct"] = 0
                day_stats["Overbought_Pct"] = 0
                day_stats["Neutral_Pct"] = 0
                day_stats["Regime_Bull_Pct"] = 0
                day_stats["Regime_Bear_Pct"] = 0
                day_stats["Vol_High_Pct"] = 0
            
            timeseries_results.append(day_stats)
        
        progress_bar.empty()
        status_text.empty()
        
        if not timeseries_results:
            st.warning("No data could be analyzed for the selected period.")
            return
        
        ts_df = pd.DataFrame(timeseries_results)
        ts_df['Date'] = pd.to_datetime(ts_df['Date'])
        ts_df = ts_df.sort_values('Date')
        
        # Show actual analyzed date range
        actual_start = ts_df['Date'].min().strftime('%d %b %Y')
        actual_end = ts_df['Date'].max().strftime('%d %b %Y')
        st.toast(f"ETF Time Series Complete! {len(ts_df)} days ({actual_start} to {actual_end})", icon="✅")
        
        # Summary metrics
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        
        with c1:
            avg_oversold = ts_df['Oversold_Pct'].mean()
            st.markdown(f'<div class="metric-card success"><h4>Avg Oversold</h4><h2>{avg_oversold:.1f}%</h2><div class="sub-metric">Daily Average</div></div>', unsafe_allow_html=True)
        with c2:
            avg_overbought = ts_df['Overbought_Pct'].mean()
            st.markdown(f'<div class="metric-card danger"><h4>Avg Overbought</h4><h2>{avg_overbought:.1f}%</h2><div class="sub-metric">Daily Average</div></div>', unsafe_allow_html=True)
        with c3:
            total_buys = ts_df['Buy_Signals'].sum()
            st.markdown(f'<div class="metric-card primary"><h4>Total Buy Signals</h4><h2>{total_buys:,}</h2><div class="sub-metric">Over Period</div></div>', unsafe_allow_html=True)
        with c4:
            total_sells = ts_df['Sell_Signals'].sum()
            st.markdown(f'<div class="metric-card warning"><h4>Total Sell Signals</h4><h2>{total_sells:,}</h2><div class="sub-metric">Over Period</div></div>', unsafe_allow_html=True)
        with c5:
            avg_signal = ts_df['Avg_Signal'].mean()
            regime = "BULLISH" if avg_signal < -1 else "BEARISH" if avg_signal > 1 else "NEUTRAL"
            regime_color = "success" if avg_signal < -1 else "danger" if avg_signal > 1 else "neutral"
            st.markdown(f'<div class="metric-card {regime_color}"><h4>Period Regime</h4><h2 style="font-size: 1.1rem;">{regime}</h2><div class="sub-metric">Avg: {avg_signal:.2f}</div></div>', unsafe_allow_html=True)
        with c6:
            st.markdown(f'<div class="metric-card info"><h4>Trading Days</h4><h2>{len(ts_df)}</h2><div class="sub-metric">Analyzed</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Tabs for different views
        tab3, tab1, tab2, tab4 = st.tabs(["**🎯 Regime Analysis**", "**📈 Zone Trends**", "**📊 Signal Trends**", "**📋 Data Table**"])
        
        with tab1:
            st.markdown("##### Overbought / Oversold Distribution Over Time")
            st.markdown('<p style="color: #888888; font-size: 0.85rem;">Shows the percentage of ETFs in each zone daily</p>', unsafe_allow_html=True)
            
            fig_zones = go.Figure()
            
            fig_zones.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Oversold_Pct'],
                mode='lines', name='Oversold %',
                fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.3)',
                line=dict(color='#10b981', width=2)
            ))
            
            fig_zones.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Overbought_Pct'],
                mode='lines', name='Overbought %',
                fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.3)',
                line=dict(color='#ef4444', width=2)
            ))
            
            fig_zones.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=400,
                margin=dict(l=10, r=10, t=30, b=50),
                xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', title='% of ETFs', range=[0, max(ts_df['Oversold_Pct'].max(), ts_df['Overbought_Pct'].max()) * 1.1]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
                font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified'
            )
            st.plotly_chart(fig_zones, width="stretch", config={'displayModeBar': False})
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Raw Counts Over Time")
            
            fig_counts = go.Figure()
            
            fig_counts.add_trace(go.Bar(
                x=ts_df['Date'], y=ts_df['Oversold'],
                name='Oversold', 
                marker=dict(color='#10b981', line=dict(color='#10b981', width=1))
            ))
            
            fig_counts.add_trace(go.Bar(
                x=ts_df['Date'], y=ts_df['Overbought'],
                name='Overbought', 
                marker=dict(color='#ef4444', line=dict(color='#ef4444', width=1))
            ))
            
            fig_counts.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=350,
                margin=dict(l=10, r=10, t=10, b=50), barmode='group',
                xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', title='ETF Count'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
                font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified',
                colorway=['#10b981', '#ef4444']
            )
            st.plotly_chart(fig_counts, width="stretch", config={'displayModeBar': False})
        
        with tab2:
            st.markdown("##### Buy / Sell Signal Counts Over Time")
            
            fig_signals = go.Figure()
            
            fig_signals.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Buy_Signals'],
                mode='lines+markers', name='Buy Signals',
                line=dict(color='#10b981', width=2),
                marker=dict(size=6, color='#10b981')
            ))
            
            fig_signals.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Sell_Signals'],
                mode='lines+markers', name='Sell Signals',
                line=dict(color='#ef4444', width=2),
                marker=dict(size=6, color='#ef4444')
            ))
            
            fig_signals.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=400,
                margin=dict(l=10, r=10, t=30, b=50),
                xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', title='Signal Count'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
                font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified'
            )
            st.plotly_chart(fig_signals, width="stretch", config={'displayModeBar': False})
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Divergence Signals Over Time")
            
            fig_div = go.Figure()
            
            fig_div.add_trace(go.Bar(
                x=ts_df['Date'], y=ts_df['Bull_Div'],
                name='Bullish Divergence', 
                marker=dict(color='#FFC300', line=dict(color='#FFC300', width=1))
            ))
            
            fig_div.add_trace(go.Bar(
                x=ts_df['Date'], y=-ts_df['Bear_Div'],
                name='Bearish Divergence', 
                marker=dict(color='#06b6d4', line=dict(color='#06b6d4', width=1))
            ))
            
            fig_div.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=300,
                margin=dict(l=10, r=10, t=10, b=50), barmode='relative',
                xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', title='Divergence Count'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
                font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified',
                colorway=['#FFC300', '#06b6d4']
            )
            st.plotly_chart(fig_div, width="stretch", config={'displayModeBar': False})
        
        with tab3:
            # ORIGINAL: Average Signal Value Over Time
            st.markdown("##### Average Signal Value Over Time")
            st.markdown('<p style="color: #888888; font-size: 0.85rem;">Negative = Bullish Bias | Positive = Bearish Bias</p>', unsafe_allow_html=True)
            
            fig_avg = go.Figure()
            
            colors = ['#10b981' if v < -2 else '#ef4444' if v > 2 else '#888888' for v in ts_df['Avg_Signal']]
            
            fig_avg.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Avg_Signal'].clip(lower=0),
                fill='tozeroy', fillcolor='rgba(239,68,68,0.15)',
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            
            fig_avg.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Avg_Signal'].clip(upper=0),
                fill='tozeroy', fillcolor='rgba(16,185,129,0.15)',
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            
            fig_avg.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Avg_Signal'],
                mode='lines+markers', name='Avg Signal',
                line=dict(color='#FFC300', width=2),
                marker=dict(size=6, color=colors)
            ))
            
            fig_avg.add_hline(y=2, line=dict(color='rgba(239,68,68,0.5)', width=1, dash='dash'))
            fig_avg.add_hline(y=-2, line=dict(color='rgba(16,185,129,0.5)', width=1, dash='dash'))
            fig_avg.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', width=1))
            
            fig_avg.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=350,
                margin=dict(l=10, r=10, t=30, b=50),
                xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', title='Average Signal', range=[-8, 8]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
                font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified'
            )
            st.plotly_chart(fig_avg, width="stretch", config={'displayModeBar': False})
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # NEW: HMM Regime Distribution Over Time
            st.markdown("##### HMM Regime Distribution Over Time")
            st.markdown('<p style="color: #888888; font-size: 0.85rem;">Percentage of ETFs in each HMM regime daily</p>', unsafe_allow_html=True)
            
            # Regime trend chart
            fig_regime = go.Figure()
            
            fig_regime.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Regime_Bull_Pct'],
                mode='lines', name='Bull Regime %',
                fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.3)',
                line=dict(color='#10b981', width=2)
            ))
            
            fig_regime.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Regime_Bear_Pct'],
                mode='lines', name='Bear Regime %',
                fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.3)',
                line=dict(color='#ef4444', width=2)
            ))
            
            fig_regime.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=300,
                margin=dict(l=10, r=10, t=30, b=50),
                xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', title='% of ETFs', range=[0, 100]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
                font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified'
            )
            st.plotly_chart(fig_regime, width="stretch", config={'displayModeBar': False})
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Volatility Regime & Change Points Over Time")
            
            # Volatility regime chart
            fig_vol = go.Figure()
            
            fig_vol.add_trace(go.Scatter(
                x=ts_df['Date'], y=ts_df['Vol_High_Pct'],
                mode='lines+markers', name='High Vol %',
                line=dict(color='#f59e0b', width=2),
                marker=dict(size=5)
            ))
            
            fig_vol.add_trace(go.Bar(
                x=ts_df['Date'], y=ts_df['Change_Points'],
                name='Change Points',
                marker=dict(color='#a855f7', opacity=0.7)
            ))
            
            fig_vol.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=250,
                margin=dict(l=10, r=10, t=10, b=50),
                xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', title='Count / %'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
                font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified'
            )
            st.plotly_chart(fig_vol, width="stretch", config={'displayModeBar': False})
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                st.markdown("##### HMM Regime Statistics")
                avg_bull = ts_df['Regime_Bull_Pct'].mean()
                avg_bear = ts_df['Regime_Bear_Pct'].mean()
                total_changes = ts_df['Change_Points'].sum()
                
                regime_stats = {
                    "Metric": ["Avg Bull Regime %", "Avg Bear Regime %", "Total Change Points", "Avg High Vol %"],
                    "Value": [f"{avg_bull:.1f}%", f"{avg_bear:.1f}%", f"{int(total_changes)}", f"{ts_df['Vol_High_Pct'].mean():.1f}%"]
                }
                st.dataframe(pd.DataFrame(regime_stats), width="stretch", hide_index=True)
            
            with col_r2:
                st.markdown("##### Signal Statistics")
                signal_stats = {
                    "Metric": ["Mean Signal", "Median Signal", "Min Signal", "Max Signal", "Std Dev"],
                    "Value": [
                        f"{ts_df['Avg_Signal'].mean():.2f}",
                        f"{ts_df['Avg_Signal'].median():.2f}",
                        f"{ts_df['Avg_Signal'].min():.2f}",
                        f"{ts_df['Avg_Signal'].max():.2f}",
                        f"{ts_df['Avg_Signal'].std():.2f}"
                    ]
                }
                st.dataframe(pd.DataFrame(signal_stats), width="stretch", hide_index=True)
        
        with tab4:
            st.markdown(f"##### Daily ETF Time Series Data ({len(ts_df)} trading days)")
            
            # Include regime data in display
            display_ts = ts_df[['Date', 'Total_Analyzed', 'Oversold', 'Overbought', 
                               'Buy_Signals', 'Sell_Signals', 'Avg_Signal', 
                               'Regime_Bull', 'Regime_Bear', 'Change_Points']].copy()
            display_ts['Date'] = display_ts['Date'].dt.strftime('%Y-%m-%d')
            display_ts['Avg_Signal'] = display_ts['Avg_Signal'].round(2)
            display_ts.columns = ['Date', 'ETFs', 'Oversold', 'Overbought', 
                                 'Buy Sig', 'Sell Sig', 'Avg Sig', 'Bull Regime', 'Bear Regime', 'Changes']
            
            st.dataframe(display_ts, width="stretch", hide_index=True, height=500)
            
            st.markdown("<br>", unsafe_allow_html=True)
            csv_data = ts_df.to_csv(index=False).encode('utf-8')
            actual_start_str = ts_df['Date'].min().strftime('%Y%m%d')
            actual_end_str = ts_df['Date'].max().strftime('%Y%m%d')
            st.download_button(
                label="📥 Download ETF Time Series Data (CSV)",
                data=csv_data,
                file_name=f"nirnay_etf_timeseries_{actual_start_str}_{actual_end_str}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
