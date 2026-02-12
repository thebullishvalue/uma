"""
PRAJNA - Data Engine

Unified data fetching and processing:
1. ETF Universe data
2. F&O Stocks data  
3. Index Constituents data
4. Macro indicators (Bonds, Currencies, Commodities)
5. Historical snapshot generation

Version: 1.0.0
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Any
import logging
import warnings
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

ETF_UNIVERSE = [
    "NIFTYIETF.NS", "MON100.NS", "MAKEINDIA.NS", "SILVERIETF.NS",
    "HEALTHIETF.NS", "CONSUMIETF.NS", "GOLDIETF.NS", "INFRAIETF.NS",
    "CPSEETF.NS", "TNIDETF.NS", "COMMOIETF.NS", "MODEFENCE.NS",
    "MOREALTY.NS", "PSUBNKIETF.NS", "MASPTOP50.NS", "FMCGIETF.NS",
    "ITIETF.NS", "EVINDIA.NS", "MNC.NS", "FINIETF.NS",
    "AUTOIETF.NS", "PVTBANIETF.NS", "MONIFTY500.NS", "ECAPINSURE.NS",
    "MIDCAPIETF.NS", "MOSMALL250.NS", "OILIETF.NS", "METALIETF.NS",
    "CHEMICAL.NS", "GROWWPOWER.NS"
]

ETF_NAMES = {
    "NIFTYIETF.NS": "NIFTY 50", "MON100.NS": "NIFTY 100",
    "MAKEINDIA.NS": "Make India", "SILVERIETF.NS": "Silver",
    "HEALTHIETF.NS": "Healthcare", "CONSUMIETF.NS": "Consumer",
    "GOLDIETF.NS": "Gold", "INFRAIETF.NS": "Infra",
    "CPSEETF.NS": "CPSE", "TNIDETF.NS": "TN Index",
    "COMMOIETF.NS": "Commodities", "MODEFENCE.NS": "Defence",
    "MOREALTY.NS": "Realty", "PSUBNKIETF.NS": "PSU Bank",
    "MASPTOP50.NS": "Top 50", "FMCGIETF.NS": "FMCG",
    "ITIETF.NS": "IT", "EVINDIA.NS": "EV India",
    "MNC.NS": "MNC", "FINIETF.NS": "Financial",
    "AUTOIETF.NS": "Auto", "PVTBANIETF.NS": "Pvt Bank",
    "MONIFTY500.NS": "NIFTY 500", "ECAPINSURE.NS": "Insurance",
    "MIDCAPIETF.NS": "Midcap", "MOSMALL250.NS": "Smallcap",
    "OILIETF.NS": "Oil & Gas", "METALIETF.NS": "Metal",
    "CHEMICAL.NS": "Chemical", "GROWWPOWER.NS": "Power"
}

# Macro indicators - Bonds from Stooq, others from Yahoo Finance
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

# Index constituent URLs
INDEX_LIST = [
    "NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY 200", "NIFTY 500",
    "NIFTY MIDCAP 50", "NIFTY MIDCAP 100", "NIFTY SMLCAP 100", "NIFTY BANK",
    "NIFTY AUTO", "NIFTY FIN SERVICE", "NIFTY FMCG", "NIFTY IT",
    "NIFTY MEDIA", "NIFTY METAL", "NIFTY PHARMA"
]

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

UNIVERSE_OPTIONS = ["ETF Universe", "F&O Stocks", "Index Constituents"]


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_display_name(symbol: str) -> str:
    """Get human-readable name for symbol"""
    return ETF_NAMES.get(symbol, symbol.replace(".NS", ""))


def get_fno_stock_list() -> List[str]:
    """Fetch F&O stock list from NSE"""
    try:
        url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        response = session.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                symbols = [item['symbol'] + '.NS' for item in data['data'] if 'symbol' in item]
                return symbols
    except Exception as e:
        logging.warning(f"Failed to fetch F&O list: {e}")
    
    # Fallback to static list
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
        "SUNPHARMA.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS", "BAJFINANCE.NS"
    ]


def get_index_stock_list(index: str) -> List[str]:
    """Fetch index constituent list"""
    try:
        url = INDEX_URL_MAP.get(index)
        if not url:
            return []
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        
        if response.status_code == 200:
            stock_data = pd.read_csv(io.StringIO(response.text))
            
            if 'symbol' in stock_data.columns:
                symbols = stock_data['symbol'].tolist()
            elif 'Symbol' in stock_data.columns:
                symbols = stock_data['Symbol'].tolist()
            else:
                return []
            
            return [f"{s.strip()}.NS" for s in symbols if isinstance(s, str)]
    except Exception as e:
        logging.warning(f"Failed to fetch index constituents: {e}")
    
    return []


def get_universe_symbols(universe_type: str, index_name: str = None) -> Tuple[List[str], str]:
    """
    Get symbols for the selected universe.
    
    Returns:
        Tuple of (symbols_list, status_message)
    """
    if universe_type == "ETF Universe":
        return ETF_UNIVERSE, f"Loaded {len(ETF_UNIVERSE)} ETFs"
    
    elif universe_type == "F&O Stocks":
        symbols = get_fno_stock_list()
        return symbols, f"Loaded {len(symbols)} F&O stocks"
    
    elif universe_type == "Index Constituents" and index_name:
        symbols = get_index_stock_list(index_name)
        return symbols, f"Loaded {len(symbols)} stocks from {index_name}"
    
    return ETF_UNIVERSE, "Default: ETF Universe"


# ══════════════════════════════════════════════════════════════════════════════
# MACRO DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_stooq_data(symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.Series]:
    """Fetch single symbol from Stooq via direct URL"""
    try:
        url = f"https://stooq.com/q/d/l/?s={symbol}&d1={start_date.strftime('%Y%m%d')}&d2={end_date.strftime('%Y%m%d')}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and len(response.text) > 50:
            df = pd.read_csv(io.StringIO(response.text))
            if 'Date' in df.columns and 'Close' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
                return df['Close']
    except Exception as e:
        logging.debug(f"Stooq fetch failed for {symbol}: {e}")
    return None


def fetch_macro_data(days_back: int = 100) -> pd.DataFrame:
    """
    Fetch macro indicator data from Stooq (bonds) and Yahoo Finance (forex, commodities).
    
    Returns:
        DataFrame with macro indicator columns indexed by date
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 30)
    
    macro_df = pd.DataFrame()
    
    # Fetch from Stooq (bonds) - using direct requests
    for name, symbol in MACRO_SYMBOLS_STOOQ.items():
        series = fetch_stooq_data(symbol, start_date, end_date)
        if series is not None and len(series) > 0:
            macro_df[symbol] = series
    
    # Fetch from Yahoo Finance (forex, commodities)
    yf_symbols = list(MACRO_SYMBOLS_YF.values())
    try:
        yf_data = yf.download(yf_symbols, start=start_date, end=end_date, progress=False)
        if not yf_data.empty:
            if 'Close' in yf_data.columns.get_level_values(0):
                yf_close = yf_data['Close']
            else:
                yf_close = yf_data
            yf_close = yf_close.sort_index()
            macro_df = pd.concat([macro_df, yf_close], axis=1)
    except Exception as e:
        logging.warning(f"Yahoo Finance fetch failed: {e}")
    
    # Forward fill missing values
    if not macro_df.empty:
        macro_df = macro_df.ffill().bfill()
    
    return macro_df


def get_macro_columns() -> List[str]:
    """Get list of all macro indicator column names"""
    return list(MACRO_SYMBOLS.values())


# ══════════════════════════════════════════════════════════════════════════════
# SYMBOL DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_symbol_data(
    symbol: str,
    days_back: int = 100,
    end_date: datetime = None
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a single symbol.
    
    Returns:
        DataFrame with Open, High, Low, Close, Volume columns
    """
    if end_date is None:
        end_date = datetime.now()
    
    start_date = end_date - timedelta(days=days_back + 30)
    
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty:
            return None
        
        # Standardize column names
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        
        # Ensure required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                return None
        
        return df[required].dropna()
    
    except Exception as e:
        logging.warning(f"Failed to fetch {symbol}: {e}")
        return None


def fetch_batch_data(
    symbols: List[str],
    days_back: int = 100,
    end_date: datetime = None,
    progress_callback=None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple symbols.
    
    Returns:
        Dict of {symbol: DataFrame}
    """
    if end_date is None:
        end_date = datetime.now()
    
    start_date = end_date - timedelta(days=days_back + 30)
    
    result = {}
    
    try:
        # Batch download
        all_data = yf.download(symbols, start=start_date, end=end_date, progress=False, group_by='ticker')
        
        if all_data.empty:
            return {}
        
        for i, symbol in enumerate(symbols):
            try:
                if symbol in all_data.columns.get_level_values(0):
                    symbol_df = all_data[symbol].copy()
                else:
                    continue
                
                # Standardize
                symbol_df.columns = [c[0] if isinstance(c, tuple) else c for c in symbol_df.columns]
                
                if 'Close' in symbol_df.columns and symbol_df['Close'].notna().sum() > 20:
                    result[symbol] = symbol_df.dropna()
                
                if progress_callback:
                    progress_callback((i + 1) / len(symbols))
            
            except Exception:
                continue
    
    except Exception as e:
        logging.warning(f"Batch download failed: {e}")
    
    return result


def fetch_symbol_with_macro(
    symbol: str,
    macro_df: pd.DataFrame,
    days_back: int = 100,
    end_date: datetime = None
) -> Optional[pd.DataFrame]:
    """
    Fetch symbol data and merge with macro indicators.
    
    Returns:
        DataFrame with OHLCV + macro columns
    """
    df = fetch_symbol_data(symbol, days_back, end_date)
    
    if df is None or df.empty:
        return None
    
    if macro_df is not None and not macro_df.empty:
        merged = df.join(macro_df, how='left')
        macro_cols = [c for c in macro_df.columns if c in merged.columns]
        merged[macro_cols] = merged[macro_cols].ffill().bfill()
        return merged
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL DATA GENERATION (FOR TIME SERIES ANALYSIS)
# ══════════════════════════════════════════════════════════════════════════════

def generate_historical_snapshots(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    progress_callback=None
) -> List[Tuple[datetime, Dict[str, pd.DataFrame]]]:
    """
    Generate historical data snapshots for time series regime analysis.
    
    Returns:
        List of (date, {symbol: DataFrame}) tuples
    """
    # Calculate required lookback
    lookback_days = (end_date - start_date).days + 100
    
    # Fetch all data
    fetch_start = start_date - timedelta(days=100)
    all_data = {}
    
    try:
        raw_data = yf.download(symbols, start=fetch_start, end=end_date, progress=False, group_by='ticker')
        
        if raw_data.empty:
            return []
        
        for symbol in symbols:
            try:
                if symbol in raw_data.columns.get_level_values(0):
                    symbol_df = raw_data[symbol].copy()
                    symbol_df.columns = [c[0] if isinstance(c, tuple) else c for c in symbol_df.columns]
                    if 'Close' in symbol_df.columns:
                        all_data[symbol] = symbol_df.dropna()
            except:
                continue
    
    except Exception as e:
        logging.error(f"Historical data fetch failed: {e}")
        return []
    
    # Generate snapshots for each trading day
    snapshots = []
    trading_days = pd.bdate_range(start=start_date, end=end_date)
    
    for i, date in enumerate(trading_days):
        date_data = {}
        
        for symbol, df in all_data.items():
            # Get data up to this date
            mask = df.index <= date
            if mask.sum() >= 30:
                date_data[symbol] = df[mask].tail(100)
        
        if date_data:
            snapshots.append((date.to_pydatetime(), date_data))
        
        if progress_callback:
            progress_callback((i + 1) / len(trading_days))
    
    return snapshots


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR CALCULATIONS (FOR BREADTH ANALYSIS)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> Optional[pd.Series]:
    """Calculate RSI"""
    close = df['Close'] if 'Close' in df.columns else df['close']
    
    if len(close) < period + 1:
        return None
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_breadth_metrics(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Calculate market breadth metrics from multiple symbols.
    
    Returns:
        Dict with breadth metrics
    """
    total = len(data_dict)
    if total == 0:
        return {}
    
    rsi_above_50 = 0
    above_200ma = 0
    positive_momentum = 0
    
    for symbol, df in data_dict.items():
        try:
            close = df['Close'] if 'Close' in df.columns else df['close']
            
            # RSI
            rsi = calculate_rsi(df)
            if rsi is not None and rsi.iloc[-1] > 50:
                rsi_above_50 += 1
            
            # Above 200 MA
            if len(close) >= 200:
                ma200 = close.rolling(200).mean().iloc[-1]
                if close.iloc[-1] > ma200:
                    above_200ma += 1
            
            # Positive momentum (20-day)
            if len(close) >= 20:
                if close.iloc[-1] > close.iloc[-20]:
                    positive_momentum += 1
        
        except:
            continue
    
    return {
        'rsi_bullish_pct': rsi_above_50 / total * 100,
        'above_200ma_pct': above_200ma / total * 100,
        'positive_momentum_pct': positive_momentum / total * 100,
        'total_symbols': total
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'ETF_UNIVERSE',
    'ETF_NAMES',
    'MACRO_SYMBOLS',
    'INDEX_LIST',
    'UNIVERSE_OPTIONS',
    'get_display_name',
    'get_fno_stock_list',
    'get_index_stock_list',
    'get_universe_symbols',
    'fetch_macro_data',
    'get_macro_columns',
    'fetch_symbol_data',
    'fetch_batch_data',
    'fetch_symbol_with_macro',
    'generate_historical_snapshots',
    'calculate_rsi',
    'calculate_breadth_metrics'
]
