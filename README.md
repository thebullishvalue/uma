# ◈ NIRNAY (निर्णय) - Unified Market Analysis

**Quantitative Signal + Regime Intelligence System**  
**A Pragyam Product Family Member**

Version: 1.0.0 - Unified Intelligence

---

## Overview

NIRNAY combines signal generation from UMA with regime intelligence from AVASTHA into a unified market analysis system. It provides the same features and UI/UX as UMA, but with enhanced regime-aware analysis underneath.

## Key Features

### 🏦 ETF Screener
- Full MSF + MMR + Regime analysis across 30 curated ETFs
- Single Day and Time Series modes
- Macro correlation analysis
- HMM regime detection

### 📊 Market Screener  
- MSF-based signal analysis
- F&O stocks and Index constituents universe
- Time Series tracking
- Volatility regime (GARCH)

### 📈 Chart Analysis
- Deep dive into individual securities
- Price & oscillator charts
- HMM state probabilities
- CUSUM change point detection
- Macro driver correlations

---

## Analysis Methodology

### Signal Generation (from UMA)

**MSF - Market Structure & Flow**
- Momentum Analysis (ROC dynamics)
- Microstructure (Price efficiency)
- Flow Detection (Volume-weighted)

**MMR - Macro Market Regression**
- Bond Markets (US/IN 10Y yields)
- Currencies (DXY, USD/INR)
- Commodities (Gold, Crude)

### Regime Intelligence (from AVASTHA)

**HMM - Hidden Markov Model**
- 3-state regime discovery (BULL/NEUTRAL/BEAR)
- Online learning with adaptive parameters
- State probability tracking

**GARCH - Volatility Regime**
- GARCH(1,1) volatility estimation
- Regime classification (LOW/NORMAL/HIGH/EXTREME)
- Dynamic signal scaling

**CUSUM - Change Point Detection**
- Cumulative sum monitoring
- Structural break detection
- Regime transition alerts

**Kalman Filter**
- Adaptive signal smoothing
- Noise estimation
- Real-time filtering

---

## Signal Interpretation

| Zone | Signal Range | Interpretation |
|------|-------------|----------------|
| 🟢 Oversold | < -5 | Potential buying opportunity |
| ⚪ Neutral | -5 to +5 | No clear directional bias |
| 🔴 Overbought | > +5 | Potential selling opportunity |

### Regime States

| Regime | Description |
|--------|-------------|
| BULL | Strong bullish state (P > 0.6) |
| WEAK_BULL | Moderate bullish bias |
| NEUTRAL | No clear direction |
| WEAK_BEAR | Moderate bearish bias |
| BEAR | Strong bearish state (P > 0.6) |
| TRANSITION | Change point detected |

---

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Requirements

- streamlit>=1.28.0
- pandas>=2.0.0
- numpy>=1.24.0
- yfinance>=0.2.31
- plotly>=5.18.0
- requests>=2.31.0

## File Structure

```
nirnay/
├── app.py           # Complete application (single file)
├── requirements.txt # Dependencies
└── README.md        # Documentation
```

---

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Lookback Period | 20 | 10-50 | MSF calculation window |
| ROC Length | 14 | 5-30 | Rate of change period |
| Regime Sensitivity | 1.5 | 0.5-3.0 | Adaptive weighting power |
| Base MSF Weight | 0.5 | 0.0-1.0 | MSF vs MMR base allocation |

---

## License

Proprietary - Pragyam Product Family
