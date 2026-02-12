"""
NIRNAY (निर्णय) - Decisive Market Intelligence

Core Engine: Unified Signal Generation + Regime Intelligence
A Pragyam Product Family Member

This module integrates:
1. MSF (Market Strength Factor) - Momentum, Microstructure, Trend, Flow
2. MMR (Macro-Micro Regime) - Macro correlation-based signals
3. HMM (Hidden Markov Model) - Probabilistic regime states
4. Kalman Filter - Adaptive signal smoothing
5. GARCH - Volatility regime detection
6. CUSUM - Change point detection
7. Bayesian Confidence - Reliability scoring

The key innovation: Signals are REGIME-AWARE and use ADAPTIVE thresholds
instead of fixed values like "overbought > 5".

Version: 1.0.0
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class MarketRegime(Enum):
    """Market regime states discovered by HMM"""
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    WEAK_BULL = "WEAK_BULL"
    NEUTRAL = "NEUTRAL"
    WEAK_BEAR = "WEAK_BEAR"
    BEAR = "BEAR"
    CRISIS = "CRISIS"
    TRANSITION = "TRANSITION"


class SignalType(Enum):
    """Signal classification"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    NEUTRAL = "NEUTRAL"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class VolatilityRegime(Enum):
    """Volatility environment"""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class KalmanState:
    """Kalman filter state for signal smoothing"""
    estimate: float = 0.0
    error_covariance: float = 1.0
    process_variance: float = 0.01
    measurement_variance: float = 0.1


@dataclass
class HMMState:
    """Hidden Markov Model state"""
    n_states: int = 3
    transition_matrix: np.ndarray = None
    emission_means: np.ndarray = None
    emission_stds: np.ndarray = None
    state_probabilities: np.ndarray = None
    
    def __post_init__(self):
        if self.transition_matrix is None:
            self.transition_matrix = np.array([
                [0.85, 0.10, 0.05],
                [0.10, 0.80, 0.10],
                [0.05, 0.10, 0.85]
            ])
        if self.emission_means is None:
            self.emission_means = np.array([0.6, 0.0, -0.6])
        if self.emission_stds is None:
            self.emission_stds = np.array([0.3, 0.25, 0.3])
        if self.state_probabilities is None:
            self.state_probabilities = np.array([0.33, 0.34, 0.33])


@dataclass
class GARCHState:
    """GARCH volatility state"""
    current_variance: float = 0.04
    omega: float = 0.0001
    alpha: float = 0.1
    beta: float = 0.85
    long_term_mean: float = 0.04


@dataclass
class CUSUMState:
    """CUSUM change point detection state"""
    positive_cusum: float = 0.0
    negative_cusum: float = 0.0
    threshold: float = 4.0
    drift: float = 0.5


@dataclass
class SignalComponents:
    """Individual signal components"""
    msf: float  # Market Strength Factor [-1, 1]
    mmr: float  # Macro-Micro Regime [-1, 1]
    momentum: float  # Momentum component
    micro: float  # Microstructure component
    flow: float  # Accumulation/Distribution flow
    unified_raw: float  # Raw unified signal
    unified_filtered: float  # Kalman-filtered signal


@dataclass
class RegimeState:
    """Complete regime state"""
    regime: MarketRegime
    hmm_probabilities: Dict[str, float]
    volatility_regime: VolatilityRegime
    volatility_multiplier: float
    regime_persistence: int
    change_point_detected: bool
    confidence: float


@dataclass
class AdaptiveThresholds:
    """Percentile-based adaptive thresholds"""
    overbought_threshold: float  # e.g., 80th percentile value
    oversold_threshold: float  # e.g., 20th percentile value
    strong_buy_threshold: float
    strong_sell_threshold: float
    signal_percentile: float  # Current signal's percentile rank


@dataclass 
class NirnayResult:
    """Complete NIRNAY analysis result"""
    # Core signal
    signal: SignalType
    signal_strength: float  # -1 to +1
    
    # Signal components
    components: SignalComponents
    
    # Regime context
    regime: RegimeState
    
    # Adaptive thresholds
    thresholds: AdaptiveThresholds
    
    # Recommendations
    action: str
    position_size_factor: float  # 0 to 1, based on confidence
    
    # Metadata
    analysis_date: str
    symbol: str
    warnings: List[str]
    
    # Driver details (for MMR)
    macro_drivers: List[Dict]


# ══════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class MathUtils:
    """Statistical utilities for adaptive analysis"""
    
    @staticmethod
    def sigmoid(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Sigmoid transformation to bound values"""
        return 2.0 / (1.0 + np.exp(-scale * x)) - 1.0
    
    @staticmethod
    def zscore_clipped(series: pd.Series, window: int, clip: float = 3.0) -> pd.Series:
        """Rolling z-score with clipping"""
        roll_mean = series.rolling(window).mean()
        roll_std = series.rolling(window).std()
        z = (series - roll_mean) / roll_std.replace(0, np.nan)
        return z.clip(-clip, clip).fillna(0)
    
    @staticmethod
    def percentile_rank(value: float, history: np.ndarray) -> float:
        """Calculate percentile rank of value within history"""
        if len(history) == 0:
            return 0.5
        return np.sum(history <= value) / len(history)
    
    @staticmethod
    def adaptive_threshold(history: np.ndarray, percentile: float) -> float:
        """Get adaptive threshold based on percentile of history"""
        if len(history) == 0:
            return 0.0
        return np.percentile(history, percentile)
    
    @staticmethod
    def gaussian_pdf(x: float, mean: float, std: float) -> float:
        """Gaussian probability density"""
        if std < 1e-8:
            return 1.0 if abs(x - mean) < 1e-8 else 0.0
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
        """Average True Range calculation"""
        high = df['High'] if 'High' in df.columns else df['high']
        low = df['Low'] if 'Low' in df.columns else df['low']
        close = df['Close'] if 'Close' in df.columns else df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(length).mean()


# ══════════════════════════════════════════════════════════════════════════════
# KALMAN FILTER
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveKalmanFilter:
    """Kalman filter for signal smoothing with adaptive noise estimation"""
    
    def __init__(self, process_var: float = 0.01, measurement_var: float = 0.1):
        self.state = KalmanState(process_variance=process_var, measurement_variance=measurement_var)
        self.innovation_history = []
    
    def update(self, measurement: float) -> float:
        """Update filter with new measurement, return filtered estimate"""
        # Predict
        predicted_estimate = self.state.estimate
        predicted_covariance = self.state.error_covariance + self.state.process_variance
        
        # Innovation
        innovation = measurement - predicted_estimate
        self.innovation_history.append(innovation)
        if len(self.innovation_history) > 50:
            self.innovation_history.pop(0)
        
        # Kalman gain
        innovation_cov = predicted_covariance + self.state.measurement_variance
        kalman_gain = predicted_covariance / innovation_cov
        
        # Update
        self.state.estimate = predicted_estimate + kalman_gain * innovation
        self.state.error_covariance = (1 - kalman_gain) * predicted_covariance
        
        # Adaptive noise estimation
        if len(self.innovation_history) >= 5:
            innovation_var = np.var(self.innovation_history[-min(20, len(self.innovation_history)):])
            self.state.measurement_variance = 0.9 * self.state.measurement_variance + 0.1 * innovation_var
        
        return self.state.estimate
    
    def get_uncertainty(self) -> float:
        return np.sqrt(self.state.error_covariance)
    
    def reset(self, initial: float = 0.0):
        self.state.estimate = initial
        self.state.error_covariance = 1.0
        self.innovation_history = []


# ══════════════════════════════════════════════════════════════════════════════
# HIDDEN MARKOV MODEL
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveHMM:
    """HMM for regime state estimation with online learning"""
    
    def __init__(self):
        self.state = HMMState()
        self.observation_history = []
        self.state_history = []
    
    def emission_prob(self, observation: float, state: int) -> float:
        return MathUtils.gaussian_pdf(
            observation, 
            self.state.emission_means[state],
            self.state.emission_stds[state]
        )
    
    def forward_step(self, observation: float) -> np.ndarray:
        """Single step of forward algorithm"""
        predicted = self.state.transition_matrix.T @ self.state.state_probabilities
        
        emissions = np.array([self.emission_prob(observation, s) for s in range(3)])
        updated = emissions * predicted
        
        total = updated.sum()
        if total > 1e-10:
            updated /= total
        else:
            updated = np.array([0.33, 0.34, 0.33])
        
        self.state.state_probabilities = updated
        return updated
    
    def update(self, observation: float) -> Dict[str, float]:
        """Update HMM with new observation"""
        self.observation_history.append(observation)
        probs = self.forward_step(observation)
        
        most_likely = np.argmax(probs)
        self.state_history.append(most_likely)
        
        # Adapt parameters
        if len(self.observation_history) >= 10:
            self._adapt_emissions()
        if len(self.state_history) >= 5:
            self._adapt_transitions()
        
        return {"BULL": probs[0], "NEUTRAL": probs[1], "BEAR": probs[2]}
    
    def _adapt_emissions(self):
        recent_obs = np.array(self.observation_history[-50:])
        recent_states = self.state_history[-len(recent_obs):]
        
        for state in range(3):
            mask = np.array(recent_states) == state
            if mask.sum() >= 2:
                state_obs = recent_obs[mask]
                new_mean = np.mean(state_obs)
                new_std = max(np.std(state_obs), 0.1)
                self.state.emission_means[state] = 0.9 * self.state.emission_means[state] + 0.1 * new_mean
                self.state.emission_stds[state] = 0.9 * self.state.emission_stds[state] + 0.1 * new_std
    
    def _adapt_transitions(self):
        recent = self.state_history[-30:]
        counts = np.zeros((3, 3))
        for i in range(len(recent) - 1):
            counts[recent[i], recent[i + 1]] += 1
        
        for i in range(3):
            row_sum = counts[i].sum()
            if row_sum >= 2:
                new_probs = (counts[i] + 1) / (row_sum + 3)
                self.state.transition_matrix[i] = 0.8 * self.state.transition_matrix[i] + 0.2 * new_probs
    
    def get_persistence(self) -> int:
        if len(self.state_history) < 2:
            return 1
        current = self.state_history[-1]
        persistence = 1
        for i in range(len(self.state_history) - 2, -1, -1):
            if self.state_history[i] == current:
                persistence += 1
            else:
                break
        return persistence
    
    def reset(self):
        self.state = HMMState()
        self.observation_history = []
        self.state_history = []


# ══════════════════════════════════════════════════════════════════════════════
# GARCH VOLATILITY DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class GARCHDetector:
    """GARCH-inspired volatility regime detection"""
    
    def __init__(self):
        self.state = GARCHState()
        self.shock_history = []
    
    def update(self, shock: float) -> float:
        self.shock_history.append(shock)
        
        shock_sq = shock ** 2
        new_var = self.state.omega + self.state.alpha * shock_sq + self.state.beta * self.state.current_variance
        new_var = np.clip(new_var, 0.001, 1.0)
        self.state.current_variance = new_var
        
        if len(self.shock_history) >= 10:
            realized = np.var(self.shock_history[-min(50, len(self.shock_history)):])
            self.state.long_term_mean = 0.95 * self.state.long_term_mean + 0.05 * realized
        
        return np.sqrt(new_var)
    
    def get_regime(self) -> Tuple[VolatilityRegime, float]:
        """Returns (regime, sensitivity_multiplier)"""
        current_vol = np.sqrt(self.state.current_variance)
        long_term_vol = np.sqrt(self.state.long_term_mean)
        
        ratio = current_vol / long_term_vol if long_term_vol > 0 else 1.0
        
        if ratio < 0.6:
            return VolatilityRegime.LOW, 1.3
        elif ratio < 0.9:
            return VolatilityRegime.NORMAL, 1.0
        elif ratio < 1.4:
            return VolatilityRegime.HIGH, 0.8
        else:
            return VolatilityRegime.EXTREME, 0.6
    
    def reset(self):
        self.state = GARCHState()
        self.shock_history = []


# ══════════════════════════════════════════════════════════════════════════════
# CUSUM CHANGE POINT DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class CUSUMDetector:
    """CUSUM change point detection"""
    
    def __init__(self, threshold: float = 4.0, drift: float = 0.5):
        self.state = CUSUMState(threshold=threshold, drift=drift)
        self.value_history = []
        self.running_mean = 0.0
        self.running_std = 1.0
    
    def update(self, value: float) -> bool:
        self.value_history.append(value)
        
        if len(self.value_history) >= 3:
            recent = self.value_history[-min(20, len(self.value_history)):]
            self.running_mean = np.mean(recent)
            self.running_std = max(np.std(recent), 0.1)
        
        z = (value - self.running_mean) / self.running_std
        
        self.state.positive_cusum = max(0, self.state.positive_cusum + z - self.state.drift)
        self.state.negative_cusum = max(0, self.state.negative_cusum - z - self.state.drift)
        
        change_detected = (
            self.state.positive_cusum > self.state.threshold or
            self.state.negative_cusum > self.state.threshold
        )
        
        if change_detected:
            self.state.positive_cusum = 0
            self.state.negative_cusum = 0
        
        return change_detected
    
    def reset(self):
        self.state = CUSUMState()
        self.value_history = []


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATORS (FROM UMA)
# ══════════════════════════════════════════════════════════════════════════════

class MSFCalculator:
    """
    Market Strength Factor - Multi-component signal generator
    
    Components:
    1. Momentum: Rate of change normalized
    2. Microstructure: Volume-weighted direction vs impact
    3. Trend: Multi-timeframe trend composite
    4. Flow: Accumulation/Distribution + Regime counting
    """
    
    def __init__(self, length: int = 20, roc_len: int = 14):
        self.length = length
        self.roc_len = roc_len
    
    def calculate(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Calculate MSF components.
        Returns: (msf_signal, micro_norm, momentum_norm, flow_norm)
        """
        close = df['Close'] if 'Close' in df.columns else df['close']
        high = df['High'] if 'High' in df.columns else df['high']
        low = df['Low'] if 'Low' in df.columns else df['low']
        open_price = df['Open'] if 'Open' in df.columns else df['open']
        volume = df['Volume'] if 'Volume' in df.columns else df['volume']
        
        # 1. MOMENTUM COMPONENT
        roc_raw = close.pct_change(self.roc_len, fill_method=None)
        roc_z = MathUtils.zscore_clipped(roc_raw, self.length, 3.0)
        momentum_norm = MathUtils.sigmoid(roc_z, 1.5)
        
        # 2. MICROSTRUCTURE COMPONENT
        intrabar_dir = (high + low) / 2 - open_price
        vol_ma = volume.rolling(self.length).mean()
        vol_ratio = (volume / vol_ma).fillna(1.0)
        
        vw_direction = (intrabar_dir * vol_ratio).rolling(self.length).mean()
        price_change_imp = close.diff(5)
        vw_impact = (price_change_imp * vol_ratio).rolling(self.length).mean()
        
        micro_raw = vw_direction - vw_impact
        micro_z = MathUtils.zscore_clipped(micro_raw, self.length, 3.0)
        micro_norm = MathUtils.sigmoid(micro_z, 1.5)
        
        # 3. TREND COMPONENT
        trend_fast = close.rolling(5).mean()
        trend_slow = close.rolling(self.length).mean()
        trend_diff_z = MathUtils.zscore_clipped(trend_fast - trend_slow, self.length, 3.0)
        
        mom_accel_raw = close.diff(5).diff(5)
        mom_accel_z = MathUtils.zscore_clipped(mom_accel_raw, self.length, 3.0)
        
        atr = MathUtils.calculate_atr(df, 14)
        vol_adj_mom_raw = close.diff(5) / atr
        vol_adj_mom_z = MathUtils.zscore_clipped(vol_adj_mom_raw, self.length, 3.0)
        
        mean_rev_z = MathUtils.zscore_clipped(close - trend_slow, self.length, 3.0)
        
        composite_trend_z = (trend_diff_z + mom_accel_z + vol_adj_mom_z + mean_rev_z) / np.sqrt(4.0)
        composite_trend_norm = MathUtils.sigmoid(composite_trend_z, 1.5)
        
        # 4. FLOW COMPONENT
        typical_price = (high + low + close) / 3
        mf = typical_price * volume
        mf_pos = np.where(close > close.shift(1), mf, 0)
        mf_neg = np.where(close < close.shift(1), mf, 0)
        
        mf_pos_smooth = pd.Series(mf_pos, index=df.index).rolling(self.length).mean()
        mf_neg_smooth = pd.Series(mf_neg, index=df.index).rolling(self.length).mean()
        mf_total = mf_pos_smooth + mf_neg_smooth
        
        accum_ratio = mf_pos_smooth / mf_total.replace(0, np.nan)
        accum_ratio = accum_ratio.fillna(0.5)
        accum_norm = 2.0 * (accum_ratio - 0.5)
        
        # Regime counting
        pct_change = close.pct_change(fill_method=None)
        threshold = 0.0033
        regime_signals = np.select([pct_change > threshold, pct_change < -threshold], [1, -1], default=0)
        regime_count = pd.Series(regime_signals, index=df.index).cumsum()
        regime_raw = regime_count - regime_count.rolling(self.length).mean()
        regime_z = MathUtils.zscore_clipped(regime_raw, self.length, 3.0)
        regime_norm = MathUtils.sigmoid(regime_z, 1.5)
        
        flow_norm = (accum_norm + regime_norm) / np.sqrt(2.0)
        
        # COMBINE ALL COMPONENTS
        osc_momentum = momentum_norm
        osc_structure = (micro_norm + composite_trend_norm) / np.sqrt(2.0)
        osc_flow = flow_norm
        
        msf_raw = (osc_momentum + osc_structure + osc_flow) / np.sqrt(3.0)
        msf_signal = MathUtils.sigmoid(msf_raw * np.sqrt(3.0), 1.0)
        
        return msf_signal, micro_norm, momentum_norm, flow_norm


class MMRCalculator:
    """
    Macro-Micro Regime Calculator
    
    Uses rolling regression against macro indicators to determine
    if the asset is trading rich or cheap relative to macro factors.
    """
    
    def __init__(self, length: int = 20, num_vars: int = 5):
        self.length = length
        self.num_vars = num_vars
    
    def calculate(self, df: pd.DataFrame, macro_columns: List[str]) -> Tuple[pd.Series, List[Dict], pd.Series]:
        """
        Calculate MMR signal.
        Returns: (mmr_signal, driver_details, mmr_quality)
        """
        close = df['Close'] if 'Close' in df.columns else df['close']
        
        available_macros = [m for m in macro_columns if m in df.columns]
        
        if len(df) < self.length + 10 or not available_macros:
            return pd.Series(0, index=df.index), [], pd.Series(0, index=df.index)
        
        # Find top correlated macro factors
        correlations = df[available_macros].corrwith(close).abs().sort_values(ascending=False)
        top_drivers = correlations.head(self.num_vars).index.tolist()
        
        preds = []
        r2_sum = 0
        r2_sq_sum = 0
        y_mean = close.rolling(self.length).mean()
        y_std = close.rolling(self.length).std()
        
        driver_details = []
        
        for ticker in top_drivers:
            x = df[ticker]
            x_mean = x.rolling(self.length).mean()
            x_std = x.rolling(self.length).std()
            roll_corr = x.rolling(self.length).corr(close)
            slope = roll_corr * (y_std / x_std)
            intercept = y_mean - (slope * x_mean)
            
            pred = (slope * x) + intercept
            r2 = roll_corr ** 2
            
            preds.append(pred * r2)
            r2_sum += r2
            r2_sq_sum += r2 ** 2
            
            driver_details.append({
                "Symbol": ticker,
                "Correlation": round(df[ticker].corr(close), 4)
            })
        
        r2_sum = r2_sum.replace(0, np.nan)
        
        if len(preds) > 0:
            y_predicted = sum(preds) / r2_sum
        else:
            y_predicted = y_mean
        
        deviation = close - y_predicted
        mmr_z = MathUtils.zscore_clipped(deviation, self.length, 3.0)
        mmr_signal = MathUtils.sigmoid(mmr_z, 1.5)
        
        model_r2 = r2_sq_sum / r2_sum
        mmr_quality = np.sqrt(model_r2.fillna(0))
        
        return mmr_signal, driver_details, mmr_quality


# ══════════════════════════════════════════════════════════════════════════════
# NIRNAY UNIFIED ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class NirnayEngine:
    """
    NIRNAY - Unified Market Intelligence Engine
    
    Combines signal generation (MSF, MMR) with regime intelligence
    (HMM, Kalman, GARCH, CUSUM) for regime-aware, adaptive analysis.
    
    Key Features:
    1. Signals are contextualized by market regime
    2. Thresholds are adaptive (percentile-based)
    3. Confidence weights signal reliability
    4. Change points trigger transition mode
    """
    
    def __init__(
        self,
        msf_length: int = 20,
        roc_len: int = 14,
        regime_sensitivity: float = 1.0,
        base_weight: float = 0.6
    ):
        self.msf_length = msf_length
        self.roc_len = roc_len
        self.regime_sensitivity = regime_sensitivity
        self.base_weight = base_weight
        
        # Signal generators
        self.msf_calc = MSFCalculator(msf_length, roc_len)
        self.mmr_calc = MMRCalculator(msf_length)
        
        # Regime intelligence
        self.kalman = AdaptiveKalmanFilter()
        self.hmm = AdaptiveHMM()
        self.garch = GARCHDetector()
        self.cusum = CUSUMDetector()
        
        # History for adaptive thresholds
        self.signal_history = []
    
    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        macro_columns: List[str] = None
    ) -> NirnayResult:
        """
        Main analysis method.
        
        Args:
            df: OHLCV DataFrame with optional macro columns
            symbol: Symbol being analyzed
            macro_columns: List of macro indicator column names
        
        Returns:
            NirnayResult with complete analysis
        """
        if macro_columns is None:
            macro_columns = []
        
        warnings = []
        
        # 1. CALCULATE SIGNALS
        msf, micro, momentum, flow = self.msf_calc.calculate(df)
        
        if macro_columns:
            mmr, drivers, mmr_quality = self.mmr_calc.calculate(df, macro_columns)
        else:
            mmr = pd.Series(0, index=df.index)
            drivers = []
            mmr_quality = pd.Series(0, index=df.index)
        
        # 2. ADAPTIVE WEIGHTING (from UMA)
        msf_clarity = msf.abs()
        mmr_clarity = mmr.abs()
        msf_clarity_scaled = msf_clarity.pow(self.regime_sensitivity)
        mmr_clarity_scaled = (mmr_clarity * mmr_quality).pow(self.regime_sensitivity)
        clarity_sum = msf_clarity_scaled + mmr_clarity_scaled + 0.001
        
        msf_w_adaptive = msf_clarity_scaled / clarity_sum
        mmr_w_adaptive = mmr_clarity_scaled / clarity_sum
        
        msf_w_final = 0.5 * self.base_weight + 0.5 * msf_w_adaptive
        mmr_w_final = 0.5 * (1.0 - self.base_weight) + 0.5 * mmr_w_adaptive
        w_sum = msf_w_final + mmr_w_final
        msf_w_norm = msf_w_final / w_sum
        mmr_w_norm = mmr_w_final / w_sum
        
        unified_raw = (msf_w_norm * msf) + (mmr_w_norm * mmr)
        
        # Agreement multiplier
        agreement = msf * mmr
        agree_strength = agreement.abs()
        multiplier = np.where(agreement > 0, 1.0 + 0.2 * agree_strength, 1.0 - 0.1 * agree_strength)
        unified_raw = (unified_raw * multiplier).clip(-1.0, 1.0)
        
        # 3. GET LATEST VALUES
        latest_idx = df.index[-1]
        latest_unified = float(unified_raw.iloc[-1])
        latest_msf = float(msf.iloc[-1])
        latest_mmr = float(mmr.iloc[-1])
        latest_momentum = float(momentum.iloc[-1])
        latest_micro = float(micro.iloc[-1])
        latest_flow = float(flow.iloc[-1])
        
        # 4. REGIME INTELLIGENCE
        # Update GARCH with shock
        if self.signal_history:
            shock = latest_unified - self.signal_history[-1]
        else:
            shock = 0.0
        self.garch.update(shock)
        vol_regime, vol_multiplier = self.garch.get_regime()
        
        # Adjust signal for volatility
        adjusted_signal = latest_unified * vol_multiplier
        
        # Kalman filter
        filtered_signal = self.kalman.update(adjusted_signal)
        
        # HMM update
        hmm_probs = self.hmm.update(filtered_signal)
        
        # CUSUM change point
        change_detected = self.cusum.update(filtered_signal)
        if change_detected:
            warnings.append("STRUCTURAL BREAK DETECTED - Market conditions shifting")
            self.kalman.reset(filtered_signal)
        
        # 5. CLASSIFY REGIME
        regime = self._classify_regime(filtered_signal, hmm_probs)
        persistence = self.hmm.get_persistence()
        
        # 6. CALCULATE CONFIDENCE
        confidence = self._calculate_confidence(hmm_probs, filtered_signal)
        
        # 7. ADAPTIVE THRESHOLDS
        self.signal_history.append(filtered_signal)
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]
        
        history_arr = np.array(self.signal_history)
        thresholds = AdaptiveThresholds(
            overbought_threshold=MathUtils.adaptive_threshold(history_arr, 80),
            oversold_threshold=MathUtils.adaptive_threshold(history_arr, 20),
            strong_buy_threshold=MathUtils.adaptive_threshold(history_arr, 10),
            strong_sell_threshold=MathUtils.adaptive_threshold(history_arr, 90),
            signal_percentile=MathUtils.percentile_rank(filtered_signal, history_arr)
        )
        
        # 8. CLASSIFY SIGNAL (REGIME-AWARE)
        signal_type = self._classify_signal(filtered_signal, thresholds, regime)
        
        # 9. GENERATE ACTION
        action, position_factor = self._generate_action(signal_type, regime, confidence, change_detected)
        
        # Add warnings
        if confidence < 0.4:
            warnings.append("LOW CONFIDENCE - Signal reliability reduced")
        if max(hmm_probs.values()) < 0.5:
            warnings.append("REGIME AMBIGUITY - No dominant state")
        if vol_regime == VolatilityRegime.EXTREME:
            warnings.append("EXTREME VOLATILITY - Reduce position sizes")
        
        # Build result
        return NirnayResult(
            signal=signal_type,
            signal_strength=filtered_signal,
            components=SignalComponents(
                msf=latest_msf,
                mmr=latest_mmr,
                momentum=latest_momentum,
                micro=latest_micro,
                flow=latest_flow,
                unified_raw=latest_unified,
                unified_filtered=filtered_signal
            ),
            regime=RegimeState(
                regime=regime,
                hmm_probabilities=hmm_probs,
                volatility_regime=vol_regime,
                volatility_multiplier=vol_multiplier,
                regime_persistence=persistence,
                change_point_detected=change_detected,
                confidence=confidence
            ),
            thresholds=thresholds,
            action=action,
            position_size_factor=position_factor,
            analysis_date=str(latest_idx.date()) if hasattr(latest_idx, 'date') else str(latest_idx),
            symbol=symbol,
            warnings=warnings,
            macro_drivers=drivers
        )
    
    def _classify_regime(self, score: float, hmm_probs: Dict[str, float]) -> MarketRegime:
        """Classify market regime from HMM and score"""
        bull_prob = hmm_probs['BULL']
        bear_prob = hmm_probs['BEAR']
        neutral_prob = hmm_probs['NEUTRAL']
        
        max_prob = max(bull_prob, bear_prob, neutral_prob)
        if max_prob < 0.4:
            return MarketRegime.TRANSITION
        
        score_pct = MathUtils.percentile_rank(score, np.array(self.signal_history)) if self.signal_history else 0.5
        
        if bull_prob > bear_prob and bull_prob > neutral_prob:
            if score_pct >= 0.9:
                return MarketRegime.STRONG_BULL
            elif score_pct >= 0.7:
                return MarketRegime.BULL
            else:
                return MarketRegime.WEAK_BULL
        elif bear_prob > bull_prob and bear_prob > neutral_prob:
            if score_pct <= 0.1:
                return MarketRegime.CRISIS
            elif score_pct <= 0.3:
                return MarketRegime.BEAR
            else:
                return MarketRegime.WEAK_BEAR
        else:
            return MarketRegime.NEUTRAL
    
    def _calculate_confidence(self, hmm_probs: Dict[str, float], signal: float) -> float:
        """Calculate Bayesian confidence"""
        probs = np.array(list(hmm_probs.values()))
        hmm_conf = max(probs) - np.median(probs)
        
        history_len = len(self.signal_history)
        data_conf = min(1.0, history_len / 50)
        
        kalman_unc = self.kalman.get_uncertainty()
        kalman_conf = max(0, 1.0 - kalman_unc)
        
        return np.clip(0.4 * hmm_conf + 0.3 * data_conf + 0.3 * kalman_conf, 0, 1)
    
    def _classify_signal(
        self, 
        signal: float, 
        thresholds: AdaptiveThresholds,
        regime: MarketRegime
    ) -> SignalType:
        """
        Classify signal type using ADAPTIVE thresholds and REGIME context.
        
        This is the key innovation: signals are interpreted differently
        based on the current market regime.
        """
        pct = thresholds.signal_percentile
        
        # Regime adjustment factors
        if regime in [MarketRegime.BULL, MarketRegime.STRONG_BULL]:
            # In bull markets, buy signals are more reliable
            buy_boost = 0.1
            sell_penalty = 0.1
        elif regime in [MarketRegime.BEAR, MarketRegime.CRISIS]:
            # In bear markets, sell signals are more reliable
            buy_boost = -0.1
            sell_penalty = -0.1
        else:
            buy_boost = 0
            sell_penalty = 0
        
        adjusted_pct = pct + buy_boost - sell_penalty
        
        if adjusted_pct <= 0.10:
            return SignalType.STRONG_BUY
        elif adjusted_pct <= 0.25:
            return SignalType.BUY
        elif adjusted_pct <= 0.40:
            return SignalType.WEAK_BUY
        elif adjusted_pct <= 0.60:
            return SignalType.NEUTRAL
        elif adjusted_pct <= 0.75:
            return SignalType.WEAK_SELL
        elif adjusted_pct <= 0.90:
            return SignalType.SELL
        else:
            return SignalType.STRONG_SELL
    
    def _generate_action(
        self,
        signal: SignalType,
        regime: MarketRegime,
        confidence: float,
        change_detected: bool
    ) -> Tuple[str, float]:
        """Generate actionable recommendation"""
        
        # Base position factor from signal
        signal_factors = {
            SignalType.STRONG_BUY: 1.0,
            SignalType.BUY: 0.75,
            SignalType.WEAK_BUY: 0.5,
            SignalType.NEUTRAL: 0.0,
            SignalType.WEAK_SELL: -0.5,
            SignalType.SELL: -0.75,
            SignalType.STRONG_SELL: -1.0
        }
        base_factor = signal_factors.get(signal, 0)
        
        # Adjust for confidence
        position_factor = abs(base_factor) * confidence
        
        # Reduce during transitions
        if change_detected or regime == MarketRegime.TRANSITION:
            position_factor *= 0.5
        
        # Generate action text
        if signal in [SignalType.STRONG_BUY, SignalType.BUY]:
            if regime in [MarketRegime.BULL, MarketRegime.STRONG_BULL]:
                action = f"🟢 BUY - Aligned with {regime.value} regime"
            elif regime in [MarketRegime.BEAR, MarketRegime.CRISIS]:
                action = f"🟡 CAUTIOUS BUY - Counter-trend in {regime.value}"
            else:
                action = f"🟢 BUY - {signal.value} signal"
        
        elif signal in [SignalType.STRONG_SELL, SignalType.SELL]:
            if regime in [MarketRegime.BEAR, MarketRegime.CRISIS]:
                action = f"🔴 SELL - Aligned with {regime.value} regime"
            elif regime in [MarketRegime.BULL, MarketRegime.STRONG_BULL]:
                action = f"🟡 CAUTIOUS SELL - Counter-trend in {regime.value}"
            else:
                action = f"🔴 SELL - {signal.value} signal"
        
        elif signal in [SignalType.WEAK_BUY, SignalType.WEAK_SELL]:
            action = f"📊 MONITOR - Weak {signal.value} signal, wait for confirmation"
        
        else:
            action = "📊 HOLD - Neutral conditions"
        
        if change_detected:
            action += " ⚡ [TRANSITION]"
        
        return action, position_factor
    
    def reset(self):
        """Reset all adaptive components"""
        self.kalman.reset()
        self.hmm.reset()
        self.garch.reset()
        self.cusum.reset()
        self.signal_history = []


# ══════════════════════════════════════════════════════════════════════════════
# BATCH ANALYSIS FOR SCREENER MODE
# ══════════════════════════════════════════════════════════════════════════════

def run_batch_analysis(
    data_dict: Dict[str, pd.DataFrame],
    macro_df: pd.DataFrame = None,
    msf_length: int = 20,
    roc_len: int = 14
) -> List[Dict]:
    """
    Run NIRNAY analysis on multiple symbols for screener mode.
    
    Args:
        data_dict: Dict of {symbol: DataFrame}
        macro_df: Optional macro data to merge
        msf_length: MSF lookback period
        roc_len: Rate of change period
    
    Returns:
        List of result dictionaries
    """
    results = []
    engine = NirnayEngine(msf_length, roc_len)
    
    macro_columns = list(macro_df.columns) if macro_df is not None else []
    
    for symbol, df in data_dict.items():
        try:
            # Merge macro data if available
            if macro_df is not None and not macro_df.empty:
                merged_df = df.join(macro_df, how='left')
                merged_df[macro_columns] = merged_df[macro_columns].ffill()
            else:
                merged_df = df
            
            result = engine.analyze(merged_df, symbol, macro_columns)
            
            results.append({
                'symbol': symbol,
                'signal': result.signal.value,
                'signal_strength': result.signal_strength,
                'msf': result.components.msf,
                'mmr': result.components.mmr,
                'momentum': result.components.momentum,
                'micro': result.components.micro,
                'flow': result.components.flow,
                'regime': result.regime.regime.value,
                'hmm_bull': result.regime.hmm_probabilities['BULL'],
                'hmm_bear': result.regime.hmm_probabilities['BEAR'],
                'volatility': result.regime.volatility_regime.value,
                'confidence': result.regime.confidence,
                'action': result.action,
                'position_factor': result.position_size_factor,
                'warnings': result.warnings
            })
            
            # Reset engine for next symbol
            engine.reset()
            
        except Exception as e:
            results.append({
                'symbol': symbol,
                'signal': 'ERROR',
                'signal_strength': 0,
                'error': str(e)
            })
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'NirnayEngine',
    'NirnayResult',
    'MarketRegime',
    'SignalType',
    'VolatilityRegime',
    'SignalComponents',
    'RegimeState',
    'AdaptiveThresholds',
    'MSFCalculator',
    'MMRCalculator',
    'AdaptiveKalmanFilter',
    'AdaptiveHMM',
    'GARCHDetector',
    'CUSUMDetector',
    'MathUtils',
    'run_batch_analysis'
]
