"""
PRAJNA - Chart Components

Unified visualization for all analysis modes:
1. Price charts with signals
2. Oscillator charts (MSF, MMR, Unified)
3. Regime gauges and indicators
4. Heatmaps and distributions
5. HMM probability visualizations

Version: 1.0.0
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# Color scheme - Pragyam Design System
COLORS = {
    'primary': '#FFC300',
    'background': '#0F0F0F',
    'card': '#1A1A1A',
    'border': '#2A2A2A',
    'text': '#EAEAEA',
    'muted': '#888888',
    'success': '#10b981',
    'danger': '#ef4444',
    'warning': '#f59e0b',
    'info': '#06b6d4',
    'bull': '#10b981',
    'bear': '#ef4444',
    'neutral': '#888888'
}


def create_price_chart(df: pd.DataFrame, symbol: str, signals: pd.DataFrame = None) -> go.Figure:
    """
    Create interactive price chart with optional signal markers.
    """
    close = df['Close'] if 'Close' in df.columns else df['close']
    high = df['High'] if 'High' in df.columns else df['high']
    low = df['Low'] if 'Low' in df.columns else df['low']
    open_price = df['Open'] if 'Open' in df.columns else df['open']
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=open_price,
        high=high,
        low=low,
        close=close,
        name=symbol,
        increasing_line_color=COLORS['success'],
        decreasing_line_color=COLORS['danger'],
        increasing_fillcolor=COLORS['success'],
        decreasing_fillcolor=COLORS['danger']
    ))
    
    # Moving averages
    if len(close) >= 20:
        ma20 = close.rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=ma20, mode='lines',
            name='MA20', line=dict(color=COLORS['warning'], width=1)
        ))
    
    if len(close) >= 50:
        ma50 = close.rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=ma50, mode='lines',
            name='MA50', line=dict(color=COLORS['info'], width=1)
        ))
    
    # Signal markers
    if signals is not None and 'buy' in signals.columns:
        buy_mask = signals['buy'] == True
        if buy_mask.any():
            fig.add_trace(go.Scatter(
                x=df.index[buy_mask], y=low[buy_mask] * 0.99,
                mode='markers', name='Buy',
                marker=dict(symbol='triangle-up', size=12, color=COLORS['success'])
            ))
    
    if signals is not None and 'sell' in signals.columns:
        sell_mask = signals['sell'] == True
        if sell_mask.any():
            fig.add_trace(go.Scatter(
                x=df.index[sell_mask], y=high[sell_mask] * 1.01,
                mode='markers', name='Sell',
                marker=dict(symbol='triangle-down', size=12, color=COLORS['danger'])
            ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS['card'],
        height=400,
        margin=dict(l=10, r=10, t=30, b=30),
        xaxis=dict(showgrid=True, gridcolor=COLORS['border'], rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor=COLORS['border'], title='Price'),
        font=dict(family='Inter', color=COLORS['text']),
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
        hovermode='x unified'
    )
    
    return fig


def create_oscillator_chart(
    df: pd.DataFrame,
    msf: pd.Series = None,
    mmr: pd.Series = None,
    unified: pd.Series = None
) -> go.Figure:
    """
    Create multi-panel oscillator chart showing MSF, MMR, and Unified signals.
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=('Unified Signal', 'MSF (Market Strength)', 'MMR (Macro Regime)')
    )
    
    # Unified oscillator
    if unified is not None:
        unified_osc = unified * 10
        colors = [COLORS['success'] if v >= 0 else COLORS['danger'] for v in unified_osc]
        fig.add_trace(go.Bar(
            x=df.index, y=unified_osc, name='Unified',
            marker_color=colors, opacity=0.8
        ), row=1, col=1)
        
        # Threshold lines
        fig.add_hline(y=5, line_dash='dash', line_color=COLORS['warning'], row=1, col=1)
        fig.add_hline(y=-5, line_dash='dash', line_color=COLORS['info'], row=1, col=1)
        fig.add_hline(y=0, line_color=COLORS['muted'], row=1, col=1)
    
    # MSF
    if msf is not None:
        msf_osc = msf * 10
        fig.add_trace(go.Scatter(
            x=df.index, y=msf_osc, mode='lines',
            name='MSF', line=dict(color=COLORS['primary'], width=2),
            fill='tozeroy', fillcolor='rgba(255,195,0,0.2)'
        ), row=2, col=1)
        fig.add_hline(y=0, line_color=COLORS['muted'], row=2, col=1)
    
    # MMR
    if mmr is not None:
        mmr_osc = mmr * 10
        fig.add_trace(go.Scatter(
            x=df.index, y=mmr_osc, mode='lines',
            name='MMR', line=dict(color=COLORS['info'], width=2),
            fill='tozeroy', fillcolor='rgba(6,182,212,0.2)'
        ), row=3, col=1)
        fig.add_hline(y=0, line_color=COLORS['muted'], row=3, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS['card'],
        height=450,
        margin=dict(l=10, r=10, t=40, b=30),
        font=dict(family='Inter', color=COLORS['text']),
        showlegend=True,
        legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
        hovermode='x unified'
    )
    
    for i in range(1, 4):
        fig.update_xaxes(showgrid=True, gridcolor=COLORS['border'], row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=COLORS['border'], row=i, col=1)
    
    return fig


def create_signal_gauge(value: float, title: str = "Signal") -> go.Figure:
    """
    Create a gauge chart for signal strength visualization.
    Value should be in range [-1, 1] or [-10, 10].
    """
    # Normalize to [-10, 10] if needed
    if -1 <= value <= 1:
        display_value = value * 10
    else:
        display_value = value
    
    # Determine color
    if display_value < -5:
        color = COLORS['info']  # Oversold
    elif display_value > 5:
        color = COLORS['warning']  # Overbought
    elif display_value >= 0:
        color = COLORS['success']
    else:
        color = COLORS['danger']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_value,
        number={'suffix': '', 'font': {'size': 28, 'color': COLORS['text']}},
        gauge={
            'axis': {'range': [-10, 10], 'tickwidth': 1, 'tickcolor': COLORS['muted']},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': COLORS['card'],
            'borderwidth': 2,
            'bordercolor': COLORS['border'],
            'steps': [
                {'range': [-10, -5], 'color': 'rgba(6,182,212,0.3)'},
                {'range': [-5, 0], 'color': 'rgba(239,68,68,0.2)'},
                {'range': [0, 5], 'color': 'rgba(16,185,129,0.2)'},
                {'range': [5, 10], 'color': 'rgba(245,158,11,0.3)'}
            ],
            'threshold': {
                'line': {'color': COLORS['primary'], 'width': 4},
                'thickness': 0.8,
                'value': display_value
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color=COLORS['text']),
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_regime_gauge(score: float, confidence: float) -> go.Figure:
    """
    Create a regime strength gauge with confidence indicator.
    """
    # Map score to 0-100 range
    gauge_value = (score + 1) / 2 * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        number={'suffix': '%', 'font': {'size': 24, 'color': COLORS['text']}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': COLORS['primary'], 'thickness': 0.6},
            'bgcolor': COLORS['card'],
            'borderwidth': 2,
            'bordercolor': COLORS['border'],
            'steps': [
                {'range': [0, 25], 'color': 'rgba(239,68,68,0.4)'},
                {'range': [25, 40], 'color': 'rgba(239,68,68,0.2)'},
                {'range': [40, 60], 'color': 'rgba(136,136,136,0.2)'},
                {'range': [60, 75], 'color': 'rgba(16,185,129,0.2)'},
                {'range': [75, 100], 'color': 'rgba(16,185,129,0.4)'}
            ]
        }
    ))
    
    # Add confidence annotation
    fig.add_annotation(
        x=0.5, y=-0.1,
        text=f"Confidence: {confidence*100:.0f}%",
        showarrow=False,
        font=dict(size=12, color=COLORS['muted'])
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color=COLORS['text']),
        height=220,
        margin=dict(l=20, r=20, t=40, b=40)
    )
    
    return fig


def create_hmm_probability_chart(hmm_probs: Dict[str, float]) -> go.Figure:
    """
    Create horizontal bar chart for HMM state probabilities.
    """
    states = list(hmm_probs.keys())
    probs = [hmm_probs[s] * 100 for s in states]
    colors = [COLORS['success'] if s == 'BULL' else COLORS['danger'] if s == 'BEAR' else COLORS['neutral'] for s in states]
    
    fig = go.Figure(go.Bar(
        x=probs, y=states, orientation='h',
        marker_color=colors,
        text=[f"{p:.1f}%" for p in probs],
        textposition='outside',
        textfont=dict(color=COLORS['text'], size=14)
    ))
    
    fig.add_vline(x=50, line_dash='dash', line_color=COLORS['primary'], opacity=0.7)
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS['card'],
        font=dict(family='Inter', color=COLORS['text']),
        height=180,
        margin=dict(l=80, r=60, t=20, b=20),
        xaxis=dict(range=[0, 100], showgrid=True, gridcolor=COLORS['border'], title='Probability %'),
        yaxis=dict(gridcolor=COLORS['border'])
    )
    
    return fig


def create_component_radar(components: Dict[str, float]) -> go.Figure:
    """
    Create radar chart for signal components.
    """
    categories = list(components.keys())
    values = [(v + 1) / 2 * 100 for v in components.values()]  # Map -1,1 to 0,100
    
    # Close the radar
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories,
        fill='toself', fillcolor='rgba(255,195,0,0.2)',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=8, color=COLORS['primary'])
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                ticktext=['Bear', '', 'Neutral', '', 'Bull'],
                gridcolor=COLORS['border'],
                linecolor=COLORS['border'],
                tickfont=dict(size=9, color=COLORS['muted'])
            ),
            angularaxis=dict(
                gridcolor=COLORS['border'],
                linecolor=COLORS['border'],
                tickfont=dict(size=10, color=COLORS['text'])
            ),
            bgcolor=COLORS['card']
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color=COLORS['text']),
        height=350,
        margin=dict(l=60, r=60, t=30, b=30),
        showlegend=False
    )
    
    return fig


def create_heatmap(results_df: pd.DataFrame, value_col: str = 'signal_strength') -> go.Figure:
    """
    Create heatmap visualization for screener results.
    """
    if results_df.empty:
        return go.Figure()
    
    # Sort by value
    sorted_df = results_df.sort_values(value_col, ascending=False)
    
    symbols = sorted_df['symbol'].apply(lambda x: x.replace('.NS', '')).tolist()
    values = sorted_df[value_col].tolist()
    
    # Create grid
    n_cols = min(8, len(symbols))
    n_rows = (len(symbols) + n_cols - 1) // n_cols
    
    # Pad to fill grid
    while len(symbols) < n_rows * n_cols:
        symbols.append('')
        values.append(0)
    
    # Reshape
    z = np.array(values).reshape(n_rows, n_cols)
    text = np.array(symbols).reshape(n_rows, n_cols)
    
    fig = go.Figure(go.Heatmap(
        z=z,
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=10, color='white'),
        colorscale=[
            [0.0, COLORS['danger']],
            [0.5, COLORS['neutral']],
            [1.0, COLORS['success']]
        ],
        zmid=0,
        showscale=True,
        colorbar=dict(
            title=dict(text='Signal', font=dict(color=COLORS['muted'])),
            tickfont=dict(color=COLORS['muted'])
        )
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS['card'],
        font=dict(family='Inter', color=COLORS['text']),
        height=max(200, n_rows * 50),
        margin=dict(l=10, r=80, t=30, b=10),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, autorange='reversed')
    )
    
    return fig


def create_distribution_chart(results_df: pd.DataFrame) -> go.Figure:
    """
    Create distribution chart of signals across universe.
    """
    if results_df.empty or 'signal' not in results_df.columns:
        return go.Figure()
    
    signal_counts = results_df['signal'].value_counts()
    
    # Color mapping
    color_map = {
        'STRONG_BUY': COLORS['success'],
        'BUY': '#34d399',
        'WEAK_BUY': '#86efac',
        'NEUTRAL': COLORS['neutral'],
        'WEAK_SELL': '#fbbf24',
        'SELL': '#f87171',
        'STRONG_SELL': COLORS['danger']
    }
    
    colors = [color_map.get(s, COLORS['neutral']) for s in signal_counts.index]
    
    fig = go.Figure(go.Pie(
        labels=signal_counts.index,
        values=signal_counts.values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color=COLORS['card'], width=2)),
        textinfo='label+percent',
        textfont=dict(size=11, color='white')
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color=COLORS['text']),
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False
    )
    
    return fig


def create_regime_distribution_chart(results: List[Dict]) -> go.Figure:
    """
    Create regime distribution pie chart from time series results.
    """
    if not results:
        return go.Figure()
    
    regime_counts = {}
    for r in results:
        regime = r.get('regime', 'UNKNOWN')
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    color_map = {
        'STRONG_BULL': COLORS['success'],
        'BULL': '#34d399',
        'WEAK_BULL': '#86efac',
        'NEUTRAL': COLORS['neutral'],
        'WEAK_BEAR': '#fbbf24',
        'BEAR': '#f87171',
        'CRISIS': COLORS['danger'],
        'TRANSITION': '#a855f7'
    }
    
    labels = list(regime_counts.keys())
    values = list(regime_counts.values())
    colors = [color_map.get(l, COLORS['neutral']) for l in labels]
    
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.5,
        marker=dict(colors=colors, line=dict(color=COLORS['card'], width=2)),
        textinfo='label+percent',
        textfont=dict(size=11, color='white')
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color=COLORS['text']),
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False
    )
    
    return fig


def create_time_series_chart(results: List[Dict]) -> go.Figure:
    """
    Create time series chart showing signal/score evolution with regime zones.
    """
    if not results:
        return go.Figure()
    
    dates = [r['date'] for r in results]
    scores = [r.get('signal_strength', r.get('score', 0)) for r in results]
    regimes = [r.get('regime', 'NEUTRAL') for r in results]
    
    fig = go.Figure()
    
    # Background regime zones
    regime_colors = {
        'STRONG_BULL': 'rgba(16,185,129,0.15)',
        'BULL': 'rgba(16,185,129,0.1)',
        'WEAK_BULL': 'rgba(16,185,129,0.05)',
        'NEUTRAL': 'rgba(136,136,136,0.05)',
        'WEAK_BEAR': 'rgba(239,68,68,0.05)',
        'BEAR': 'rgba(239,68,68,0.1)',
        'CRISIS': 'rgba(239,68,68,0.15)',
        'TRANSITION': 'rgba(168,85,247,0.1)'
    }
    
    # Score line
    fig.add_trace(go.Scatter(
        x=dates, y=scores, mode='lines',
        name='Signal Score',
        line=dict(color=COLORS['primary'], width=2),
        fill='tozeroy',
        fillcolor='rgba(255,195,0,0.1)'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash='dash', line_color=COLORS['muted'])
    
    # Threshold lines
    fig.add_hline(y=0.5, line_dash='dot', line_color=COLORS['success'], opacity=0.5)
    fig.add_hline(y=-0.5, line_dash='dot', line_color=COLORS['danger'], opacity=0.5)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS['card'],
        font=dict(family='Inter', color=COLORS['text']),
        height=350,
        margin=dict(l=10, r=10, t=30, b=50),
        xaxis=dict(showgrid=True, gridcolor=COLORS['border']),
        yaxis=dict(showgrid=True, gridcolor=COLORS['border'], title='Score'),
        hovermode='x unified',
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center')
    )
    
    return fig


def create_ranking_chart(results_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Create horizontal bar chart ranking top/bottom symbols.
    """
    if results_df.empty:
        return go.Figure()
    
    # Sort by signal strength
    sorted_df = results_df.sort_values('signal_strength', ascending=True)
    
    # Get top and bottom
    top = sorted_df.tail(top_n)
    bottom = sorted_df.head(top_n)
    
    combined = pd.concat([bottom, top])
    
    symbols = combined['symbol'].apply(lambda x: x.replace('.NS', '')).tolist()
    values = combined['signal_strength'].tolist()
    colors = [COLORS['success'] if v >= 0 else COLORS['danger'] for v in values]
    
    fig = go.Figure(go.Bar(
        x=values, y=symbols, orientation='h',
        marker_color=colors,
        text=[f"{v:+.2f}" for v in values],
        textposition='outside',
        textfont=dict(color=COLORS['text'])
    ))
    
    fig.add_vline(x=0, line_color=COLORS['muted'])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS['card'],
        font=dict(family='Inter', color=COLORS['text']),
        height=max(400, len(combined) * 25),
        margin=dict(l=100, r=60, t=30, b=30),
        xaxis=dict(showgrid=True, gridcolor=COLORS['border'], title='Signal Strength'),
        yaxis=dict(gridcolor=COLORS['border'])
    )
    
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'COLORS',
    'create_price_chart',
    'create_oscillator_chart',
    'create_signal_gauge',
    'create_regime_gauge',
    'create_hmm_probability_chart',
    'create_component_radar',
    'create_heatmap',
    'create_distribution_chart',
    'create_regime_distribution_chart',
    'create_time_series_chart',
    'create_ranking_chart'
]
