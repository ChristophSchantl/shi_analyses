
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

try:
    import empyrical as ep
except ImportError:
    st.warning("Bitte installiere empyrical via `pip install empyrical` für alle Metriken!")

warnings.simplefilter("ignore", FutureWarning)
sns.set_theme(style="darkgrid")
plt.style.use("seaborn-v0_8-darkgrid")

# Konfiguration
RISK_FREE_RATE = 0.02
CSV_PATHS = {
    'SHI Income': 'SHI_INCOME_28Mai2025.csv',
    'SHI Alpha': 'SHI_ALPHA_28Mai2025.csv'
}
YAHOO_TICKERS = {
    'BW-Bank Potenzial T1': '0P0000J5K3.F',
    'BW-Bank Potenzial T2': '0P0000J5K8.F',
    'BW-Bank Potenzial T4': '0P0000JM36.F',
    'BW-Bank Aktienallokation 75 P Dis': '0P0001HPL2.F'
}

def load_returns_from_csv(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    close = pd.to_numeric(df['Close'], errors='coerce').ffill().dropna()
    returns = close.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    return returns, cumulative

def load_returns_from_yahoo(ticker):
    df = yf.download(ticker, start="2024-01-01", progress=False)['Close'].dropna()
    returns = df.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    return returns, cumulative

@st.cache_data
def load_and_sync_data():
    returns_dict, cumulative_dict = {}, {}
    for name, path in CSV_PATHS.items():
        ret, cum = load_returns_from_csv(path)
        returns_dict[name] = ret
        cumulative_dict[name] = cum
    for name, ticker in YAHOO_TICKERS.items():
        ret, cum = load_returns_from_yahoo(ticker)
        returns_dict[name] = ret
        cumulative_dict[name] = cum
    common_index = sorted(set.intersection(*(set(r.index) for r in returns_dict.values())))
    for name in returns_dict:
        returns_dict[name] = returns_dict[name].loc[common_index]
        cumulative_dict[name] = cumulative_dict[name].loc[common_index]
    return returns_dict, cumulative_dict

def calculate_metrics(returns_dict, cumulative_dict):
    metrics = pd.DataFrame()
    for name in returns_dict:
        ret = returns_dict[name]
        cum = cumulative_dict[name]
        if isinstance(ret, pd.DataFrame):
            ret = ret.iloc[:, 0]
        ret = pd.to_numeric(ret, errors='coerce').dropna()
        if ret.empty or cum.empty:
            continue
        days = (cum.index[-1] - cum.index[0]).days
        total_ret = float(cum.iloc[-1] / cum.iloc[0] - 1)
        annual_ret = (1 + total_ret)**(365/days) - 1 if days > 0 else np.nan
        annual_vol = ret.std() * np.sqrt(252)
        sharpe = (annual_ret - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else np.nan
        try:
            sortino = ep.sortino_ratio(ret, annualization=252)
            omega = ep.omega_ratio(ret, annualization=252)
            tail = ep.tail_ratio(ret)
        except:
            sortino = omega = tail = np.nan
        drawdowns = (cum / cum.cummax()) - 1
        mdd = float(drawdowns.min())  # sicherstellen, dass es ein Skalar ist
        calmar = annual_ret / abs(mdd) if mdd < 0 else np.nan

        var_95 = ret.quantile(0.05)
        cvar_95 = ret[ret <= var_95].mean()
        win_rate = len(ret[ret > 0]) / len(ret)
        avg_win = ret[ret > 0].mean()
        avg_loss = ret[ret < 0].mean()
        profit_factor = -avg_win / avg_loss if avg_loss < 0 else np.nan
        monthly_ret = ret.resample('M').apply(lambda x: (1 + x).prod() - 1)
        positive_months = (monthly_ret > 0).mean()

        metrics.loc[name, 'Total Return'] = total_ret
        metrics.loc[name, 'Annual Return'] = annual_ret
        metrics.loc[name, 'Annual Volatility'] = annual_vol
        metrics.loc[name, 'Sharpe Ratio'] = sharpe
        metrics.loc[name, 'Sortino Ratio'] = sortino
        metrics.loc[name, 'Max Drawdown'] = mdd
        metrics.loc[name, 'Calmar Ratio'] = calmar
        metrics.loc[name, 'VaR (95%)'] = var_95
        metrics.loc[name, 'CVaR (95%)'] = cvar_95
        metrics.loc[name, 'Omega Ratio'] = omega
        metrics.loc[name, 'Tail Ratio'] = tail
        metrics.loc[name, 'Win Rate'] = win_rate
        metrics.loc[name, 'Avg Win'] = avg_win
        metrics.loc[name, 'Avg Loss'] = avg_loss
        metrics.loc[name, 'Profit Factor'] = profit_factor
        metrics.loc[name, 'Positive Months'] = positive_months
    return metrics

# ---- STREAMLIT UI ----

st.set_page_config(layout="wide", page_title="Strategieanalyse Dashboard")
st.title("📊 Strategie-Analyse & Risiko-Kennzahlen")

returns_dict, cumulative_dict = load_and_sync_data()
metrics = calculate_metrics(returns_dict, cumulative_dict)

tab1, tab2, tab3 = st.tabs(["🔍 Metriken", "📈 Performance", "📉 Drawdown & Korrelationen"])

with tab1:
    st.subheader("Erweiterte Risikokennzahlen")
    st.dataframe(metrics.style.format({
        'Total Return': '{:.2%}',
        'Annual Return': '{:.2%}',
        'Annual Volatility': '{:.2%}',
        'Sharpe Ratio': '{:.2f}',
        'Sortino Ratio': '{:.2f}',
        'Max Drawdown': '{:.2%}',
        'Calmar Ratio': '{:.2f}',
        'VaR (95%)': '{:.2%}',
        'CVaR (95%)': '{:.2%}',
        'Win Rate': '{:.2%}',
        'Avg Win': '{:.2%}',
        'Avg Loss': '{:.2%}',
        'Profit Factor': '{:.2f}',
        'Positive Months': '{:.2%}',
        'Omega Ratio': '{:.2f}',
        'Tail Ratio': '{:.2f}'
    }), use_container_width=True)

with tab2:
    st.subheader("Kumulative Performance")
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, cum in cumulative_dict.items():
        ax.plot(cum.index, cum / cum.iloc[0], label=name)
    ax.set_title("Kumulative Performance (Start = 1.0)")
    ax.legend()
    st.pyplot(fig)

with tab3:
    st.subheader("📉 Drawdown-Verlauf")

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, cum in cumulative_dict.items():
        drawdown = (cum / cum.cummax()) - 1
        # Sicherstellen, dass es eine Series ist
        if isinstance(drawdown, pd.DataFrame):
            drawdown = drawdown.iloc[:, 0]
        ax.plot(drawdown.index, drawdown, label=name, alpha=0.8)
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📊 Korrelation der Tagesrenditen")

    # 💡 Fix: sicherstellen, dass alle returns 1D Series sind
    returns_cleaned = {
        k: (v.iloc[:, 0] if isinstance(v, pd.DataFrame) else v).dropna()
        for k, v in returns_dict.items()
    }

    df_corr = pd.DataFrame(returns_cleaned)
    corr = df_corr.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax_corr)
    ax_corr.set_title("Korrelationsmatrix der täglichen Renditen")
    st.pyplot(fig_corr)
