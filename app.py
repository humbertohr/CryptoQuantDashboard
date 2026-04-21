import streamlit as st
from streamlit_option_menu import option_menu
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.stats as stats
import seaborn as sns
from scipy.optimize import minimize
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.ticker as mticker

# ---------------------------
# PAGE CONFIG (ONLY ONCE)
# ---------------------------
st.set_page_config(
    page_title="Crypto Quant Dashboard",
    layout="wide"
)

# ---------------------------
# GLOBAL STYLE (FINAL UI FIX)
# ---------------------------
st.markdown("""
<style>

/* ===== FONT ===== */
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Roboto', sans-serif !important;
}

/* ===== LAYOUT IMPROVEMENTS ===== */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
}

/* ===== TYPOGRAPHY ===== */
h1 {
    font-size: 42px !important;
    font-weight: 600 !important;
    letter-spacing: -0.5px;
    margin-bottom: 0.5rem !important;
}

h2, h3 {
    font-weight: 500 !important;
    letter-spacing: -0.3px;
    margin-bottom: 0.5rem !important;
}

p, span, label {
    font-weight: 400 !important;
    color: #cfd3dc !important;
}

/* ===== TABLE STYLE ===== */
thead tr th {
    font-weight: 500 !important;
    color: #9aa4b2 !important;
}

div[data-testid="stDataFrame"] {
    font-size: 13px;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 6px;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background-color: #0b0f19;
}

/* ===== FORCE REMOVE RED ===== */
ul[data-testid="stSidebarNav"] li a[aria-selected="true"],
.nav-link.active {
    background-color: #2e7b7b !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

/* Hover */
ul[data-testid="stSidebarNav"] li a:hover,
.nav-link:hover {
    background-color: rgba(46, 123, 123, 0.25) !important;
    color: #ffffff !important;
}

/* Icons */
.nav-link i {
    color: #2e7b7b !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #2e7b7b !important;
    border-bottom: 2px solid #2e7b7b !important;
}

/* Buttons */
button[kind="primary"] {
    background-color: #2e7b7b !important;
    border: none !important;
}

/* Kill any default red globally */
* {
    --primary-color: #2e7b7b !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# TITLE
# ---------------------------
st.title("Crypto Quant Dashboard")

# ---------------------------
# SIDEBAR MENU
# ---------------------------
menu_options = {
    "Market Data": "market",
    "Financial Analysis": "analysis",
    "Risk Management": "risk",
    "Financial Modeling": "modeling",
    "Portfolio": "portfolio"
}

with st.sidebar:
    selected = option_menu(
        "Navigation",
        list(menu_options.keys()),
        icons=["bar-chart", "graph-up", "exclamation-triangle", "cpu", "pie-chart"],
        menu_icon="cast",
        default_index=0,
        
        styles={
            "container": {
                "background-color": "#0b0f19",
                "padding": "10px"
            },
            "icon": {
                "color": "#2e7b7b",
                "font-size": "18px"
            },
            "nav-link": {
                "color": "#cfd3dc",
                "font-size": "14px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#1a2233",
            },
            "nav-link-selected": {
                "background-color": "#2e7b7b",  # ✅ replaces red
                "color": "white",
                "font-weight": "500",
            },
        }
    )

main_topic = menu_options[selected]

# ---------------------------
# DATA (GLOBAL CACHE)
# ---------------------------
tickers = ["BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD"]

@st.cache_data
def load_data():
    data = yf.download(tickers, start="2021-01-01", end="2025-12-31")["Close"]
    log_returns = np.log(data / data.shift(1)).dropna()
    return data, log_returns

data, log_returns = load_data()

# ---------------------------
# METRICS
# ---------------------------
mean_returns = log_returns.mean() * 252
volatility = log_returns.std() * np.sqrt(252)

# Growth of $1
cumulative_returns = (1 + log_returns).cumprod()
rolling_max = cumulative_returns.cummax()
drawdowns = (cumulative_returns - rolling_max) / rolling_max
max_drawdown = drawdowns.min()

# VaR & CVaR
var_5 = log_returns.quantile(0.05)
cvar_5 = log_returns[log_returns <= var_5].mean()

# Sharpe Ratio
sharpe = mean_returns / volatility

summary_table = pd.DataFrame({
    "Return": mean_returns,
    "Volatility": volatility,
    "Sharpe": sharpe,
    "Max Drawdown": max_drawdown,
    "CVaR (5%)": cvar_5
})

# ---------------------------
# DISPLAY
# ---------------------------
st.subheader("Summary Table")
st.dataframe(summary_table.style.format("{:.2%}"), use_container_width=True)

    # ---------------------------
    # MAIN PANEL
    # ---------------------------

# 1. MARKET DATA ---------------------------------------------------------------------------------------------------------------
if main_topic == "market":
    st.header("Market Data")

    tab1, tab2, tab3 = st.tabs([
        "Crypto Price Data & Returns",
        "Returns Distribution",
        "Kurtosis"
    ])

    # ---------------------------
    # TAB 1
    # ---------------------------
    with tab1:
        st.subheader("💹 Crypto Price Data & Returns")

        col1, col2 = st.columns([1.2, 1])  # slightly wider for price table

        with col1:
            st.markdown("### 💰 Price Data (Last Rows)")
            st.dataframe(
                data.tail().style.format("${:,.2f}"),
                use_container_width=True
            )
            st.markdown("""
            Daily closing prices in **USD** for each asset.  
            The table displays the latest observations used for return and risk calculations.
            """)

        with col2:
            st.markdown("### 📉 Log Returns (Last Rows)")
            st.dataframe(
                log_returns.tail().style.format("{:.4f}"),
                use_container_width=True
            )
            st.markdown("""
            Daily **log returns** computed from closing prices.  
            These series form the basis for volatility modeling and risk estimation.
            """)

        # PRICE CHARTS
        st.markdown("### 📈 Crypto Price Evolution")

        # Filter out USD-USDT if present
        assets = [asset for asset in data.columns if asset != "USD-USDT"]

        # Create 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # balanced size for 4 charts
        axes = axes.flatten()

        for i, asset in enumerate(assets):
            ax = axes[i]
            ax.plot(data.index, data[asset], label=asset, color="#4C72B0")  # consistent muted palette

            ax.set_title(f"{asset}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Date", fontsize=9)
            ax.set_ylabel("Price (USD)", fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            ax.legend(fontsize=7, loc="upper left", frameon=False)

            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(lambda x, _: f'${x:,.0f}')
            )

        # Adjust layout
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        Time series of daily closing prices (**USD**) for selected cryptocurrencies.  
        Each subplot shows one asset individually, making it easier to compare long‑term trends, volatility cycles, and market regimes side by side.
        """)


    # ---------------------------
    # TAB 2 (DISTRIBUTIONS)
    # ---------------------------
    with tab2:
        st.subheader("📊 Log Returns Distribution")
        st.markdown("""
        Empirical distribution of daily log returns.  
        Shared axis enables comparison of volatility and tail behavior across assets.  
        The left tail represents downside tail risk (large losses), and the right tail captures extreme positive returns.  
        Differences in tail thickness and asymmetry reveal skewness and excess kurtosis, signaling deviations from normality and the likelihood of extreme events.
        """)

        xmin = log_returns.min().min()
        xmax = log_returns.max().max()

        # Define insights for each coin
        insights = {
            "BNB-USD": [
                "Distribution is centered near zero, suggesting most daily returns are small.",
                "Slight right skew indicates occasional larger positive moves.",
                "Long right tail implies potential for sudden rallies.",
                "Long left tail highlights downside risk from sharp sell-offs."
            ],
            "BTC-USD": [
                "High peak near zero shows frequent small fluctuations.",
                "Left skew suggests more frequent negative shocks than positive jumps.",
                "Extended left tail signals vulnerability to steep crashes.",
                "Right tail, though thinner, shows potential for speculative surges."
            ],
            "ETH-USD": [
                "Returns cluster tightly around zero, reflecting moderate volatility.",
                "Mild right skew points to more upside bursts than downside drops.",
                "Right tail indicates potential for strong bullish runs.",
                "Left tail, though shorter, still warns of sudden downturns."
            ],
            "XRP-USD": [
                "Bell-shaped distribution suggests relatively balanced return behavior.",
                "Slight left skew shows tendency toward negative shocks.",
                "Long left tail highlights risk of sharp declines.",
                "Right tail reveals occasional explosive upward moves."
            ]
        }

        for asset in log_returns.columns:
            # Each row: chart on the left, insights on the right
            col1, col2 = st.columns([1.6, 1])  # slightly wider text column

            with col1:
                # Compact, aesthetic figure size
                fig, ax = plt.subplots(figsize=(5.5, 2.8))

                # Consistent muted color palette
                sns.histplot(log_returns[asset], bins=40, kde=True, ax=ax, color="#4C72B0")

                # Same scale for all
                ax.set_xlim(xmin, xmax)

                # Mean line in consistent accent color
                mean = log_returns[asset].mean()
                ax.axvline(mean, linestyle='--', color="#C44E52")

                # Fixed limits for comparability
                ax.set_xlim(-0.25, 0.25)

                ax.set_title(f"{asset}", fontsize=12, fontweight="bold")
                ax.set_xlabel("Log Return", fontsize=9)
                ax.set_ylabel("Count", fontsize=9)
                ax.tick_params(axis='both', labelsize=8)

                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                st.markdown(f"### {asset} Insights")
                for point in insights[asset]:
                    st.markdown(f"- {point}")

            # Add spacing between rows for clarity
            st.markdown("---")


    # ---------------------------
    # TAB 3
    # ---------------------------
    with tab3:
        st.subheader("Advanced Tail Risk Visualization")

        # General description at the top
        st.markdown("""
        These four histogram visualizations analyze and compare the return distributions of different cryptocurrencies.  
        The code separates **normal events** (returns within the 5th–95th percentile range) from **extreme events** (returns outside that range), 
        allowing us to see both the typical daily behavior and the rare, high‑impact movements.  
        By overlaying a theoretical normal distribution curve, the plots highlight how real data deviates from standard assumptions of normality, 
        revealing skewness, excess kurtosis, and tail risk.

        - **Normal vs Extreme regions:** Most returns cluster near the center, while extreme tails capture rare but impactful shocks.  
        - **VaR 5% line:** Threshold below which only 5% of returns fall, representing downside risk.  
        - **VaR 95% line:** Threshold above which only 5% of returns fall, highlighting rare but significant gains.  
        - **Purpose overall:** Combining histograms, extreme tails, and VaR thresholds provides a clear view of tail risk and how each asset’s empirical distribution diverges from the idealized normal curve.

        """)

        # Pre-compute insights for each asset
        insights = {
            "BNB-USD": [
                "Distribution is centered near zero, showing most daily returns are small.",
                "Slight right skew suggests occasional larger positive moves.",
                "Long right tail indicates potential for sudden rallies.",
                "Left tail highlights downside risk from sharp sell-offs.",
                "BNB’s long‑term rise is driven by infrequent but outsized rallies that offset the more common small losses."
            ],
            "BTC-USD": [
                "High peak near zero shows frequent small fluctuations.",
                "Left skew suggests more frequent negative shocks than positive jumps.",
                "Extended left tail signals vulnerability to steep crashes.",
                "Right tail, though thinner, shows potential for speculative surges.",
                "Despite more frequent downside shocks, Bitcoin’s long‑term growth comes from rare but very large rallies that outweigh those losses."
            ],
            "ETH-USD": [
                "Returns cluster tightly around zero, reflecting moderate volatility.",
                "Mild right skew points to more upside bursts than downside drops.",
                "Right tail indicates potential for strong bullish runs.",
                "Left tail, though shorter, still warns of sudden downturns.",
                "Ethereum’s price trend reflects that occasional large upward bursts have outweighed smaller, more frequent declines."
            ],
            "XRP-USD": [
                "Bell-shaped distribution suggests relatively balanced return behavior.",
                "Slight left skew shows tendency toward negative shocks.",
                "Long left tail highlights risk of sharp declines.",
                "Right tail reveals occasional explosive upward moves.",
                "XRP’s price history shows that while downside shocks occur, rare explosive rallies have lifted its long‑term trajectory."
            ]
        }

        # Loop through each asset and plot individually with insights
        for asset in log_returns.columns:
            col1, col2 = st.columns([1.6, 1])  # slightly wider text column

            with col1:
                data_asset = log_returns[asset]

                # --- Statistics ---
                mu = data_asset.mean()
                sigma = data_asset.std()
                var_5 = data_asset.quantile(0.05)
                var_95 = data_asset.quantile(0.95)

                # --- Split data ---
                normal_data = data_asset[(data_asset >= var_5) & (data_asset <= var_95)]
                extreme_data = data_asset[(data_asset < var_5) | (data_asset > var_95)]

                # --- Plot ---
                fig, ax = plt.subplots(figsize=(5.5, 2.8))  # slightly wider, balanced height

                sns.histplot(normal_data, bins=50, stat="density", label="Normal", ax=ax, color="#4C72B0")
                sns.histplot(extreme_data, bins=50, stat="density", label="Extreme", ax=ax, color="#DD8452")

                # Normal distribution curve
                x = np.linspace(data_asset.min(), data_asset.max(), 1000)
                y = stats.norm.pdf(x, mu, sigma)
                ax.plot(x, y, linestyle='--', label='Normal Dist', color="black")

                # VaR lines
                ax.axvline(var_5, linestyle=':', label='VaR 5%', color="#C44E52")
                ax.axvline(var_95, linestyle=':', label='VaR 95%', color="#55A868")

                ax.set_title(asset, fontsize=12, fontweight="bold")
                ax.set_xlabel("Log Return", fontsize=9)
                ax.set_ylabel("Density", fontsize=9)
                ax.tick_params(axis='both', labelsize=8)
                ax.legend(fontsize=7, loc="upper right", frameon=False)

                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                st.markdown(f"### {asset} Insights")
                for point in insights[asset]:
                    st.markdown(f"- {point}")

            # Add spacing between rows for clarity
            st.markdown("---")

 
# 2. FINANCIAL ANALYSIS ---------------------------------------------------------------------------------------------------------------

elif main_topic == "analysis":
    st.header("Financial Analysis")

    tab1, tab2, tab3 = st.tabs([
        "Summary Statistics",
        "Volatility & Regimes",
        "Risk Metrics & Correlations"
    ])

    # --- TAB 1: Summary Statistics ---
    with tab1:
        st.subheader("📊 Annualized Return & Volatility")

        # Calculate mean and standard deviation of log returns
        mean_returns = log_returns.mean() * 252   # annualized mean
        volatility = log_returns.std() * np.sqrt(252)  # annualized volatility

        # Combine into a summary table
        summary_stats = pd.DataFrame({
            "Annualized Return": mean_returns,
            "Annualized Volatility": volatility
        })

        st.dataframe(summary_stats.style.format({
            "Annualized Return": "{:.2%}",
            "Annualized Volatility": "{:.2%}"
        }), use_container_width=True)

        st.markdown("""
        **Interpretation:**  
        - Annualized return shows the average growth rate if daily log returns persisted for a year.  
        - Annualized volatility measures risk (variability of returns).  
        - Comparing assets side by side highlights which coins deliver higher returns relative to their risk.
        """)

    # --- TAB 2: Volatility & Regimes ---
    with tab2:
        st.subheader("🌪 Rolling Volatility & Risk Regimes")
        
        st.markdown("""
        ### 📌 Interpretations:
        - Short windows (10-day) capture quick spikes in risk.  
        - Longer windows (30–60 day) smooth volatility, showing broader regimes.  
        - Comparing windows helps identify when BTC shifted from calm to turbulent markets.
        """)        

        # Rolling volatility (different windows)
        vol_10 = log_returns.rolling(window=10).std() * np.sqrt(252)
        vol_30 = log_returns.rolling(window=30).std() * np.sqrt(252)
        vol_60 = log_returns.rolling(window=60).std() * np.sqrt(252)

        # Example: BTC rolling volatility comparison
        asset = "BTC-USD"
        fig, ax = plt.subplots(figsize=(8,4))

        # Professional muted colors
        ax.plot(vol_10.index, vol_10[asset], label="10-day", color="#1f77b4", linewidth=1)
        ax.plot(vol_30.index, vol_30[asset], label="30-day", color="#7f7f7f", linewidth=1)
        ax.plot(vol_60.index, vol_60[asset], label="60-day", color="#2ca02c", linewidth=1)

        # Title and labels
        ax.set_title(f"{asset} — Rolling Volatility", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("Volatility", fontsize=9)

        # Gridlines for clarity
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Legend styling
        ax.legend(fontsize=8, frameon=False, loc="upper right")

        # Tick parameters
        ax.tick_params(axis='both', labelsize=8)

        st.pyplot(fig)
        
        st.markdown("""
        ### 🔎 Insights:
        - **Short-term spikes:** The 10-day volatility line shows sharp, frequent peaks, capturing sudden market shocks more quickly than longer windows.  
        - **Smoother trends:** The 30-day and 60-day lines filter out noise, highlighting broader volatility regimes and sustained risk periods.  
        - **Risk cycles:** Noticeable clusters of elevated volatility suggest recurring phases of instability, useful for identifying high-risk market environments.  
        """)

        st.markdown("---")
        
        st.markdown("""
        ### 📌 Interpretations:
        - Visualizes Bitcoin’s price history alongside periods classified as **high risk**, making volatility regimes more tangible.  
        - Helps investors connect **price movements** with **risk environments**, rather than viewing them in isolation.  
        - Provides a framework to assess whether **risk-adjusted strategies** (avoiding high-risk phases) could improve outcomes. 
        """)      
        
        # Risk regime example
        high_risk = vol_10 > vol_30
        regime = high_risk.astype(int)
        aligned_price = data.loc[regime.index, asset]

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(regime.index, aligned_price, label="Price", color="#4C72B0")
        ax.fill_between(
            regime.index,
            aligned_price.min(),
            aligned_price.max(),
            where=regime[asset] == 1,
            alpha=0.2,
            color="#C44E52",
            label="High Risk"
        )
        ax.set_title(f"{asset} — Price with High Risk Regimes", fontsize=11, fontweight="bold")
        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("Price (USD)", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        st.pyplot(fig)
        
        st.markdown("""
        ### 🔎 Insights:
        - High risk periods often coincide with **sharp declines or unstable sideways movements**, highlighting vulnerability.  
        - Some rallies occur **outside high-risk zones**, suggesting calmer regimes can still deliver strong gains.  
        - Extended red bands show **clusters of instability**, indicating that risk is not random but tends to persist in cycles.  
        """)        

    # --- TAB 3: Risk Metrics & Correlations ---
    with tab3:
        st.subheader("⚖️ Risk Metrics & Correlation Matrix")

        st.dataframe(summary_table.style.format({
            "Return": "{:.2%}",
            "Volatility": "{:.2%}",
            "Sharpe": "{:.2f}",
            "Max Drawdown": "{:.2%}",
            "CVaR (5%)": "{:.2%}"
        }), use_container_width=True)

        st.markdown("""
        - **BNB‑USD**
          - Strongest annualized return (43.14%), showing robust long‑term growth potential.  
          - Volatility is high (67.60%), but lower than ETH and XRP, suggesting relatively more stability.  
          - Sharpe ratio (0.64) is the best among peers, meaning BNB delivers the most efficient risk‑adjusted performance.  
          - Max drawdown (‑86.59%) highlights vulnerability to deep crashes despite strong returns.  
          - CVaR (‑9.39%) indicates moderate tail risk compared to ETH and XRP, making downside shocks less severe.  

        - **BTC‑USD**
          - Annualized return (15.23%) is the lowest, reflecting slower growth relative to other cryptos.  
          - Volatility (48.56%) is the lowest, showing BTC is the most stable asset in this group.  
          - Sharpe ratio (0.31) is modest, meaning BTC’s risk‑adjusted performance lags behind BNB.  
          - Max drawdown (‑83.72%) confirms BTC is not immune to severe crashes.  
          - CVaR (‑7.28%) is the smallest, showing BTC has the mildest extreme downside risk among peers.  

        - **ETH‑USD**
          - Annualized return (19.39%) is moderate, higher than BTC but lower than BNB and XRP.  
          - Volatility (65.48%) is high, reflecting Ethereum’s exposure to sharp swings.  
          - Sharpe ratio (0.30) is weak, showing limited efficiency in risk‑adjusted returns.  
          - Max drawdown (‑88.92%) is among the deepest, highlighting vulnerability in downturns.  
          - CVaR (‑9.71%) shows significant tail risk, meaning extreme losses are more severe than BTC or BNB.  

        - **XRP‑USD**
          - Annualized return (28.55%) is strong, second only to BNB.  
          - Volatility (81.92%) is the highest, making XRP the most unstable asset in this set.  
          - Sharpe ratio (0.35) is slightly better than BTC and ETH, but still far below BNB.  
          - Max drawdown (‑94.15%) is the worst, showing XRP suffered the deepest historical crash.  
          - CVaR (‑11.14%) is the largest, meaning XRP carries the highest tail risk and most severe downside shocks.  
        """)

        
        st.markdown("---")

        # Correlation matrix
        corr_matrix = log_returns.corr()
        fig, ax = plt.subplots(figsize=(7,5))

        # Professional color palette: blue-white-red but muted
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",       # balanced red/blue diverging palette
            center=0,
            linewidths=0.5,      # thin lines between cells
            linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
            annot_kws={"size":8, "color":"black"}  # smaller, clean annotations
        )

        # Title styling
        ax.set_title("Correlation Matrix of Crypto Returns", fontsize=12, fontweight="bold", pad=12)

        # Adjust ticks
        ax.tick_params(axis='both', labelsize=8)

        st.pyplot(fig)

        st.markdown("""
        ### 🔎 Insights from the Correlation Matrix
        - **BTC & ETH (0.81):** This is the strongest correlation, showing that Bitcoin and Ethereum often move together, reducing diversification benefits between them.  
        - **BNB & BTC (0.66):** Moderate correlation suggests BNB tends to follow Bitcoin’s trends but with some independence, offering partial diversification.  
        - **BNB & ETH (0.67):** Similar to BTC, Ethereum and BNB are moderately correlated, meaning they share market drivers but not perfectly.  
        - **XRP correlations (0.53–0.61):** XRP shows weaker correlations with the others, especially with BNB (0.53), making it the most distinct asset in this set.  
        - **Overall takeaway:** All assets are positively correlated, but varying strengths imply limited diversification within crypto; XRP provides the most differentiation, while BTC and ETH behave most alike.  
        """)
 
        st.markdown("""
        ### 📘 Leverage vs. Pair Trading
        - **Leverage**: Borrowed capital used to magnify position size, amplifying both gains and losses.  
        - **Pair Trading**: Strategy of going long one asset and short another to exploit correlation or relative performance differences.  
        - **Combined Use**: Traders often apply leverage to pair trades, but leverage itself is not defined by the act of buying one crypto and selling another.  
        """)

 
# 3. RISK MANAGEMENT ----------------------------------------------------------------------------------------------------------
elif main_topic == "risk":
    st.header("Risk Management")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Value at Risk (VaR)",
        "Score-Based Allocation",
        "Stress Testing",
        "Markowitz Optimization",
        "Liquidity Risk"
    ])

    # --- TAB 1: Value at Risk ---
    with tab1:
        st.subheader("📉 Value at Risk (VaR)")
        st.markdown("""
        Value at Risk (VaR) estimates potential **worst-case daily losses** under normal conditions.  
        - **95% VaR:** On 95% of days, losses will not exceed this level.  
        - **99% VaR:** On 99% of days, losses will not exceed this level.  
        """)

        VaR_95 = log_returns[["BTC-USD","ETH-USD","BNB-USD","XRP-USD"]].quantile(0.05)
        VaR_99 = log_returns[["BTC-USD","ETH-USD","BNB-USD","XRP-USD"]].quantile(0.01)

        st.write("**95% VaR (1-day):**")
        st.dataframe(VaR_95.to_frame("VaR 95%").style.format("{:.2%}"))
        st.write("**99% VaR (1-day):**")
        st.dataframe(VaR_99.to_frame("VaR 99%").style.format("{:.2%}"))

        st.markdown("---")
        
# --- TAB 2: Score-Based Allocation ---
    with tab2:
        st.markdown("""
        ### ⚙️ Score-Based Allocation

        This method ranks assets by **Sharpe ratio**, penalized for **volatility** and **tail risk (CVaR, drawdown)**.  
        Weights are normalized scores, producing a portfolio that emphasizes **stability** and penalizes **extreme downside risk**.
        """)

        st.markdown("""
        ### 📘 Conclusion

        1. **Score-based allocation works best in highly volatile markets** such as cryptocurrencies (BTC, BNB, XRP).  
           - These assets often experience extreme drawdowns and fat-tailed distributions.  
           - By penalizing volatility and downside risk (CVaR, drawdown), this method produces allocations that are more robust to sudden shocks.
        """)

        # Copy summary_table from Risk Metrics tab
        df = summary_table.copy()
        df = df.drop("USDT-USD", errors="ignore")

        # Add absolute risk measures
        df["Abs Drawdown"] = df["Max Drawdown"].abs()
        df["Abs CVaR"] = df["CVaR (5%)"].abs()

        # Score formula
        df["Score"] = df["Sharpe"] / (df["Volatility"] * df["Abs CVaR"])

        # Normalize into weights
        df["Weight"] = df["Score"] / df["Score"].sum()

        # Define consistent colors once
        colors = {
            "BNB-USD": "#1f77b4",  # blue
            "BTC-USD": "#2ca02c",  # green
            "ETH-USD": "#c44e52",  # red
            "XRP-USD": "#7f7f7f"   # gray
        }

        # --- Layout: table left, pie chart right ---
        col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Normalized Portfolio Weights**")
        weights_df = pd.DataFrame(df["Weight"]).rename(columns={"Weight":"Allocation"})
        st.dataframe(weights_df.style.format("{:.2%}"), use_container_width=True)

        # Place Key Takeaways directly under the table
        st.markdown("""
        ### 📘 Key Takeaways
        - Assets with strong **risk-adjusted performance** get higher weights.  
        - **Downside risk (CVaR, drawdown)** directly reduces allocations.  
        - Provides a more **intuitive and practical allocation** method for volatile markets like crypto.  
        - Complements **Markowitz optimization** by showing how heuristic scoring leads to different portfolio mixes.  
        """)

    with col2:
        # Filter out near-zero weights
        plot_df = df[df["Weight"] > 1e-4]

        fig, ax = plt.subplots(figsize=(4.5, 4.5))

        # Define consistent colors
        colors_map = {
            "BNB-USD": "#1f77b4",  # blue
            "BTC-USD": "#2ca02c",  # green
            "ETH-USD": "#c44e52",  # red
            "XRP-USD": "#7f7f7f"   # gray
        }

        plot_colors = [colors_map[a] for a in plot_df.index]

        # Custom percentage formatter
        def autopct_format(pct):
            return f"{pct:.1f}%" if pct > 1 else ""

        ax.pie(
            plot_df["Weight"],
            labels=plot_df.index,
            autopct=autopct_format,
            startangle=90,
            colors=plot_colors
        )

        ax.set_title(
            "Portfolio Allocation (Score-Based)",
            fontsize=12,
            fontweight="bold"
        )

        st.pyplot(fig)

        
    # --- TAB 3: Stress Testing ---
    with tab3:
        st.subheader("💥 Stress Testing Scenarios")
        st.markdown("""
        Stress testing asks: *What happens if BTC and ETH suddenly crash by 20% in one day?*  
        Equal weighting is used to remove bias, so differences are due to the shock itself.  
        """)

        stress_scenario = log_returns[["BTC-USD","ETH-USD","BNB-USD","XRP-USD"]].copy()
        stress_scenario["BTC-USD"] -= 0.20
        stress_scenario["ETH-USD"] -= 0.20

        weights = np.array([0.25,0.25,0.25,0.25])  # 4 assets, equal weights
        portfolio_returns = (log_returns[["BTC-USD","ETH-USD","BNB-USD","XRP-USD"]] * weights).sum(axis=1)
        portfolio_stress = (stress_scenario * weights).sum(axis=1)

        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(portfolio_returns, bins=50, alpha=0.6, label="Normal", color="#1f77b4")
        ax.hist(portfolio_stress, bins=50, alpha=0.6, label="Stress -20% BTC/ETH", color="#c44e52")
        ax.set_title("Portfolio Return Distribution: Normal vs Stress",     size=12, fontweight="bold")
        ax.set_xlabel("Return", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        st.pyplot(fig)

        st.markdown("---")

    # --- TAB 4: Markowitz Optimization ---
    with tab4:
        st.markdown("""
        ### ⚙️ Markowitz Portfolio Optimization

        Mean–variance optimization (Harry Markowitz) balances **expected return vs risk**.  
        The goal is to find optimal weights that maximize return for a given risk aversion.  
        """)

        st.markdown("""
        ### 📘 Conclusion

        2. **Markowitz optimization works best in less volatile, traditional markets** such as equities (Apple, Tesla, Google).  
           - In these markets, returns are closer to normal distributions and variance is a reliable measure of risk.  
           - The model captures diversification benefits through correlations, producing mathematically efficient portfolios.  
        """)

        # Assets and data
        assets = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD"]
        data = log_returns[assets]
        mu = data.mean().values
        Sigma = data.cov().values
        n = len(mu)
        lmbda = 10  # risk aversion parameter

        # Objective function
        def objective(w):
            port_return = w @ mu
            port_risk = w.T @ Sigma @ w
            return -(port_return - lmbda * port_risk)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0,1)] * n
        w0 = np.ones(n)/n

        result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights_opt = result.x

        # Build dataframe
        opt_df = pd.DataFrame({"Weight": weights_opt}, index=assets)

        # Define consistent colors
        colors_map = {
            "BNB-USD": "#1f77b4",  # blue
            "BTC-USD": "#2ca02c",  # green
            "ETH-USD": "#c44e52",  # red
            "XRP-USD": "#7f7f7f"   # gray
        }

        # --- Layout: table left, pie chart right ---
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Optimal Portfolio Weights**")
            st.dataframe(opt_df.style.format({"Weight":"{:.2%}"}), use_container_width=True)

            # Place Key Takeaways directly under the table
            st.markdown("""
            ---
            ### 📘 Key Takeaways
            - Weights are chosen to **maximize expected return** while controlling risk.  
            - Diversification is explicitly modeled through the **covariance matrix**.  
            - Assets with unfavorable risk–return tradeoffs may be excluded entirely.  
            - Produces the **efficient frontier** portfolios under chosen risk aversion.  
            """)

        with col2:
            # Filter out near-zero weights only for the chart
            plot_df = opt_df[opt_df["Weight"] > 1e-4]

            fig, ax = plt.subplots(figsize=(4.5, 4.5))

            # Custom percentage formatter (hide labels for very small slices)
            def autopct_format(pct):
                return f"{pct:.1f}%" if pct > 1 else ""

            plot_colors = [colors_map[a] for a in plot_df.index]

            ax.pie(
                plot_df["Weight"],
                labels=plot_df.index,
                autopct=autopct_format,
                startangle=90,
                colors=plot_colors
            )

            ax.set_title(
                "Portfolio Allocation (Markowitz)",
                fontsize=12,
                fontweight="bold"
            )

            st.pyplot(fig)


    # --- TAB 5: Liquidity Risk ---
    with tab5:
        st.subheader("💧 Liquidity Risk Proxy")
        st.markdown("""
        Liquidity risk is proxied by **trading volume**.  
        - High volume = easier to buy/sell without moving price.  
        - Low volume = harder to execute trades, higher slippage.  
        """)

        # ---------------------------
        # LIQUIDITY DATA
        # ---------------------------
        assets = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD"]

        @st.cache_data
        def load_volume():
            return yf.download(assets, start="2021-01-01", end="2025-12-31")["Volume"]

        volume_data = load_volume()

        # ---------------------------
        # PLOT
        # ---------------------------
        fig, ax = plt.subplots(figsize=(10,5))

        for asset in volume_data.columns:
            ax.plot(volume_data.index, volume_data[asset], label=asset, linewidth=1.2)

        # Log scale (KEY improvement)
        ax.set_yscale("log")

        # Titles and labels
        ax.set_title("Liquidity Proxy: Trading Volume (Log Scale)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("Volume (log scale)", fontsize=9)

        # Clean legend
        ax.legend(fontsize=8, frameon=False, loc="upper left")

        # Grid styling
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Render
        st.pyplot(fig)

        st.markdown("---")

        st.subheader("⚖️ Liquidity-Adjusted Optimization")
        st.markdown("""
        Liquidity-adjusted optimization adds a **penalty for trading frictions**, ensuring portfolios are not only risk-efficient but also executable in real markets.  
        """)

        assets = ["BTC-USD","ETH-USD","BNB-USD","XRP-USD"]
        data_opt = log_returns[assets]

        mu = data_opt.mean().values
        Sigma = data_opt.cov().values

        # 🔥 Adjusted liquidity penalties (more realistic scaling)
        gamma = np.array([0.0005, 0.0007, 0.001, 0.0015])
        
        lmbda = 5      # lower risk aversion (was too high)
        kappa = 2      # lower liquidity penalty (was too aggressive)
        n = len(mu)

        def objective(w):
            ret = w @ mu
            risk = w.T @ Sigma @ w
            liquidity_cost = np.sum(gamma * np.abs(w))
            return -(ret - lmbda*risk - kappa*liquidity_cost)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0,1)] * n
        w0 = np.ones(n)/n

        result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights_liq = result.x

        liq_df = pd.DataFrame({"Weight": weights_liq}, index=assets)

        # ---------------------------
        # LAYOUT: TABLE + PIE
        # ---------------------------
        col1, col2 = st.columns(2)

        # --- LEFT: TABLE ---
        with col1:
            st.markdown("**Liquidity-Adjusted Portfolio Weights**")
            st.dataframe(liq_df.style.format("{:.2%}"), use_container_width=True)

            st.markdown("""
            ### 📘 Key Takeaways
            - **Liquidity-adjusted optimization strongly favors highly tradable assets**, leading to a dominant allocation in BTC.  
            - **BNB retains a smaller allocation** as a balance between return potential and acceptable liquidity costs.  
            - Assets like ETH and XRP may be excluded despite high volume if their **risk-adjusted performance is less favorable**.  
            - The model integrates **return, volatility, and execution cost**, producing portfolios that are more realistic and implementable.
            """)

        # --- RIGHT: PIE CHART ---
        with col2:
            # Filter very small weights for cleaner chart
            plot_df = liq_df[liq_df["Weight"] > 1e-3]

            fig, ax = plt.subplots(figsize=(4.5, 4.5))

            colors_map = {
                "BTC-USD": "#2ca02c",
                "ETH-USD": "#c44e52",
                "BNB-USD": "#1f77b4",
                "XRP-USD": "#7f7f7f"
            }

            plot_colors = [colors_map[a] for a in plot_df.index]

            def autopct_format(pct):
                return f"{pct:.1f}%" if pct > 1 else ""

            ax.pie(
                plot_df["Weight"],
                labels=plot_df.index,
                autopct=autopct_format,
                startangle=90,
                colors=plot_colors
            )

            ax.set_title(
                "Liquidity-Adjusted Allocation",
                fontsize=12,
                fontweight="bold"
            )

            st.pyplot(fig)

        st.markdown("---")
 
# 4. FINANCIAL MODELING ------------------------------------------------------------------------------------------------------
elif main_topic == "modeling":
    st.header("Financial Modeling")
    
    # Equal weights across all assets
    weights = np.ones(len(log_returns.columns)) / len(log_returns.columns)

    # Normal portfolio returns
    portfolio_returns = (log_returns * weights).sum(axis=1)

    # Scenario: BTC +10%, ETH -15%
    scenario_returns = log_returns.copy()
    scenario_returns["BTC-USD"] = scenario_returns["BTC-USD"] + 0.10
    scenario_returns["ETH-USD"] = scenario_returns["ETH-USD"] - 0.15

    # Scenario portfolio returns
    portfolio_scenario = (scenario_returns * weights).sum(axis=1)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([ 
        "ARIMA (All Assets)", 
        "ARIMA + GARCH", 
        "GARCH Volatility", 
        "Exponential Smoothing (ETH)", 
        "Scenario Analysis", 
        "Monte Carlo Simulation",
        "Monte Carlo Simulated Equity Curves"
    ])

    # --- TAB 1: ARIMA Forecast All Assets ---
    with tab1:
        st.subheader("ARIMA Forecasts for All Assets")
        
        st.markdown("ARIMA forecasts collapse to a flat mean in crypto returns, highlighting the unpredictability of these markets. This demonstrates why more advanced models (GARCH, Monte Carlo) are needed to capture volatility and risk.")

        assets = log_returns.columns
        steps = 30
        fig, axes = plt.subplots(len(assets), 1, figsize=(12, 4 * len(assets)))

        if len(assets) == 1:
            axes = [axes]

        for i, asset in enumerate(assets):
            series = log_returns[asset].dropna()
            model = ARIMA(series, order=(1,0,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)
            forecast_index = pd.date_range(series.index[-1], periods=steps, freq="D")

            axes[i].plot(series.index[-100:], series[-100:], label="Historical")
            axes[i].plot(forecast_index, forecast, linestyle="--", label="Forecast")
            axes[i].set_title(f"ARIMA(1,0,1) Forecast - {asset}")
            axes[i].legend()

        plt.tight_layout()
        st.pyplot(fig)

    # --- TAB 2: ARIMA + GARCH ---
    with tab2:
        st.subheader("ARIMA + GARCH Forecasts")
        
        st.markdown("""

        The ARIMA + GARCH framework combines two complementary models. ARIMA captures short‑term dynamics in returns, which in crypto tend to flatten toward the average. GARCH models volatility clustering, a strong feature of crypto markets. By overlaying ARIMA forecasts with GARCH volatility bands, we see not only the expected mean path but also the range of uncertainty around it. This highlights that while mean forecasts collapse toward zero, volatility forecasts remain informative, showing periods of heightened or reduced risk.
        """)

        assets = log_returns.columns
        horizon = 30
        fig, axes = plt.subplots(len(assets), 1, figsize=(12, 4 * len(assets)))

        if len(assets) == 1:
            axes = [axes]

        for i, asset in enumerate(assets):
            series = log_returns[asset].dropna()

            arima = ARIMA(series, order=(1,0,1))
            arima_fit = arima.fit()
            mean_forecast = arima_fit.forecast(steps=horizon)

            garch = arch_model(series * 100, vol="Garch", p=1, q=1, mean="Zero")
            garch_fit = garch.fit(disp="off")
            garch_forecast = garch_fit.forecast(horizon=horizon)
            volatility_forecast = np.sqrt(garch_forecast.variance.values[-1, :]) / 100

            future_index = pd.date_range(series.index[-1], periods=horizon, freq="D")

            axes[i].plot(series.index[-100:], series[-100:], label="Historical Returns")
            axes[i].plot(future_index, mean_forecast, linestyle="--", label="ARIMA Mean Forecast")
            axes[i].fill_between(future_index,
                                 mean_forecast - volatility_forecast,
                                 mean_forecast + volatility_forecast,
                                 alpha=0.3,
                                 label="GARCH Volatility Band")
            axes[i].set_title(f"ARIMA + GARCH Forecast - {asset}")
            axes[i].legend()

        plt.tight_layout()
        st.pyplot(fig)

    # --- TAB 3: GARCH Volatility ---
    with tab3:
        st.subheader("GARCH Volatility Forecasts")
        
        st.markdown("""

        GARCH models are designed to capture **volatility clustering**, a key feature of financial time series where periods of high volatility tend to follow each other. In crypto markets, this behavior is especially pronounced. By fitting a GARCH(1,1) model, we can forecast future volatility and compare it with realized volatility from rolling windows. The forecasts highlight how risk evolves over time, showing when markets are expected to remain calm or enter more turbulent phases. This makes GARCH a more informative tool than ARIMA alone, since it focuses on the **risk dynamics** rather than the mean return.
        """)


        assets = log_returns.columns
        horizon = 30
        fig, axes = plt.subplots(len(assets), 1, figsize=(12, 4 * len(assets)))

        if len(assets) == 1:
            axes = [axes]

        for i, asset in enumerate(assets):
            series = log_returns[asset].dropna()
            model = arch_model(series * 100, vol="Garch", p=1, q=1, mean="Zero")
            model_fit = model.fit(disp="off")
            forecast = model_fit.forecast(horizon=horizon)
            volatility = np.sqrt(forecast.variance.values[-1, :]) / 100

            rolling_vol = series.rolling(window=20).std()
            future_index = pd.date_range(series.index[-1], periods=horizon, freq="D")

            axes[i].plot(series.index[-100:], rolling_vol[-100:], label="Realized Volatility (20D)")
            axes[i].plot(future_index, volatility, linestyle="--", label="Forecast Volatility (GARCH)")
            axes[i].set_title(f"GARCH(1,1) Volatility Forecast - {asset}")
            axes[i].legend()

        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        ### 📘 Insights from GARCH Volatility Forecasts

        The charts show how realized volatility compares with forecasted volatility for each cryptocurrency:

        - **Volatility clustering is evident**: periods of high volatility tend to persist, which GARCH captures well.  
        - **BTC and BNB show more stable patterns**, with forecasted volatility bands that taper smoothly into the future.  
        - **ETH and XRP exhibit sharper spikes in realized volatility**, but the forecasts converge toward calmer levels, reflecting mean reversion in risk.  
        - Overall, GARCH provides a forward-looking view of **risk dynamics**, helping identify when markets may remain turbulent versus when volatility is expected to subside.  

        This highlights that while returns are hard to predict, **volatility itself is more structured and can be modeled**, making GARCH a valuable tool for risk management in crypto portfolios.
        """)


    # --- TAB 4: Exponential Smoothing ---
    with tab4:
        st.subheader("Exponential Smoothing Forecasts for All Assets")
        
        st.markdown("""

        Exponential smoothing extends recent price trends into the future by giving more weight to the latest data. It smooths out short‑term noise, offering a simple baseline forecast of crypto prices.  
        While less advanced than ARIMA or GARCH, it’s useful for showing how recent momentum may continue in the near term. Given current history, forecasts suggest prices will stay close to recent levels without a strong upward or downward drift.
        Exponential smoothing extends recent price trends into the future by giving more weight to the latest data. It smooths out short‑term noise, offering a simple baseline forecast of crypto prices.  
        While less advanced than ARIMA or GARCH, it’s useful for showing how recent momentum may continue in the near term. Given current history, forecasts suggest prices will stay close to recent levels without a strong upward or downward drift.
        """)


        assets = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD"]
        horizon = 30

        fig, axes = plt.subplots(len(assets), 1, figsize=(12, 4 * len(assets)))

        if len(assets) == 1:
            axes = [axes]



        for i, asset in enumerate(assets):
            series = data[asset].dropna()

            # Fit exponential smoothing model
            model_es = ExponentialSmoothing(series, trend="add", seasonal=None)
            model_es_fit = model_es.fit()
            forecast_es = model_es_fit.forecast(horizon)

            future_index = pd.date_range(series.index[-1], periods=horizon, freq="D")

            # Plot historical prices
            axes[i].plot(series.index[-200:], series[-200:], label=f"Historical {asset} Prices")

            # Plot forecast
            axes[i].plot(future_index, forecast_es, linestyle="--", label="Exponential Smoothing Forecast")

            axes[i].set_title(f"Exponential Smoothing Forecast - {asset}")
            axes[i].legend()

            # Format y-axis with dollar sign and commas
            axes[i].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        plt.tight_layout()
        st.pyplot(fig)

        # Add interpretation text
        st.markdown("""
        ### 📘 Interpretation

        Exponential smoothing provides a simple way to forecast future prices by extending recent trends.  
        - For assets like **BTC and ETH**, the model highlights ongoing upward or downward drifts.  
        - For more volatile assets like **BNB and XRP**, forecasts smooth out noise, showing the expected trajectory without overreacting to short-term swings.  
        - This method is less sophisticated than ARIMA or GARCH, but it offers an intuitive baseline for price forecasting, especially when the goal is to capture **trend continuation** rather than volatility dynamics.
        """)


    # --- TAB 5: Scenario Analysis ---
    with tab5:
        st.subheader("Scenario Analysis: BTC +10%, ETH -15%")
        
        st.markdown("""

        Scenario analysis is used to test how a portfolio might behave under specific market shocks.  
        In this case, we assume **Bitcoin rises by 10%** while **Ethereum falls by 15%**, with other assets unchanged.  
        By comparing the distribution of portfolio returns under normal conditions versus this scenario, we can see how sensitive the portfolio is to large moves in its major components.  

        This approach helps investors understand **risk exposure** and prepare for potential outcomes that standard forecasts may not capture.  
        It highlights the importance of diversification and shows how extreme events in key assets can shift overall portfolio performance.  
        Only two histograms are shown here: one for the normal portfolio and one for the scenario portfolio, because we are testing a single shock.  
        Additional scenarios could be added to explore different market conditions and compare how each one reshapes the portfolio’s risk profile.
        """)


        scenario_returns = log_returns.copy()
        scenario_returns["BTC-USD"] = scenario_returns["BTC-USD"] + 0.10
        scenario_returns["ETH-USD"] = scenario_returns["ETH-USD"] - 0.15

        portfolio_scenario = (scenario_returns * weights).sum(axis=1)

        fig, ax = plt.subplots(figsize=(10,5))
        ax.hist(portfolio_returns, bins=50, alpha=0.6, label="Normal Portfolio")
        ax.hist(portfolio_scenario, bins=50, alpha=0.6, label="Scenario Portfolio")
        ax.set_title("Scenario Analysis: BTC +10%, ETH -15%")
        ax.legend()
        st.pyplot(fig)

    # --- TAB 6: Monte Carlo Simulation ---
    with tab6:
        st.subheader("Monte Carlo Simulation of Portfolio Returns")
        
        st.markdown("""
        Monte Carlo simulation is used to estimate the range of possible portfolio outcomes by repeatedly sampling from the distribution of returns.  
        Instead of relying on a single forecast, it generates thousands of scenarios to capture uncertainty and risk.  
        This approach helps investors understand the **probability of gains or losses**, visualize the distribution of outcomes, and make more informed decisions under uncertainty.
        """)
        n_simulations = 10000
        simulated_returns = np.random.normal(portfolio_returns.mean(),
                                             portfolio_returns.std(),
                                             n_simulations)

        fig, ax = plt.subplots(figsize=(10,5))
        ax.hist(simulated_returns, bins=100, alpha=0.6, color="skyblue", edgecolor="white", density=True)

        # Overlay density curve
        from scipy.stats import norm
        x = np.linspace(simulated_returns.min(), simulated_returns.max(), 500)
        ax.plot(x, norm.pdf(x, simulated_returns.mean(), simulated_returns.std()), "r--", lw=2, label="Normal PDF")

        # Add vertical lines for mean and percentiles
        mean = simulated_returns.mean()
        p5, p95 = np.percentile(simulated_returns, [5, 95])
        ax.axvline(mean, color="black", linestyle="--", lw=1.5, label=f"Mean: {mean:.2%}")
        ax.axvline(p5, color="green", linestyle="--", lw=1, label=f"5th pct: {p5:.2%}")
        ax.axvline(p95, color="orange", linestyle="--", lw=1, label=f"95th pct: {p95:.2%}")

        ax.set_title("Monte Carlo Simulation of Portfolio Returns")
        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
        ax.legend()

        # Format x-axis as percentages
        import matplotlib.ticker as mticker
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        st.pyplot(fig)
        
        st.markdown("""
        ### 📘 Key Takeaways from Monte Carlo Simulation

        - **Mean return is close to zero** (around +0.14%), suggesting the portfolio is expected to hover near break‑even in the average case.  
        - **Downside risk is visible**: the 5th percentile at –5.67% shows that in 1 out of 20 simulations, the portfolio could lose more than 5% in a period.  
        - **Upside potential is limited**: the 95th percentile at +6.00% indicates that in 1 out of 20 simulations, gains above 6% are possible, but not much higher.  
        - **Distribution is fairly symmetric**: the histogram and fitted normal curve align closely, meaning returns behave much like a normal distribution — no extreme skew or fat tails in this simple model.
        """)
        
# --- TAB 7: Monte Carlo Equity Curve ---
    with tab7:
        st.subheader("Monte Carlo Simulated Equity Curves")

        st.markdown("""
        The equity curve shows how portfolio value might evolve over time under repeated random shocks.  
        Each line represents one simulated path of cumulative returns, while the median path highlights the central tendency.  
        This complements the histogram view by illustrating not just single-period outcomes, but the trajectory of portfolio growth or decline.
        """)

        # Parameters
        n_paths = 1000          # number of simulated paths
        n_periods = 252         # trading days in a year
        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std()

        # Simulate paths
        paths = np.zeros((n_paths, n_periods))
        for i in range(n_paths):
            random_returns = np.random.normal(mu, sigma, n_periods)
            paths[i] = np.cumprod(1 + random_returns)  # cumulative equity curve

        # Plot
        fig, ax = plt.subplots(figsize=(10,5))
        for i in range(50):  # plot a subset for readability
            ax.plot(paths[i], color="skyblue", alpha=0.3)

        # Median path
        median_path = np.median(paths, axis=0)
        ax.plot(median_path, color="red", lw=2, label="Median Path")

        ax.set_title("Monte Carlo Simulated Equity Curves")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Portfolio Value (normalized)")
        ax.legend()

        st.pyplot(fig)

# 5. PORTFOLIO ANALYSIS ---------------------------------------------------------------------------------------------
elif main_topic == "portfolio":
    st.header("Portfolio Analysis")

    tab1, tab2, tab3 = st.tabs(["Portfolio vs Assets", "Risk Contribution", "Drawdowns"])

    # --- TAB 1: Portfolio vs Individual Assets ---
    with tab1:
        st.subheader("Portfolio vs Individual Crypto Assets")
        
        st.markdown("""

        This chart compares the performance of an equal‑weight portfolio against each individual cryptocurrency.  
        By allocating the same weight to every asset, the portfolio smooths out extreme movements and highlights the benefits of diversification.  
        The black line shows the cumulative growth of the portfolio, while the colored lines represent the growth of each asset on its own.  
        This view helps investors see whether combining assets reduces volatility and improves stability compared to holding a single coin.
        """)


        # Exclude USD-USDT from analysis
        assets = [col for col in log_returns.columns if col != "USD-USDT"]
        log_returns_filtered = log_returns[assets]
        cumulative_returns_filtered = cumulative_returns[assets]

        # Equal-weight portfolio (dynamic length)
        n_assets = len(log_returns_filtered.columns)
        weights = np.repeat(1/n_assets, n_assets)

        # Calculate portfolio daily returns
        portfolio_returns = (log_returns_filtered * weights).sum(axis=1)

        # Calculate cumulative portfolio growth
        portfolio_cumulative = (1 + portfolio_returns).cumprod()

        # Plot portfolio vs individual assets
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(portfolio_cumulative.index, portfolio_cumulative,
                label="Equal-Weight Portfolio", linewidth=2, color="black")
        for asset in cumulative_returns_filtered.columns:
            ax.plot(cumulative_returns_filtered.index, cumulative_returns_filtered[asset],
                    label=asset, alpha=0.6)
        ax.set_title("Portfolio vs Individual Crypto Assets")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Growth")
        ax.legend()
        st.pyplot(fig)


    # --- TAB 2: Risk Contribution ---
    with tab2:
        st.subheader("Portfolio Risk Contribution")
        
        st.markdown("""
        This section shows how each asset contributes to the overall risk of the portfolio.  
        While portfolio weights reflect how much capital is allocated to each asset, risk contribution highlights how much volatility each asset adds to the portfolio as a whole.  
        By normalizing contributions into percentages, the table and pie chart reveal which assets dominate portfolio risk and whether diversification is effectively spreading risk.  
        This helps investors identify assets that may be disproportionately driving portfolio volatility and adjust allocations accordingly.
        """)


        # Exclude USD-USDT from analysis
        assets = [col for col in log_returns.columns if col != "USD-USDT"]
        log_returns_filtered = log_returns[assets]

        # Portfolio volatility
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)

        # Contribution to risk (marginal contribution)
        cov_matrix = log_returns_filtered.cov() * 252
        portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_volatility
        risk_contrib = weights * marginal_contrib

        # Normalize risk contributions to percentages
        risk_contrib_pct = risk_contrib / risk_contrib.sum()

        # Display risk contributions table
        risk_table = pd.DataFrame({
            "Weight": weights,
            "Risk Contribution (%)": risk_contrib_pct * 100
        }, index=log_returns_filtered.columns)

        st.write("### Risk Contribution Table (in %)")
        st.dataframe(risk_table.style.format({
            "Weight": "{:.2%}",
            "Risk Contribution (%)": "{:.2f}%"
        }))

        # Pie chart of risk contributions (percentages)
        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(risk_contrib_pct, labels=log_returns_filtered.columns,
               autopct="%1.1f%%", startangle=90)
        ax.set_title("Portfolio Risk Contribution by Asset")
        st.pyplot(fig)


    # --- TAB 3: Drawdowns ---
    with tab3:
        st.subheader("Portfolio Maximum Drawdowns")
        
        st.markdown("""

        This chart tracks the portfolio’s largest declines from previous peaks, known as drawdowns.  
        A drawdown measures how much the portfolio has fallen from its highest value, expressed as a percentage.  
        By visualizing drawdowns over time, investors can see periods of stress, the depth of losses, and how long it took to recover.  
        Understanding drawdowns is essential for risk management, since it highlights not just volatility but the potential severity of losses during downturns.
        """)


        # Portfolio drawdowns
        portfolio_cummax = portfolio_cumulative.cummax()
        portfolio_drawdown = (portfolio_cumulative - portfolio_cummax) / portfolio_cummax

        # Plot portfolio drawdowns
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(portfolio_drawdown.index, portfolio_drawdown,
                label="Portfolio Drawdown", color="red")
        ax.set_title("Portfolio Maximum Drawdowns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.legend()
        st.pyplot(fig)
