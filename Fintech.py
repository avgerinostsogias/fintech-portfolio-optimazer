import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

from Backtesting import run_backtest

# 1. Ρύθμιση Σελίδας
st.set_page_config(
    page_title="Live Portfolio Optimization Hub", page_icon="📈", layout="wide"
)

st.title("💼 Live FinTech: Monte Carlo Portfolio Optimization & Correlation")
st.write(
    "Εργαλείο ανάλυσης ρίσκου, απόδοσης και συσχετίσεων με **πραγματικά"
    " δεδομένα αγοράς** από το Yahoo Finance."
)

# 2. Sidebar: Επιλογές Χρήστη
st.sidebar.header("⚙️ Παράμετροι Χαρτοφυλακίου")

default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
tickers = st.sidebar.multiselect(
    "Επέλεξε Μετοχές (Tickers)",
    default_tickers,
    default=["AAPL", "MSFT", "NVDA", "AMZN"],
)

period = st.sidebar.selectbox(
    "Ιστορικό Δεδομένων", ["1y", "2y", "5y"], index=1
)
num_portfolios = st.sidebar.slider(
    "Αριθμός Προσομοιώσεων", 1000, 20000, 5000, 1000
)
risk_free_rate = (
    st.sidebar.slider("Επιτόκιο Χωρίς Ρίσκο (Risk-Free Rate %)", 0.0, 5.0, 2.0, 0.5)
    / 100
)

if not tickers or len(tickers) < 2:
    st.warning(
        "⚠️ Παρακαλώ επίλεξε τουλάχιστον **2 μετοχές** από το sidebar για να"
        " γίνει η βελτιστοποίηση."
    )
else:
    # 3. Λήψη Live Δεδομένων από Yahoo Finance (μαζί με το Benchmark SPY)
    @st.cache_data
    def load_data(tks, prd):
        all_tickers = list(set(tks + ["SPY"]))
        data = yf.download(
            all_tickers,
            period=prd,
            auto_adjust=True,
            progress=False
        )["Close"]
        return data


    raw_data = load_data(tickers, period)

    if isinstance(raw_data, pd.Series) or "SPY" not in raw_data.columns:
        st.error(
            "Παρουσιάστηκε σφάλμα στη λήψη των δεδομένων. Δοκίμασε ξανά."
        )
    else:
        # Καθαρισμός κενών τιμών
        raw_data = raw_data.dropna()

        # Διαχωρισμός μετοχών και benchmark
        stock_data = raw_data[tickers]
        spy_data = raw_data["SPY"]

        # Υπολογισμός ημερήσιων αποδόσεων
        daily_returns = stock_data.pct_change().dropna()
        mean_returns = daily_returns.mean() * 252
        cov_matrix = daily_returns.cov() * 252

        # Πίνακας Συσχετίσεων (Correlation Matrix)
        corr_matrix = daily_returns.corr()

        # Υπολογισμός στατιστικών για S&P 500 (SPY)
        spy_daily_return = spy_data.pct_change().dropna()
        spy_return = spy_daily_return.mean() * 252
        spy_volatility = spy_daily_return.std() * np.sqrt(252)

        num_assets = len(tickers)

        # 4. Προσομοίωση Monte Carlo
        #np.random.seed(42)
        results = np.zeros((3 + num_assets, num_portfolios))

        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)

            portfolio_return = np.sum(weights * mean_returns)
            portfolio_volatility = np.sqrt(
                np.dot(weights.T, np.dot(cov_matrix, weights))
            )
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

            results[0, i] = portfolio_return
            results[1, i] = portfolio_volatility
            results[2, i] = sharpe_ratio
            for j, w in enumerate(weights):
                results[3 + j, i] = w

        columns = ["Return", "Volatility", "Sharpe"] + list(tickers)
        df_res = pd.DataFrame(results.T, columns=columns)

        #max_sharpe_idx = df_res["Sharpe"].idxmax()
        #optimal_portfolio = df_res.iloc[max_sharpe_idx]
        mu = mean_returns.to_numpy()
        cov = cov_matrix.to_numpy()


        def negative_sharpe(weights):
            portfolio_return = weights @ mu
            portfolio_volatility = np.sqrt(weights @ cov @ weights)

            if portfolio_volatility < 1e-12:
                return 1e6

            return -(portfolio_return - risk_free_rate) / portfolio_volatility


        initial_weights = np.ones(num_assets) / num_assets

        bounds = [(0, 1)] * num_assets

        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1
            }
        ]

        result = minimize(
            negative_sharpe,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            st.error("Portfolio optimization failed.")
            st.stop()

        optimal_weights = result.x

        opt_return = optimal_weights @ mu
        opt_volatility = np.sqrt(
            optimal_weights @ cov @ optimal_weights
        )

        optimal_portfolio = pd.Series({
            "Return": opt_return,
            "Volatility": opt_volatility,
            "Sharpe": (opt_return - risk_free_rate) / opt_volatility,
            **dict(zip(tickers, optimal_weights))
        })

        equal_weights = np.ones(num_assets) / num_assets

        equal_return = float(equal_weights @ mean_returns.to_numpy())

        equal_volatility = float(np.sqrt(
            equal_weights @ cov_matrix.to_numpy() @ equal_weights
        ))

        equal_sharpe = (
            (equal_return - risk_free_rate) / equal_volatility
            if equal_volatility > 0 else np.nan
        )

        st.subheader("Portfolio Strategy Comparison")

        comparison = pd.DataFrame({
            "Strategy": ["Equal Weight", "Max Sharpe"],
            "Return (%)": [
                equal_return * 100,
                optimal_portfolio["Return"] * 100
            ],
            "Volatility (%)": [
                equal_volatility * 100,
                optimal_portfolio["Volatility"] * 100
            ],
            "Sharpe Ratio": [
                equal_sharpe,
                optimal_portfolio["Sharpe"]
            ]
        })

        st.dataframe(comparison.round(3), hide_index=True)

        # 5. Οπτικοποίηση & UI
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(
                "📈 Efficient Frontier & Benchmark Comparison (S&P 500)"
            )
            fig = go.Figure()

            # Scatter plot όλων των προσομοιώσεων
            fig.add_trace(
                go.Scatter(
                    x=df_res["Volatility"],
                    y=df_res["Return"],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=df_res["Sharpe"],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Sharpe Ratio"),
                    ),
                    name="Portfolios",
                )
            )

            # Σημείο Βέλτιστου Χαρτοφυλακίου (Max Sharpe)
            fig.add_trace(
                go.Scatter(
                    x=[optimal_portfolio["Volatility"]],
                    y=[optimal_portfolio["Return"]],
                    mode="markers",
                    marker=dict(size=16, color="red", symbol="star"),
                    name="Optimal Portfolio (Max Sharpe)",
                )
            )

            # Σημείο S&P 500 (Benchmark)
            fig.add_trace(
                go.Scatter(
                    x=[spy_volatility],
                    y=[spy_return],
                    mode="markers",
                    marker=dict(size=14, color="orange", symbol="diamond"),
                    name="S&P 500 (Benchmark)",
                )
            )

            fig.update_layout(
                xaxis_title="Ετήσιο Ρίσκο (Volatility - Std Dev)",
                yaxis_title="Αναμενόμενη Ετήσια Απόδοση",
                template="plotly_white",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Correlation Heatmap με Plotly Express ---
            st.subheader("🔗 Πίνακας Συσχετίσεων Μετοχών (Correlation Heatmap)")
            st.write(
                "Δείχνει πώς κινούνται οι μετοχές μεταξύ τους (τιμές κοντά στο"
                " 1.0 = κινούνται μαζί, κοντά στο 0 = ανεξάρτητες)."
            )

            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="Blues",
                aspect="auto",
            )
            fig_corr.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig_corr, use_container_width=True)

        with col2:
            st.subheader("⭐ Σύγκριση & Κατανομή")

            st.markdown("### Απόδοση vs S&P 500")
            st.metric(
                label="Optimal Portfolio Return",
                value=f"{optimal_portfolio['Return']*100:.2f}%",
                delta=f"{(optimal_portfolio['Return'] - spy_return)*100:.2f}% vs S&P",
            )
            st.metric(
                label="S&P 500 Return", value=f"{spy_return*100:.2f}%"
            )

            st.markdown("---")
            st.write("Προτεινόμενα ποσοστά επένδυσης:")
            for tkr in tickers:
                weight_val = optimal_portfolio[tkr] * 100
                st.metric(label=tkr, value=f"{weight_val:.2f}%")


        # 6. Out-of-sample Portfolio Backtesting
        st.divider()
        st.subheader("📈 Historical Portfolio Backtesting")

        initial_capital = st.number_input(
            "Initial Investment (€)",
            min_value=1000,
            value=10000,
            step=1000
        )

        try:
            bt = run_backtest(
                raw_data,
                tickers,
                initial_capital=initial_capital,
                risk_free_rate=risk_free_rate,
                train_fraction=0.70
            )
        except ValueError as error:
            st.error(str(error))

        else:
            st.caption(
                f"Training: {bt['training_start']:%Y-%m-%d}"
                f" to {bt['training_end']:%Y-%m-%d} | "
                f"Testing: {bt['testing_start']:%Y-%m-%d}"
                f" to {bt['testing_end']:%Y-%m-%d}"
            )

            fig_bt = go.Figure()

            for name in bt["equity"].columns:
                fig_bt.add_trace(go.Scatter(
                    x=bt["equity"].index,
                    y=bt["equity"][name],
                    mode="lines",
                    name=name
                ))

            fig_bt.update_layout(
                title="Portfolio vs S&P 500",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (€)",
                template="plotly_white",
                height=450
            )

            st.plotly_chart(fig_bt, use_container_width=True)

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Portfolio Final Value",
                    f"€{bt['portfolio_final']:,.2f}",
                    f"{bt['portfolio_return']:.2%}"
                )

            with c2:
                st.metric(
                    "S&P 500 Final Value",
                    f"€{bt['spy_final']:,.2f}",
                    f"{bt['spy_return']:.2%}"
                )

            with c3:
                st.metric(
                    "Maximum Drawdown",
                    f"{bt['portfolio_drawdown']:.2%}"
                )

            st.subheader("Portfolio Weights (Training Only)")
            st.dataframe(
                (bt["weights"] * 100).round(2).rename("Weight (%)"),
                use_container_width=True
            )

            st.caption(
                "Buy-and-hold backtest. No transaction fees, "
                "taxes, or EUR/USD currency movements."
            )
