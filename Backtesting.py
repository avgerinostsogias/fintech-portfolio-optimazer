"""Out-of-sample, long-only, buy-and-hold backtest for the FinTech Streamlit app.

Training data determines SLSQP Max-Sharpe portfolio weights. The test period
is unseen during optimization. Portfolio and SPY both start at the specified
capital at the end-of-training close, with the next session being the first
fully out-of-sample return.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def max_drawdown(equity: pd.Series) -> float:
    """Return the largest peak-to-trough loss as a negative fraction."""
    return float((equity / equity.cummax() - 1.0).min())


def run_backtest(
    price_data: pd.DataFrame,
    tickers: list[str],
    *,
    initial_capital: float = 10_000.0,
    risk_free_rate: float = 0.02,
    train_fraction: float = 0.70,
) -> dict[str, Any]:
    """Optimize on earlier prices, then test with fixed shares on later prices.

    `price_data` must contain historical adjusted closing prices for all
    selected tickers and `SPY`, indexed chronologically by market date.

    The output equity series includes the last training date as the starting
    point, with both portfolios valued at `initial_capital` on that date.
    """
    if not 0.50 <= train_fraction <= 0.90:
        raise ValueError("Το training ποσοστό πρέπει να είναι μεταξύ 50% και 90%.")
    if initial_capital <= 0 or not np.isfinite(initial_capital):
        raise ValueError("Το αρχικό κεφάλαιο πρέπει να είναι θετικός αριθμός.")
    if not np.isfinite(risk_free_rate):
        raise ValueError("Το επιτόκιο χωρίς ρίσκο πρέπει να είναι έγκυρο.")

    assets = list(dict.fromkeys(tickers))
    if len(assets) < 2:
        raise ValueError("Χρειάζονται τουλάχιστον δύο διαφορετικές μετοχές.")
    if "SPY" in assets:
        raise ValueError("Το SPY χρησιμοποιείται μόνο ως benchmark εδώ.")
    required = assets + ["SPY"]
    missing = [asset for asset in required if asset not in price_data.columns]
    if missing:
        raise ValueError(f"Λείπουν τιμές για: {', '.join(missing)}")

    prices = price_data.loc[:, required].copy()
    prices = prices.sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna()
    if (prices <= 0).any().any():
        raise ValueError("Οι τιμές των μετοχών πρέπει να είναι θετικές.")

    cutoff = int(len(prices) * train_fraction)
    if cutoff < 60 or len(prices) - cutoff < 20:
        raise ValueError("Δεν υπάρχουν αρκετές ημερήσιες τιμές για το backtest.")

    training = prices.iloc[:cutoff]
    # Anchor at final training close; 1st trade-to-trade return is test-only.
    test_with_anchor = prices.iloc[cutoff - 1 :]
    train_returns = training[assets].pct_change().dropna()

    annual_mu = (train_returns.mean() * 252).to_numpy(dtype=float)
    annual_cov = (train_returns.cov() * 252).to_numpy(dtype=float)
    if not np.all(np.isfinite(annual_mu)) or not np.all(np.isfinite(annual_cov)):
        raise ValueError("Ανεπαρκή στοιχεία για τον υπολογισμό συνδιακυμάνσεων.")

    def negative_sharpe(weights: np.ndarray) -> float:
        expected_return = weights @ annual_mu
        variance = weights @ annual_cov @ weights
        sigma = np.sqrt(max(float(variance), 0.0))
        return 1e6 if sigma <= 1e-12 else -(expected_return - risk_free_rate) / sigma

    n = len(assets)
    result = minimize(
        negative_sharpe,
        x0=np.full(n, 1 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Η βελτιστοποίηση δεν ολοκληρώθηκε: {result.message}")

    weights = np.clip(result.x, 0.0, 1.0)
    weights = weights / weights.sum()
    weights = pd.Series(weights, index=assets, name="Weight")

    # Buy and hold: choose shares once, on final training close, no rebalance.
    indexed_stock_prices = test_with_anchor[assets].div(test_with_anchor[assets].iloc[0])
    portfolio_equity = initial_capital * indexed_stock_prices.dot(weights)
    spy_equity = initial_capital * test_with_anchor["SPY"] / test_with_anchor["SPY"].iloc[0]
    equity = pd.DataFrame({"Optimized Portfolio": portfolio_equity, "S&P 500 (SPY)": spy_equity})

    return {
        "equity": equity,
        "weights": weights,
        "training_start": training.index[0],
        "training_end": training.index[-1],
        "testing_start": prices.index[cutoff],
        "testing_end": prices.index[-1],
        "training_days": len(training),
        "testing_days": len(prices) - cutoff,
        "portfolio_final": float(equity["Optimized Portfolio"].iloc[-1]),
        "spy_final": float(equity["S&P 500 (SPY)"].iloc[-1]),
        "portfolio_return": float(equity["Optimized Portfolio"].iloc[-1] / initial_capital - 1),
        "spy_return": float(equity["S&P 500 (SPY)"].iloc[-1] / initial_capital - 1),
        "portfolio_drawdown": max_drawdown(equity["Optimized Portfolio"]),
        "spy_drawdown": max_drawdown(equity["S&P 500 (SPY)"]),
    }
