"""Backtest of hedging strategies for a Swedish exporter paid in EUR.

Every month the exporter learns that it will receive a fixed EUR amount in
about three months. On that decision date it chooses one of:

* ``unhedged`` – do nothing, convert at the spot rate on the payment date.
* ``forward``  – lock in the 3-month forward rate today (CIP).
* ``half``     – hedge 50 % with a forward, leave 50 % open
                 (a common corporate hedge-policy compromise).
* ``option``   – buy an at-the-money-forward EUR put: a floor at the forward
                 rate, upside kept, premium paid today.

All outcomes are expressed as the *effective rate* actually achieved
(SEK received per EUR), so they are directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pricing import forward_rate, garman_kohlhagen, realized_volatility, year_fraction

STRATEGIES = ["unhedged", "forward", "half", "option"]


def decision_dates(index: pd.DatetimeIndex, min_history: int) -> list[pd.Timestamp]:
    """First available fixing date of every month, once enough history exists."""
    usable = index[min_history:]
    by_month = pd.Series(usable, index=usable).groupby([usable.year, usable.month]).min()
    return [pd.Timestamp(d) for d in by_month.values]


def run_backtest(
    market: pd.DataFrame,
    horizon_days: int = 91,
    vol_window: int = 63,
) -> pd.DataFrame:
    """Run the monthly hedging backtest.

    Parameters
    ----------
    market:
        Output of :func:`fxhedge.data.load_market` (columns spot, r_sek, r_eur).
    horizon_days:
        Calendar days between decision and payment (~3 months).
    vol_window:
        Number of daily returns used for the realised-volatility estimate
        (63 trading days ~ 3 months). Only data up to the decision date is
        used, so there is no look-ahead.
    """
    market = market.sort_index()
    idx = market.index
    rows = []

    for t0 in decision_dates(idx, min_history=vol_window + 1):
        target = t0 + pd.Timedelta(days=horizon_days)
        later = idx[idx >= target]
        if len(later) == 0:
            break  # payment date lies beyond the data
        t1 = later[0]

        s0, r_sek, r_eur = market.loc[t0, ["spot", "r_sek", "r_eur"]]
        s1 = market.loc[t1, "spot"]
        tau = year_fraction((t1 - t0).days)

        fwd = forward_rate(s0, r_sek, r_eur, tau)

        pos0 = idx.get_loc(t0)
        hist = market["spot"].iloc[pos0 - vol_window : pos0 + 1].to_numpy()
        sigma = realized_volatility(hist)

        strike = fwd  # at-the-money-forward
        premium = garman_kohlhagen(s0, strike, r_sek, r_eur, sigma, tau, option="put")
        premium_fv = premium * (1.0 + r_sek * tau)  # premium financed to payment date

        rows.append(
            {
                "decision_date": t0,
                "payment_date": t1,
                "spot_0": s0,
                "r_sek": r_sek,
                "r_eur": r_eur,
                "tau": tau,
                "forward": fwd,
                "sigma": sigma,
                "put_premium": premium,
                "spot_T": s1,
                "eff_unhedged": s1,
                "eff_forward": fwd,
                "eff_half": 0.5 * s1 + 0.5 * fwd,
                "eff_option": max(s1, strike) - premium_fv,
            }
        )

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, notional_eur: float = 1_000_000) -> pd.DataFrame:
    """Risk/return summary per strategy.

    The main metric is the effective rate relative to the spot on the decision
    date, in percent: how much better or worse than "today's rate" the
    exporter ended up. Variance in this number is exactly the FX risk.
    """
    out = {}
    for name in STRATEGIES:
        eff = results[f"eff_{name}"]
        rel = (eff / results["spot_0"] - 1.0) * 100.0
        out[name] = {
            "mean_sek_per_payment": eff.mean() * notional_eur,
            "mean_vs_spot_pct": rel.mean(),
            "std_vs_spot_pct": rel.std(ddof=1),
            "p5_vs_spot_pct": np.percentile(rel, 5),
            "worst_vs_spot_pct": rel.min(),
            "best_vs_spot_pct": rel.max(),
        }
    summary = pd.DataFrame(out).T
    summary.index.name = "strategy"
    return summary
