"""Pricing building blocks for FX hedging.

Conventions
-----------
* Spot ``S`` is quoted as SEK per 1 EUR (EUR/SEK, as published by Riksbanken).
* "Domestic" currency = SEK (the currency the exporter reports in).
* "Foreign" currency  = EUR (the currency the exporter is paid in).
* Interest rates are money-market rates in decimal form (0.02 = 2 %),
  simple compounding, ACT/360 day count.
"""

from __future__ import annotations

import math

import numpy as np

DAY_COUNT_BASIS = 360.0


def year_fraction(days: float) -> float:
    """ACT/360 year fraction for a period of ``days`` calendar days."""
    return days / DAY_COUNT_BASIS


def forward_rate(spot: float, r_dom: float, r_for: float, tau: float) -> float:
    """FX forward from covered interest parity (CIP).

    A bank can replicate a forward: borrow EUR today, convert to SEK at spot,
    deposit the SEK. No-arbitrage then gives

        F = S * (1 + r_SEK * tau) / (1 + r_EUR * tau)

    If SEK rates are lower than EUR rates, F < S: the forward "discount" is
    just the interest-rate differential, not a forecast of the future spot.
    """
    return spot * (1.0 + r_dom * tau) / (1.0 + r_for * tau)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def garman_kohlhagen(
    spot: float,
    strike: float,
    r_dom: float,
    r_for: float,
    sigma: float,
    tau: float,
    option: str = "put",
) -> float:
    """Garman-Kohlhagen (Black-Scholes for FX) price per 1 unit of foreign currency.

    Returns the premium in domestic currency (SEK per EUR of notional).
    Implemented in Black-76 form on the forward,
    ``P = DF_SEK * (K * N(-d2) - F * N(-d1))``, with discount factors from the
    simple money-market rates, so the model's forward equals
    :func:`forward_rate`. This is equivalent to textbook Garman-Kohlhagen
    with continuously compounded rates.

    A EUR **put** (= SEK call) gives the exporter the right to sell EUR at
    ``strike``: a floor on the SEK received, while keeping the upside.
    """
    if option not in {"put", "call"}:
        raise ValueError("option must be 'put' or 'call'")
    if tau <= 0:
        raise ValueError("tau must be positive")

    # Discount factors consistent with simple compounding.
    df_dom = 1.0 / (1.0 + r_dom * tau)
    df_for = 1.0 / (1.0 + r_for * tau)
    fwd = spot * df_for / df_dom  # identical to forward_rate(...)

    if sigma <= 0:
        # Zero volatility: the option is worth its discounted intrinsic value.
        intrinsic = max(strike - fwd, 0.0) if option == "put" else max(fwd - strike, 0.0)
        return df_dom * intrinsic

    vol_sqrt_t = sigma * math.sqrt(tau)
    d1 = (math.log(fwd / strike) + 0.5 * vol_sqrt_t**2) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    if option == "call":
        return df_dom * (fwd * _norm_cdf(d1) - strike * _norm_cdf(d2))
    return df_dom * (strike * _norm_cdf(-d2) - fwd * _norm_cdf(-d1))


def realized_volatility(spot: np.ndarray, trading_days_per_year: int = 252) -> float:
    """Annualised standard deviation of daily log returns.

    Used as a proxy for implied volatility, which is not freely available.
    """
    spot = np.asarray(spot, dtype=float)
    if spot.size < 3:
        raise ValueError("need at least 3 observations")
    log_ret = np.diff(np.log(spot))
    return float(np.std(log_ret, ddof=1) * math.sqrt(trading_days_per_year))
