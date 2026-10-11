import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxhedge.backtest import STRATEGIES, run_backtest, summarize  # noqa: E402


def make_market(spot: np.ndarray, r_sek=0.02, r_eur=0.02, start="2020-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(spot))
    return pd.DataFrame({"spot": spot, "r_sek": r_sek, "r_eur": r_eur}, index=idx)


def test_constant_spot_equal_rates_all_strategies_identical():
    market = make_market(np.full(600, 11.0))
    res = run_backtest(market)
    assert len(res) > 0
    for s in STRATEGIES:
        assert res[f"eff_{s}"].to_numpy() == pytest.approx(11.0)


def test_payment_date_is_at_least_horizon_after_decision():
    market = make_market(np.full(600, 11.0))
    res = run_backtest(market, horizon_days=91)
    gap = (res["payment_date"] - res["decision_date"]).dt.days
    assert (gap >= 91).all() and (gap <= 95).all()


def test_no_look_ahead_in_volatility():
    # Vol estimate on a decision date must not change if future prices change.
    rng = np.random.default_rng(1)
    path = 11 * np.exp(np.cumsum(rng.normal(0, 0.005, 800)))
    base = run_backtest(make_market(path))
    shocked = path.copy()
    shocked[500:] *= 1.3
    alt = run_backtest(make_market(shocked))
    decided_before_shock = base["decision_date"] < make_market(path).index[500]
    assert np.allclose(base.loc[decided_before_shock, "sigma"], alt.loc[decided_before_shock, "sigma"])


def test_strategy_payoff_identities():
    rng = np.random.default_rng(2)
    path = 11 * np.exp(np.cumsum(rng.normal(0, 0.006, 900)))
    res = run_backtest(make_market(path, r_sek=0.01, r_eur=0.03))
    assert res["eff_half"].to_numpy() == pytest.approx(
        (0.5 * res["eff_unhedged"] + 0.5 * res["eff_forward"]).to_numpy()
    )
    floor = res["forward"] - res["put_premium"] * (1 + res["r_sek"] * res["tau"])
    # The option strategy never ends below the floor (strike minus financed premium)
    assert (res["eff_option"] >= floor - 1e-12).all()
    assert (res["eff_forward"] < res["spot_0"]).all()  # SEK rate < EUR rate -> forward discount


def test_option_protects_against_sharp_eur_fall():
    spot = np.full(600, 11.0)
    spot[300:] = 9.5  # EUR collapses
    res = run_backtest(make_market(spot))
    hit = res[(res["spot_T"] == 9.5) & (res["spot_0"] == 11.0)]
    assert len(hit) > 0
    assert (hit["eff_option"] > hit["eff_unhedged"]).all()
    assert (hit["eff_forward"] > hit["eff_unhedged"]).all()


def test_forward_hedge_has_lowest_dispersion():
    rng = np.random.default_rng(3)
    path = 11 * np.exp(np.cumsum(rng.normal(0, 0.006, 1500)))
    summary = summarize(run_backtest(make_market(path, r_sek=0.02, r_eur=0.03)))
    stds = summary["std_vs_spot_pct"]
    assert stds["forward"] < stds["option"] < stds["unhedged"]
    assert stds["forward"] < stds["half"] < stds["unhedged"]
