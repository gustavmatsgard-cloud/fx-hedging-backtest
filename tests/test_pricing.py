import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxhedge.pricing import (  # noqa: E402
    forward_rate,
    garman_kohlhagen,
    realized_volatility,
    year_fraction,
)


def test_forward_equals_spot_when_rates_equal():
    assert forward_rate(11.0, 0.03, 0.03, 0.25) == pytest.approx(11.0)


def test_forward_below_spot_when_sek_rate_is_lower():
    # Low SEK rate vs EUR rate -> EUR trades at a forward discount in SEK
    assert forward_rate(11.0, 0.01, 0.03, 0.25) < 11.0


def test_forward_matches_hand_calculation():
    # F = 11 * (1 + 0.04*0.25) / (1 + 0.02*0.25)
    assert forward_rate(11.0, 0.04, 0.02, 0.25) == pytest.approx(11.0 * 1.01 / 1.005)


def test_put_call_parity():
    s, k, rd, rf, vol, tau = 11.2, 11.0, 0.035, 0.02, 0.08, 0.25
    call = garman_kohlhagen(s, k, rd, rf, vol, tau, "call")
    put = garman_kohlhagen(s, k, rd, rf, vol, tau, "put")
    df_dom = 1 / (1 + rd * tau)
    fwd = forward_rate(s, rd, rf, tau)
    assert call - put == pytest.approx(df_dom * (fwd - k), abs=1e-12)


def test_zero_vol_gives_discounted_intrinsic_value():
    s, rd, rf, tau = 11.0, 0.03, 0.03, 0.25
    fwd = forward_rate(s, rd, rf, tau)
    assert garman_kohlhagen(s, fwd, rd, rf, 0.0, tau, "put") == pytest.approx(0.0)
    df_dom = 1 / (1 + rd * tau)
    assert garman_kohlhagen(s, 12.0, rd, rf, 0.0, tau, "put") == pytest.approx(df_dom * (12.0 - fwd))


def test_put_value_increases_with_volatility():
    prices = [garman_kohlhagen(11, 11, 0.03, 0.02, v, 0.25, "put") for v in (0.05, 0.10, 0.20)]
    assert prices[0] < prices[1] < prices[2]


def test_atm_forward_put_approximation():
    # For ATM-forward options: price ~ DF * F * sigma * sqrt(T) / sqrt(2*pi)
    s, rd, rf, vol, tau = 11.0, 0.03, 0.02, 0.07, 0.25
    fwd = forward_rate(s, rd, rf, tau)
    price = garman_kohlhagen(s, fwd, rd, rf, vol, tau, "put")
    approx = fwd * vol * math.sqrt(tau) / math.sqrt(2 * math.pi) / (1 + rd * tau)
    assert price == pytest.approx(approx, rel=0.01)


def test_realized_volatility_recovers_simulated_vol():
    rng = np.random.default_rng(0)
    sigma = 0.10
    returns = rng.normal(0, sigma / math.sqrt(252), size=20_000)
    path = 11.0 * np.exp(np.cumsum(returns))
    assert realized_volatility(path) == pytest.approx(sigma, rel=0.03)


def test_year_fraction_act_360():
    assert year_fraction(90) == pytest.approx(0.25)
