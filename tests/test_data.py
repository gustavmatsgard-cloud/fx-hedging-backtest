import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxhedge.data import load_market  # noqa: E402


def write(path: Path, dates, values) -> None:
    pd.DataFrame({"date": dates, "value": values}).to_csv(path, index=False)


def test_rates_forward_filled_and_converted_to_decimal(tmp_path):
    days = pd.bdate_range("2020-01-01", periods=10).strftime("%Y-%m-%d")
    write(tmp_path / "eursek.csv", days, [11.0] * 10)
    write(tmp_path / "sek_3m.csv", days[::5], [2.0, 3.0])  # published less often
    write(tmp_path / "eur_3m.csv", days, [1.0] * 10)
    market = load_market(tmp_path)
    assert len(market) == 10
    assert market["r_sek"].iloc[4] == 0.02  # carried forward, in decimal
    assert market["r_sek"].iloc[5] == 0.03


def test_discontinued_rate_series_is_not_extended(tmp_path):
    days = pd.bdate_range("2020-01-01", periods=40).strftime("%Y-%m-%d")
    write(tmp_path / "eursek.csv", days, [11.0] * 40)
    write(tmp_path / "sek_3m_stibor.csv", days[:6], [2.0] * 6)  # series stops
    write(tmp_path / "eur_3m.csv", days, [1.0] * 40)
    market = load_market(tmp_path, sek_rate_file="sek_3m_stibor.csv")
    last_obs = pd.Timestamp(days[5])
    assert market.index.max() <= last_obs + pd.Timedelta(days=7)
    assert market.index.max() < pd.Timestamp(days[-1])
