"""Load and align the market data downloaded by ``fetch_data.py``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MAX_STALENESS = pd.Timedelta(days=7)


def _read_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    s = df.set_index("date")["value"].astype(float).sort_index()
    return s[~s.index.duplicated(keep="last")]


def load_market(raw_dir: Path = RAW_DIR, sek_rate_file: str = "sek_3m.csv") -> pd.DataFrame:
    """Return a DataFrame indexed by EUR/SEK fixing dates with columns

    * ``spot``  – EUR/SEK (SEK per EUR)
    * ``r_sek`` – Swedish 3-month rate, decimal (T-bill by default;
      pass ``sek_rate_file="sek_3m_stibor.csv"`` for STIBOR, 2006-2020)
    * ``r_eur`` – euro 3-month rate, decimal

    Rates are published on slightly different calendars than the FX fixing,
    so they are forward-filled onto the fixing dates (the last known rate is
    what a treasurer would have used that day; no look-ahead).
    """
    spot = _read_series(raw_dir / "eursek.csv")
    r_sek = _read_series(raw_dir / sek_rate_file) / 100.0
    r_eur = _read_series(raw_dir / "eur_3m.csv") / 100.0

    # Carry each rate forward to the fixing dates, but never by more than
    # MAX_STALENESS: a discontinued series (e.g. STIBOR after 2020) must not
    # be extended indefinitely.
    def align(rate: pd.Series) -> pd.Series:
        return rate.reindex(spot.index, method="ffill", tolerance=MAX_STALENESS)

    market = pd.DataFrame({"spot": spot, "r_sek": align(r_sek), "r_eur": align(r_eur)})
    return market.dropna()
