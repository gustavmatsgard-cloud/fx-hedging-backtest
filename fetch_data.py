"""Download EUR/SEK and 3-month SEK/EUR interest rates from Riksbanken.

Uses Riksbanken's open SWEA API (no API key needed). Standard library only,
so it runs with any Python 3 installation:

    python3 fetch_data.py

Output (in data/raw/):
    eursek.csv, sek_3m.csv, eur_3m.csv   date,value
    sek_3m_stibor.csv                     STIBOR 3M 2006-2020 (robustness check)
    series_used.json                      which series were chosen and why
    series_catalog.json                   full list of available series
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://api.riksbank.se/swea/v1"
START_DATE = "2006-01-01"
RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
PAUSE_SECONDS = 13  # the anonymous API is rate limited; be polite

# Series ids, verified against the SWEA catalogue (GET /Series, Oct 2026).
#  * SEK 3M: Swedish Treasury Bill 3M (SETB3MBENCH). STIBOR 3M is no longer
#    published by Riksbanken after mid-2020, so it is only downloaded as a
#    robustness check for 2006-2020.
#  * EUR 3M: euro-market deposit rate for EUR, 3M (EUDP3MEUR).
SERIES = {
    "eursek": "SEKEURPMI",            # EUR/SEK fixing, SEK per EUR
    "sek_3m": "SETB3MBENCH",          # Swedish T-bill 3M, %
    "eur_3m": "EUDP3MEUR",            # Euro market rate EUR 3M, %
    "sek_3m_stibor": "SEDP3MSTIBORDELAYC",  # STIBOR 3M (ends 2020), robustness only
}
DESCRIPTION_FIELDS = ["seriesId", "shortDescription", "midDescription", "longDescription"]


def get_json(url: str, retries: int = 4):
    req = urllib.request.Request(url, headers={"User-Agent": "fx-hedging-backtest/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as err:
            if "CERTIFICATE_VERIFY_FAILED" in str(err):
                sys.exit(
                    "SSL certificates are missing for this Python installation.\n"
                    "On macOS (python.org installer), run once:\n"
                    '    open "/Applications/Python 3.X/Install Certificates.command"\n'
                    "(replace 3.X with your version), then run this script again."
                )
            if not isinstance(err, urllib.error.HTTPError):
                raise
            if err.code == 429 and attempt < retries - 1:
                wait = 60 * (attempt + 1)
                print(f"  rate limited, waiting {wait}s ...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"could not fetch {url}")


def describe(series: dict) -> str:
    """Id + descriptions only (not dates, which would match digit keywords)."""
    parts = [str(series[k]) for k in DESCRIPTION_FIELDS if series.get(k)]
    return " | ".join(parts) if parts else str(series.get("seriesId", ""))


def find_series(catalog: list[dict], name: str) -> dict:
    by_id = {series.get("seriesId"): series for series in catalog}
    try:
        return by_id[SERIES[name]]
    except KeyError:
        raise LookupError(name) from None


def fetch_observations(series_id: str, start: str, end: str) -> list[tuple[str, float]]:
    data = get_json(f"{BASE_URL}/Observations/{series_id}/{start}/{end}")
    if isinstance(data, dict):  # tolerate a wrapped response
        data = next((v for v in data.values() if isinstance(v, list)), [])
    rows = []
    for obs in data:
        value = obs.get("value")
        if value is None:
            continue
        rows.append((str(obs["date"])[:10], float(value)))
    return rows


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()

    print("Fetching series catalogue ...")
    catalog = get_json(f"{BASE_URL}/Series")
    (RAW_DIR / "series_catalog.json").write_text(json.dumps(catalog, indent=1, ensure_ascii=False))

    used = {}
    for name in SERIES:
        try:
            series = find_series(catalog, name)
        except LookupError:
            print(f"\nSeries {SERIES[name]} ('{name}') not found. Series with '3M' in the id:")
            for item in catalog:
                if "3M" in str(item.get("seriesId", "")):
                    print("   ", item.get("seriesId"), "-", describe(item)[:100])
            print("\nEdit SERIES in fetch_data.py and run again.")
            return 1

        sid = series["seriesId"]
        print(f"{name:13s} -> {sid}: {describe(series)[:90]}")
        time.sleep(PAUSE_SECONDS)
        rows = fetch_observations(sid, START_DATE, today)
        if not rows:
            print(f"  no observations returned for {sid}")
            return 1

        with open(RAW_DIR / f"{name}.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["date", "value"])
            writer.writerows(rows)
        used[name] = {"seriesId": sid, "description": describe(series), "observations": len(rows),
                      "first": rows[0][0], "last": rows[-1][0]}
        print(f"  {len(rows)} observations, {rows[0][0]} .. {rows[-1][0]}")

    (RAW_DIR / "series_used.json").write_text(json.dumps(used, indent=2, ensure_ascii=False))
    print("\nDone. Data saved in data/raw/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
