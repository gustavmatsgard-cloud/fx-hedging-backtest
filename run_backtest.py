"""Run the hedging backtest on the downloaded data and write results + figures.

    python3 run_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fxhedge.backtest import STRATEGIES, run_backtest, summarize  # noqa: E402
from fxhedge.data import load_market  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LABELS = {
    "unhedged": "Unhedged",
    "forward": "Forward (100%)",
    "half": "Forward (50%)",
    "option": "Bought EUR put (ATMF)",
}


def plot_spot_and_forward(market, results, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(market.index, market["spot"], lw=0.8, color="0.3", label="EUR/SEK spot")
    ax.scatter(results["payment_date"], results["forward"], s=6, color="tab:blue",
               label="Forward rate locked 3 months earlier")
    ax.set_ylabel("SEK per EUR")
    ax.set_title("EUR/SEK and the forward rates the exporter could have locked in")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distribution(results, path):
    data = [(results[f"eff_{s}"] / results["spot_0"] - 1) * 100 for s in STRATEGIES]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, whis=(5, 95), showfliers=True)
    ax.set_xticks(range(1, len(STRATEGIES) + 1), [LABELS[s] for s in STRATEGIES])
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylabel("Effective rate vs spot on decision date (%)")
    ax.set_title("Distribution of outcomes per strategy (whiskers = 5th-95th percentile)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_timeline(results, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for s in STRATEGIES:
        rel = (results[f"eff_{s}"] / results["spot_0"] - 1) * 100
        ax.plot(results["payment_date"], rel, lw=1.0, label=LABELS[s])
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylabel("Effective rate vs spot on decision date (%)")
    ax.set_title("Outcome per payment over time")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def robustness_stibor(results):
    """Re-run 2006-2020 with STIBOR 3M instead of the T-bill rate.

    Treasury bills usually yield less than interbank rates, which lowers the
    computed SEK forward. This quantifies how much that choice matters.
    """
    if not (ROOT / "data" / "raw" / "sek_3m_stibor.csv").exists():
        return None
    alt = run_backtest(load_market(sek_rate_file="sek_3m_stibor.csv"))
    both = results.merge(alt, on="decision_date", suffixes=("_tb", "_stibor"))
    rel = lambda col, sfx: (both[f"{col}_{sfx}"] / both[f"spot_0_{sfx}"] - 1) * 100  # noqa: E731
    return {
        "decisions_compared": len(both),
        "period": f"{both['decision_date'].min():%Y-%m} .. {both['decision_date'].max():%Y-%m}",
        "mean_spread_stibor_minus_tbill_bp": ((both["r_sek_stibor"] - both["r_sek_tb"]) * 1e4).mean(),
        "mean_forward_diff_pct": ((both["forward_stibor"] / both["forward_tb"] - 1) * 100).mean(),
        "forward_outcome_mean_tb_pct": rel("eff_forward", "tb").mean(),
        "forward_outcome_mean_stibor_pct": rel("eff_forward", "stibor").mean(),
        "option_outcome_mean_tb_pct": rel("eff_option", "tb").mean(),
        "option_outcome_mean_stibor_pct": rel("eff_option", "stibor").mean(),
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    market = load_market()
    results = run_backtest(market)
    summary = summarize(results)

    results.to_csv(RESULTS / "backtest_results.csv", index=False)
    summary.to_csv(RESULTS / "summary.csv", float_format="%.4f")

    plot_spot_and_forward(market, results, FIGURES / "spot_and_forward.png")
    plot_distribution(results, FIGURES / "distribution.png")
    plot_timeline(results, FIGURES / "timeline.png")

    first, last = results["decision_date"].min(), results["payment_date"].max()
    print(f"{len(results)} hedging decisions, {first:%Y-%m} .. {last:%Y-%m}")
    print(f"Average realised vol used for option pricing: {results['sigma'].mean():.1%}")
    print(f"Average put premium: {(results['put_premium'] / results['spot_0']).mean():.2%} of spot\n")
    print(summary.round(2).to_string())

    robust = robustness_stibor(results)
    if robust:
        import pandas as pd

        pd.Series(robust).to_csv(RESULTS / "robustness_stibor.csv", header=["value"])
        print("\nRobustness, STIBOR instead of T-bill (2006-2020):")
        for key, value in robust.items():
            print(f"  {key:34s} {value:.3f}" if isinstance(value, float) else f"  {key:34s} {value}")


if __name__ == "__main__":
    main()
