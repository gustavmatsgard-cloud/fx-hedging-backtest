# FX hedging backtest – should a Swedish exporter hedge its EUR income?

A Swedish company sells in euro but has its costs in SEK. Every month it knows
that a EUR 1 million payment will arrive in about three months. A weaker euro
in the meantime means fewer kronor. This project backtests four ways of
handling that risk on real EUR/SEK and interest-rate data from Riksbanken,
and measures what each strategy did to the level **and the variability** of
the SEK actually received.

| Strategy | What the treasurer does on the decision date |
|---|---|
| **Unhedged** | Nothing. Converts at the spot rate when the money arrives. |
| **Forward (100%)** | Locks in the 3-month forward rate for the full amount. |
| **Forward (50%)** | Locks in half, leaves half open (a common hedge-policy compromise). |
| **Bought EUR put (ATMF)** | Buys the right to sell EUR at the forward rate: a floor, upside kept, premium paid up front. |

## Method

**Data.** Downloaded from Riksbanken's open SWEA API by `fetch_data.py`
(details in `data/raw/series_used.json`):

| Input | Riksbanken series |
|---|---|
| EUR/SEK fixing (SEK per EUR) | `SEKEURPMI` |
| SEK 3-month rate | `SETB3MBENCH` – Swedish Treasury bill, 3M |
| EUR 3-month rate | `EUDP3MEUR` – euro-market deposit rate for EUR, 3M |
| Robustness check | `SEDP3MSTIBORDELAYC` – STIBOR 3M, published by Riksbanken only until 2020 |

**Forward rate (covered interest parity).** A bank can replicate a forward by
borrowing EUR, converting to SEK and depositing the SEK, so

$$F = S \cdot \frac{1 + r_{SEK}\,\tau}{1 + r_{EUR}\,\tau}, \qquad \tau = \frac{\text{days}}{360}$$

The forward premium or discount is therefore the interest-rate differential,
not a market forecast of where EUR/SEK is going.

**Option price (Garman–Kohlhagen).** The EUR put is struck at the forward
(at-the-money-forward) and priced with Garman–Kohlhagen, in Black-76 form on
the forward. Implied volatility is not freely available, so the volatility
input is the realised volatility of the previous 63 trading days. The premium
is financed at the SEK rate until the payment date.

**Outcome metric.** For each payment the *effective rate* (SEK received per
EUR, after any premium) is compared with the spot rate on the decision date:

$$\text{outcome} = \frac{\text{effective rate}}{S_{t_0}} - 1$$

Its mean says whether a strategy earned or cost money on average; its
dispersion (standard deviation, 5th percentile, worst case) is the FX risk
the company actually carried.

**No look-ahead.** Every number used on a decision date (spot, rates,
volatility) is known on that date. A unit test checks that shocking future
prices does not change past volatility estimates.

<!-- RESULTS -->

## Assumptions and limitations

- **Mismatched rate types.** The SEK rate is a government bill rate (STIBOR
  is no longer published by Riksbanken), the EUR rate an interbank deposit
  rate. Bills usually yield less than interbank rates, which pushes the
  computed forward slightly down. The STIBOR robustness check below
  quantifies the effect for 2006–2020.
- **Covered interest parity holds.** In practice there is a cross-currency
  basis and the bank adds a margin, so real forwards are slightly worse than
  the CIP rate used here.
- **Realised instead of implied volatility.** Implied volatility has
  historically tended to exceed realised volatility (the volatility risk
  premium), so the option premiums here are probably too low and the option
  strategy looks somewhat better than it would have in reality.
- **Mid rates, no transaction costs, one fixing per day.**
- **Overlapping exposures.** Monthly decisions with a three-month horizon mean
  that consecutive outcomes share price moves, so the observations are not
  independent. The effective sample size is smaller than the number of rows.
- **One currency pair and one company profile.** The conclusions are about
  EUR/SEK over this period, not about hedging in general.

## Possible extensions

- Use implied volatility and real forward points to remove the two biggest
  approximations above.
- Optimise the hedge ratio (minimum-variance hedge) instead of testing fixed
  ratios.
- Layered hedging: a rolling 12-month programme that hedges more of the near
  months than of the far months.
- Counterparty risk: the forward is an OTC derivative, so its credit
  valuation adjustment (CVA) could be estimated by simulating its exposure.

## How to run

```bash
python3 fetch_data.py            # download data from Riksbanken (standard library only)
pip install -r requirements.txt
python3 run_backtest.py          # results/ and figures/
python3 -m pytest                # unit tests
```

## Project structure

```
fetch_data.py            download EUR/SEK and 3M rates (Riksbanken SWEA API)
run_backtest.py          run the backtest, write results/ and figures/
src/fxhedge/pricing.py   forward (CIP), Garman–Kohlhagen, realised volatility
src/fxhedge/data.py      load and align the series (rates forward-filled, no look-ahead)
src/fxhedge/backtest.py  monthly hedge decisions, strategy outcomes, summary
tests/                   unit tests for pricing and backtest logic
```

---

Developed with AI-assisted coding (Claude). Modelling choices, code and
results reviewed by me.
