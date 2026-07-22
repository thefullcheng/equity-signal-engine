# Equity Signal Engine

[![tests](https://github.com/paulywally123/equity-signal-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/paulywally123/equity-signal-engine/actions/workflows/tests.yml)

Cross-sectional monthly long-only equity strategy on a point-in-time S&P 500
universe with an overfitting-resistant walk-forward backtest.

**Contents**: [Results](#results-as-of-latest-run) ·
[Decomposition](#decomposition-how-much-is-the-model-actually-contributing) ·
[Factor regression](#factor-regression-is-this-just-known-factor-exposure) ·
[Sector exposure](#sector-exposure-is-this-just-long-tech) ·
[Statistical significance](#statistical-significance-is-this-distinguishable-from-luck) ·
[Market-timing overlay](#market-timing-overlay-five-attempts-to-reduce-drawdown-none-survive-validation) ·
[Phases completed](#phases-completed) ·
[Feature set](#feature-set-5-features-all-rank-normalised-cross-sectionally) ·
[Sensitivity summary](#sensitivity-summary) ·
[Quickstart](#quickstart) ·
[Key design decisions](#key-design-decisions) ·
[Live paper-trading protocol](#live-paper-trading-evaluation-protocol-pre-registered) ·
[Survivorship bias handling](#survivorship-bias-handling)

## Results (as of latest run)

*Reproduce these exact numbers*: `config/config.yaml` with `universe.mode: full`,
price/fundamentals data snapshot dated 2026-07-08, `python -m src.backtest.build_backtest`.
Numbers will drift on a fresh data pull as prices/fundamentals update.

**`top_q=0.15`, not the `0.20` used everywhere else in this document below.**
`0.20` turned out to be an in-sample pick from `sensitivity.py`'s full-sample
sweep — the same kind of leak already caught for feature selection.
`top_q_holdout.py` selected `0.15` using only pre-2020 data, then confirmed
it on the untouched 2020-2026 holdout (Sharpe 0.531 vs 0.478 for `0.20`,
same direction in both windows — a block-bootstrap on just the holdout gap
put it at 95% CI `[-0.004, +0.156]`, P(>0)=96.2%: a hair from conventional
significance, not a clean pass, but far stronger than noise). `top_q` is now
a real `config.yaml` entry (`portfolio.top_q`) instead of a hardcoded
literal. **This entire document has been regenerated at `0.15`** —
Decomposition, Factor regression, Sector exposure, all of Statistical
significance, Sensitivity, and all five market-timing overlays — with two
narrow, explicitly-flagged exceptions (dev-mode comparison, embargo-purge
effect size) that need a different kind of rerun than the rest. See
Phases completed (`6v`) for the full list of what changed and what didn't.

| Metric | Value |
|---|---|
| Sharpe ratio | **0.719** |
| CAGR | **11.0%** |
| Ann. volatility | 16.5% |
| Max drawdown | -31.0% |
| Hit rate (strategy / benchmark) | 67.5% / 66.3% |
| Avg turnover (per rebalance / annualized, two-sided\*) | 41.1% / 534.6% |
| Annualized cost drag | 0.53%/yr |
| IC (model vs realized) | +0.0050 (t = 0.48) |
| Excess return vs equal-weight BM | +2.36% / yr |
| Backtest period | 2013 – 2026 (169 periods) |
| Universe | Full point-in-time S&P 500 (607 current members, 814 ever-members) |

IC is unchanged from the `top_q=0.20` run — it's computed from predictions
vs. realized returns across the whole ranked universe, independent of how
many names the portfolio actually holds. `top_q` only moves the portfolio
construction and cost/turnover numbers, not the underlying model score's
own accuracy.

\* Turnover here is two-sided (sum of `\|weight_new - weight_old\|` — a full
portfolio replacement reads as 2.0, not 1.0), and the 10bps cost is applied
directly to that two-sided figure once (10bps per side is already baked in,
not 10bps applied twice). The 534.6%/yr annualized turnover number therefore
implies 0.53%/yr of cost drag, not ~1%/yr — worth being precise about, since
the two-sided-vs-one-sided distinction is an easy way to double- or
half-count this.

The **benchmark's hit rate (66.3%) is nearly identical to the strategy's
(67.5%)** — a reminder that "positive most months" is what any long-only
strategy does in a rising market, not evidence of stock-picking skill on
its own.

**The excess return and Sharpe gap above are point estimates from 169
periods, not precise measurements.** 95% block-bootstrap CIs (rebuilt at
`top_q=0.15`): excess return [-0.33%, +5.07%]/yr (P(>0)=95.7%), Sharpe gap
[-0.09, +0.31] (P(>0)=90.8%) — both still technically span zero, though the
excess-return interval is now close enough that it's nearly, not
comfortably, insignificant. See **Statistical significance** for
methodology and robustness checks (that section's own numbers are still the
`top_q=0.20` ones, per the scope note above).

![Equity curve, drawdown, and IC over time](docs/equity_curve.png)

Strategy: long top-15% by model score, 20-day rebalance, no market-timing
overlay, 10 bps per-side transaction costs. **No short leg** — the long-short
design was the original intent (isolating the cross-sectional signal by
canceling market beta), but shorting costs and hard-to-borrow constraints on
smaller/delisted names aren't modeled in this backtest's data, so long-only is
the more realistic framing given that limitation, at the cost of not cleanly
separating stock-selection skill from market beta. See **Decomposition**
below for how much of the return is attributable to each.

## Decomposition: how much is the model actually contributing?

A long-only, trend-filtered strategy's Sharpe conflates three things: equity
beta over a strong bull period, any market-timing overlay, and the
cross-sectional model. This breaks it apart (rows (c)/(d) rerun at
`top_q=0.15`; (a)/(b) don't use the model's predictions at all, so they're
unaffected by `top_q` and unchanged):

| Configuration | Sharpe | Ann. return | Vol | Max DD | CAGR |
|---|---|---|---|---|---|
| (a) Equal-weight buy-and-hold, no regime, no model | 0.570 | 9.83% | 17.25% | -37.8% | 8.60% |
| (b) Equal-weight + 200d regime filter, no model | 0.472 | 5.35% | 11.33% | -22.5% | 4.80% |
| **(c) Model portfolio, no regime filter (current default)** | **0.719** | **11.87%** | 16.51% | -31.0% | **11.00%** |
| (d) Model + 200d regime filter | 0.505 | 5.89% | 11.67% | -24.1% | 5.33% |

Two things this shows plainly:

1. **The regime filter hurts, in both cases it's applied** — (a)→(b) and
   (c)→(d) both lose Sharpe. It was originally intended as a risk-reducing
   market-timing overlay; once built on a correct equal-weight *return* index
   (see below), it doesn't earn its keep. It's not used by default.
2. **The model beats passive buy-and-hold on this specific historical sample**
   — (c) beats (a) by both Sharpe and absolute return. Whether that's a real,
   generalizable edge or a good draw from noise is a separate question — see
   **Statistical significance** and **Factor regression** below. The honest
   answer, once you control for known factor exposures rather than just raw
   buy-and-hold, is worse than "not confidently distinguishable from luck" —
   it's a significantly *negative* risk-adjusted alpha.

**A bug this decomposition surfaced**: the original regime filter averaged
raw *price levels* across the universe (`prices.mean(axis=1)`) rather than
compounding daily *returns* — a price-weighted index, not equal-weighted (a
$500 stock moved it 100x more than a $5 stock for the same % move). Fixed in
`equal_weight_index()` (`src/backtest/backtest.py`). Once fixed, the
200-day window — which happened to sit at the top of its own sensitivity
sweep under the old (buggy) index — no longer looks favorable at any window;
see **Sensitivity summary**.

## Factor regression: is this just known factor exposure?

The decomposition above answers "model vs. no model." The industry-standard
version of that question is a factor regression: does the strategy's return
survive controlling for the Fama-French 5 factors (market, size, value,
profitability, investment) plus momentum? Daily factor data from Ken
French's data library, compounded to match each backtest period's exact
date range (`src/backtest/factor_data.py`, `build_factor_regression.py`).

Rerun at `top_q=0.15` (was `0.20` — see Results):

| Factor | Coefficient | t-stat | p-value |
|---|---|---|---|
| Alpha (period) | -0.0036 | **-3.82** | <0.001 |
| Mkt-RF | 1.038 | 41.88 | <0.001 |
| SMB (size) | 0.089 | 2.20 | 0.029 |
| HML (value) | -0.082 | -2.25 | 0.026 |
| RMW (profitability) | 0.183 | 3.78 | <0.001 |
| CMA (investment) | 0.088 | 1.42 | 0.156 |
| MOM (momentum) | 0.058 | 2.07 | 0.040 |

R² = 0.940 (adj. 0.938), n = 168 periods. **Annualized alpha: -4.55%/yr,
still strongly significant.**

Same exposures as before, with two changes worth naming rather than
smoothing over: HML (value) is now significant (t=-2.25, was t=-1.63,
p=0.105) at a small negative loading, and MOM's loading is noticeably
weaker (t=2.07, was t=3.03) — a more concentrated top-15% book apparently
leans a little less on pure momentum and picks up a touch of value-averse
exposure instead. Neither changes the headline: near-1.0 market beta,
significant profitability tilt (RMW — `roe`/`gross_prof`), still a real
(if smaller) momentum tilt (MOM — `mom_12_1`). 94% of the strategy's return
variance is explained by six well-known, freely-investable factors — down
only slightly from 94.3%. **The residual — the part attributable to this
specific model's stock selection, beyond just having those tilts — is
still significantly *negative***, not merely insignificant. Rechecked the
transaction-cost question too: gross (pre-cost) alpha is -4.03%/yr
(t=-3.37), barely different from the cost-inclusive -4.55%/yr — same
conclusion as before, this isn't costs, the specific stocks chosen within
the factor tilts underperform what the tilts alone would predict, even
before paying to trade them.

**The core finding is robust to the `top_q` fix.** Fixing a real in-sample
leak improved the headline Sharpe (0.665→0.719) without rescuing the
model's claim to genuine stock-picking skill — alpha is still significantly
negative, R² is still ~94%. That's exactly what should happen if the
`top_q` bug was really about portfolio construction and not about hiding a
deeper problem with the signal itself.

This is the most direct answer yet to "isn't this just momentum beta in ML
clothing?" — yes, and once you control for that, there's no stock-picking
skill left to find.

## Sector exposure: is this just long tech?

The RMW/MOM factor loadings above predict a quality+momentum tilt that, over
this sample, would concentrate in tech. Measured directly — long-leg
(top-15%, `top_q=0.15` — see Results) sector weights vs. the full
equal-weight universe (unaffected by `top_q`, so unchanged from before):

| Sector | Long leg | Universe | Tilt |
|---|---|---|---|
| Technology | 21.0% | 13.2% | **+7.8pp** |
| Consumer Defensive | 11.6% | 7.6% | +4.0pp |
| Healthcare | 13.9% | 11.9% | +2.0pp |
| Consumer Cyclical | 14.9% | 13.1% | +1.8pp |
| Industrials | 14.3% | 14.1% | +0.2pp |
| Basic Materials | 3.7% | 4.4% | -0.7pp |
| Communication Services | 3.6% | 4.2% | -0.6pp |
| Energy | 1.6% | 4.6% | -3.0pp |
| Real Estate | 3.0% | 6.1% | -3.1pp |
| Utilities | 2.5% | 6.3% | -3.8pp |
| Financial Services | 9.8% | 14.0% | **-4.2pp** |

Answer: still tilted toward tech (~1.6x overweight, slightly stronger than
the `top_q=0.20` book's ~1.5x), still underweight financials and utilities,
still not "all in tech" — the largest single sector is 21% of the book.
Re-ran with sector-neutral construction (equal representation by sector,
`sectors` kwarg in `run_backtest`) at `top_q=0.15`: **Sharpe drops from
0.719 to 0.525** (by 0.194, a bigger drop than the previous 0.135 at
`top_q=0.20`). That direction makes sense mechanically — a smaller,
15%-of-universe book has more room for sector concentration to move the
number than a 20%-of-universe one does — but it also means **sector
positioning explains a larger share of the model's apparent edge now than
it did before the `top_q` fix**, not a smaller one. Worth sitting with: the
`top_q` change that improved the headline Sharpe also made the strategy
lean somewhat more on sector bets and somewhat less on pure within-sector
stock selection.

## Statistical significance: is this distinguishable from luck?

**Block-bootstrap CI on the headline comparison.** The Results table states
"+2.36%/yr excess" and the Decomposition table shows "Sharpe 0.719 vs.
0.570" as if they were precise measurements — with only 169 non-overlapping
periods, they aren't. Used a circular moving-block bootstrap (not iid
resampling, to preserve any serial dependence — performance can cluster in
trending vs. choppy regimes) on the strategy-minus-benchmark return series,
2,000 draws, block length ≈ 1 year (13 periods). Rerun at `top_q=0.15`:

| | Mean | 95% CI | P(> 0) |
|---|---|---|---|
| Annualized excess return | +2.41%/yr | [-0.33%, +5.07%]/yr | 95.7% |
| Sharpe gap (strategy - benchmark) | +0.141 | [-0.09, +0.31] | 90.8% |

**Both intervals still (barely) span zero, but this is closer to the edge
than the `top_q=0.20` version was.** Checked robustness to the block-length
choice (6, 13, 20, 30 periods) — and here, unlike before, the conclusion
does *shift a little* with the choice: block lengths 6/13/20 all still span
zero for excess return (P(>0) 95.5%, 95.7%, 97.0%), but **block=30 does
not** — 95% CI [+0.07%, +4.52%]/yr, P(>0)=97.7%. The Sharpe-gap CI spans
zero at all four block lengths regardless (P(>0) 91.0%-94.2%). So: this is
no longer a clean "both intervals comfortably span zero at every
robustness check" result the way it was at `top_q=0.20` — it's a real
point right at the boundary of significance, sensitive to a bootstrap
nuisance parameter, which is itself the honest description rather than
rounding it up to "significant" or down to "clearly not." `src/backtest/
bootstrap.py`, `build_bootstrap_ci.py`.

**Momentum-only baseline.** A naive 12-1 month momentum sort (no model, no
fundamentals, no LightGBM) over the identical 169 periods:

| | IC t-stat | Sharpe | CAGR |
|---|---|---|---|
| Momentum-only (single feature) | 0.62 | 0.638 | 7.9% |
| Full model (5 features + LightGBM) | 0.48 | 0.719 | 11.0% |

Rerun at `top_q=0.15` (`momentum_baseline.py`, reconstructed as a real
script — the original comparison predates this repo's current script
layout). **The point-estimate gap is now bigger than it used to be** —
+0.081 Sharpe, not the "rounding error" +0.012 the `top_q=0.20` version of
this table showed — but the full model still has the *lower* IC doing it,
same as before. Given how much attention this project pays to not trusting
a point estimate, that gap got the same treatment as the LightGBM-vs-Ridge
comparison: a block-bootstrap CI on the model-minus-momentum return series
(same methodology as Results' headline CI). It spans zero — Sharpe gap 95%
CI [-0.186, +0.348], P(>0)=76.1%; annualized excess return 95% CI [-1.12%,
+7.49%]/yr, P(>0)=93.3%. **So the honest conclusion is unchanged even
though the number moved**: the full pipeline's edge over sorting by
trailing 12-month return alone is not statistically distinguishable from
noise at 169 periods, and it still has worse raw ranking accuracy while
getting there. This is the honest complexity-vs-payoff comparison an
interviewer would run in their head.

**Permutation null test.** Shuffled which ticker gets which predicted score
within each rebalance date (destroying any real score↔outcome relationship
while preserving the cross-sectional and time-series structure), recomputed
IC, repeated 2,000 times to build a null distribution:

| | Value |
|---|---|
| Observed IC mean | +0.0050 |
| Null distribution mean | +0.0001 (centered at ~0, as expected) |
| Null distribution std | 0.0036 |
| **Empirical p-value** (P(null IC ≥ observed)) | **0.089** |
| Observed IC's percentile in the null | 91st |

**This does not clear conventional significance (p < 0.05).** The observed
IC is directionally positive and beats ~91% of random permutations, but a
p-value of 0.089 means we can't confidently rule out that this is a good
draw from noise rather than genuine, generalizable predictive skill.

**Full-strategy permutation null (Sharpe, not just IC).** The IC test above
checks raw ranking accuracy; a separate question is whether the *complete,
cost-inclusive strategy* — top-15% selection, 10bps costs, actual turnover —
beats naive random stock-picking. Ran the exact same portfolio construction
on 1,000 random rankings (shuffled scores, not just shuffled IC pairings).
Reconstructed as a real script (`sharpe_permutation.py` — the original
predates this repo's current layout) with an explicit guard against the
exact bug the next paragraph describes, and rerun at `top_q=0.15`:

| | Value |
|---|---|
| Observed Sharpe | 0.719 |
| Null distribution mean | 0.433 |
| Null distribution std | 0.052 |
| **Empirical p-value** (P(null Sharpe ≥ observed)) | **< 0.001** |
| Observed Sharpe's percentile in the null | 100th (beats all 1,000) |

Same conclusion as before, numbers barely moved. This clears significance
decisively — but two things temper what it actually means. First, the null
mean is 0.43, not 0, because even a fully random long-only S&P 500
portfolio (~56 names at `top_q=0.15`, vs. ~75 at the old `0.20`) captures
real market beta over this bull market; this tests "beats random
stock-picking," not "beats doing nothing." Second, and more important:
**turnover explains a real chunk of this gap, separate from ranking
skill.** The model's picks come from slow-moving features (12-month
momentum, ROE), so month-to-month selections overlap heavily — 41%
turnover vs. ~170% median for fully-reshuffled random rankings, a real
cost-drag difference, not a predictive-accuracy difference. So the honest
reconciliation of the two null tests: raw period-by-period ranking accuracy
is marginal (IC p=0.089, unaffected by `top_q`), but the full strategy's
persistence in its picks (low turnover) gives it a genuine, structural cost
advantage over naive high-turnover random selection (Sharpe p<0.001). Both
are true; they answer different questions, and neither alone is the whole
picture.

(This test also caught a real bug in its own construction the first time it
was built: an early version used `.stack(future_stack=True)` without
dropping the resulting NaN padding, silently breaking the top-quintile
selection for most permutations and producing an invalid p=0.007. Caught
via a turnover sanity check — the buggy version showed ~0.05 average
turnover for "fully random" rankings, implausible for genuine reshuffling.
`sharpe_permutation.py`'s reconstruction bakes that check in directly —
it raises rather than reports a p-value if the null distribution's median
turnover looks implausibly low — and it passed clean this time, 1.70 median
vs. the ~0.05 the broken version produced.)

Combined with the momentum-only comparison and the factor regression above,
the fair summary of this whole project is: **rigorous, correctly-built
infrastructure (point-in-time universe, walk-forward validation, EDGAR
integration) that outperforms naive random stock-picking largely through
known factor exposures (market beta, profitability, momentum) and lower
turnover, not novel stock-selection skill — and once those factors are
controlled for, the residual alpha is significantly negative, not merely
insignificant.** This is a well-built research pipeline that correctly
identifies its own strategy as a repackaging of established factors rather
than a validated edge.

**Feature selection was in-sample — tested whether fixing that changes the
conclusion; it doesn't.** Both the original 8→5 trim and the earlier Phase 4b
pruning used IC measured over the same 2013–2026 span the backtest reports
performance over. To check the impact, re-derived a feature set using
*only* pre-2020 IC (a proper held-out selection window): under the same
threshold rule, only `mom_12_1` and `roe` survive — `dollar_vol_60`, one of
the 5 currently used, actually shows *negative* IC pre-2020 (t=-0.16) despite
being the strongest feature in the 2020-2026 period (t=1.43). Individual
feature IC is that unstable across sub-periods; several features flip sign
entirely (`mom_1`: t=-2.00 pre-2020 → t=+0.90 after; `rsi_14`: t=-1.96 →
t=+0.93).

Compared honestly on the true 2020-2026 holdout (never used to select either
feature set). Reconstructed as a real script (`feature_selection_holdout.py`
— the original predates this repo's current layout) and rerun at
`top_q=0.15`; the IC t-stats below are exactly unchanged from the
`top_q=0.20` version, as expected since IC doesn't depend on portfolio
construction:

| Feature set | Full-period Sharpe / IC t | Holdout-only Sharpe / IC t |
|---|---|---|
| 5-feature (in-sample selected, current default) | 0.719 / 0.48 | 0.531 / **0.18** |
| 2-feature (honestly selected, pre-2020 only) | 0.701 / 0.22 | 0.493 / **-0.54** |

The honestly-selected set still doesn't generalize better — still worse on
the true holdout, same conclusion as before. Kept the 5-feature set as the
default, since switching doesn't objectively improve anything measurable.
The real conclusion isn't "5 vs 2
features" — it's that the underlying signal is too weak for feature selection
to meaningfully discriminate at all, which is the same conclusion the
permutation test already reached from a different angle.

**New feature tested and rejected: insider trading (Form 4).** Built a
point-in-time panel from SEC's bulk quarterly Form 3/4/5 datasets
(`src/data/insider.py`), filtered to genuine open-market purchases/sales only
(`TRANS_CODE` in `{P, S}` — excludes option exercises, tax withholding,
grants, gifts, which carry no signal about an insider's actual view), using
filing date as the point-in-time availability date (Form 4 has a strict
2-business-day filing deadline, a much tighter lag than the fundamentals
issue fixed earlier). Feature: net buy/sell count ratio over a trailing
90-day window.

Validated with the same held-out methodology as the feature-pruning
investigation above, *before* deciding whether to add it:

| Window | insider_score IC t-stat |
|---|---|
| Selection window (pre-2020) | **-0.32** |
| Holdout (2020-2026) | 0.50 |
| Full period | 0.06 |

Doesn't clear the same bar (|t|≥0.5 on the selection window) applied to every
other feature — the selection-window t-stat is negative. **Not added to the
model.** Deliberately didn't try alternate constructions (different window
lengths, dollar-weighting, restricting to officers/directors) after seeing
this result — iterating until some variant clears the bar is exactly the
in-sample selection problem this section exists to avoid. The panel-building
code and cached data remain in the repo if a differently-motivated
construction is worth testing later, decided in advance rather than reverse-
engineered from a result.

## Market-timing overlay: five attempts to reduce drawdown, none survive validation

The Decomposition section above already found that a 200-day SMA regime
filter reduces Sharpe in every configuration tested. Given the strategy is
long-only with ~1.0 market beta (see Factor regression) and offers no
protection of its own in a falling market — 2018: -4.64% ann. return, -17.3%
max DD; 2020 COVID: -31.0% max DD, the worst drawdown in the whole backtest;
2022: -8.96% ann. return, -18.7% max DD (annual attribution at `top_q=0.15`,
`src.report.attribution`) — the natural next question is whether *any*
design can teach it to reduce exposure in a bad market without giving back
more than it saves.

Five conceptually different designs were tried, each validated the same way
the feature-selection check above was: select parameters using *only*
pre-2020 periods (88 of the 169), then evaluate that one chosen config on
the true 2020-2026 holdout (81 periods, containing both the COVID crash and
the 2022 bear market) it never saw. All five functions
(`vol_target_exposure`, `trend_exposure`, `drawdown_trigger_exposure`,
`level_trigger_exposure` in `src/backtest/backtest.py`) and their sweep
scripts remain in the repo, available but **not used by default** — same
status as the pre-existing `regime`/`sectors` kwargs. Nothing about the live
paper-trading accounts changed as a result of this investigation.

**1. Binary 200-day trend filter.** Already covered above: full exit to
cash below the 200d SMA. Sharpe 0.719 → 0.505, CAGR 11.0% → 5.33%, max DD
-31.0% → -24.1%. Rejected — costs more than it saves on a full-sample
basis, before even reaching the holdout-validation question the other four
went through. (Rerun at `top_q=0.15`; conclusion unchanged from the
`top_q=0.20` version of this check.)

**2. Continuous vol-target exposure** (`vol_scaling.py`). Rather than a
binary switch, scale position size continuously so trailing realized vol
tracks a target — the portfolio is never fully out, so it can't miss a
recovery entirely the way a binary exit can. Rerun at `top_q=0.15`:

| target_vol | best window | Sharpe | CAGR | max DD | avg exposure |
|---|---|---|---|---|---|
| 12% | 10d | 0.641 | 7.25% | -20.5% | 84% |
| 16.5% (≈ baseline realized vol) | 10d | 0.691 | 9.02% | -26.6% | 93% |
| 20% | 10d | 0.715 | 9.73% | -26.3% | 96% |
| **Baseline (no overlay)** | — | **0.719** | **11.00%** | **-31.0%** | 100% |

No configuration across the full 3×3 grid beats the no-overlay baseline —
same conclusion as before, and the gap to the best config actually narrowed
slightly (0.719 vs 0.715, was 0.665 vs 0.661). Bad-year detail for the best
config (target_vol=20%, window=10d): 2020's max DD improves (-31.0% →
-20.5%) but its return is roughly halved (+15.4% → +8.2%) — the exact
whipsaw problem continuous scaling was meant to avoid, just softened rather
than eliminated; 2022 gets worse on both counts (-9.0% → -13.2% return)
because that bear market was a slow grind, not an early vol spike, so the
signal didn't react in time but still clipped 2022-10-18 (one of the five
best individual periods in the entire 13-year backtest). Rejected.

**3. Partial trend filter — a tunable floor** (`drawdown_cap.py`,
`drawdown_cap_holdout.py`). Generalises #1: instead of fully exiting below
the 200d trend, keep a `floor` fraction of exposure (0-95%). Swept 4
windows × 11 floors (44 configs) on the full sample, rerun at `top_q=0.15`:

| Config | Sharpe | CAGR | max DD | Calmar (CAGR / \|DD\|) |
|---|---|---|---|---|
| **Baseline (no filter)** | 0.719 | 11.00% | -31.0% | 0.355 |
| window=200, floor=0.0 (= attempt #1) | 0.505 | 5.33% | -24.1% | 0.221 |
| window=250, floor=0.70 | **0.725** | 9.74% | -25.4% | **0.383** |
| window=250, floor=0.75-0.85 | 0.721-0.725 | 9.96-10.38% | -26.3 to -28.2% | 0.37-0.38 |

A plateau at window=250, floor≈0.65-0.85 still dominates the baseline on
*both* Sharpe and Calmar — a narrower margin than the `top_q=0.20` version
of this check (that one's plateau led by ~0.007 Sharpe / ~0.03 Calmar; this
one leads by ~0.006 Sharpe / ~0.03 Calmar, about the same), and it
genuinely does have to clear that bar first this time: an earlier,
uncorrected rerun of this script briefly and incorrectly showed the
plateau falling *below* baseline even in-sample, from a bug where only the
baseline call got the new `top_q` and the swept configs silently kept the
old one — worth naming since it's exactly the kind of apples-to-oranges
mistake this whole exercise exists to catch, caught here by comparing
against a second, independent full rerun rather than trusting the first
one. **It still doesn't survive honest re-selection.** Picking (window,
floor) using only pre-2020 data (ranked by Calmar) again chose window=250,
**floor=0.0** — the full binary exit — for the same reason as before:
pre-2020's bad stretches (2015, 2018) were mild enough that a full exit's
whipsaw cost looked small there. Tested on the 2020-2026 holdout:

| | Sharpe | CAGR | max DD | Calmar |
|---|---|---|---|---|
| Baseline, holdout | **0.531** | 8.63% | -31.0% | **0.278** |
| Honestly-selected (250, floor=0.0), holdout | 0.231 | 2.18% | -25.4% | 0.086 |
| In-sample pick (250, floor=0.75), holdout | 0.501 | 7.19% | -26.3% | 0.273 |

Both still underperform the no-filter baseline out of sample, by almost
exactly the same margins as the `top_q=0.20` version of this check (honest
pick: Calmar gap 0.192 now vs. 0.192 before; in-sample pick: Calmar gap
0.005 now vs. 0.006 before) — the honestly-selected config still collapses
because it happened to optimize for ordinary corrections, then met the
fastest crash-and-V-recovery in the whole dataset, and the full-sample
"plateau" is still fit to the same data it's evaluated on. Rejected, same
as before — this is the one overlay result the `top_q` fix left almost
completely untouched.

**4. Fast tail-drawdown trigger** (`tail_trigger_holdout.py`). A
structurally different design: instead of a slow 200-250 day trend (slow
to re-enter after a rally), react to the index's drawdown from its own
trailing peak over a *short* window (10-40 days), so it re-arms as soon as
the market actually recovers. Swept lookback ∈ {10,20,40} × threshold ∈
{8,10,15,20%} × floor ∈ {0, 0.5}, selected on pre-2020 (same winner as
before: lookback=40, threshold=8%, floor=0.5), evaluated on holdout at
`top_q=0.15`:

| | Sharpe | CAGR | max DD | Calmar |
|---|---|---|---|---|
| Baseline, holdout | **0.531** | 8.63% | -31.0% | **0.278** |
| Honestly-selected trigger, holdout | 0.472 | 6.35% | -25.9% | 0.245 |

Still closest of the five attempts to breaking even, and the Sharpe/Calmar
gaps to baseline are essentially unchanged from the `top_q=0.20` version
(Calmar gap 0.033 both times). Still cuts 2020's max DD (-31.0% → -21.7%),
still underperforms baseline on Sharpe and Calmar, and still makes 2022
worse (-9.0% → -13.8% return for almost no drawdown improvement there).
Rejected, same as before.

**5. Independent macro signals — VIX and credit spreads**
(`macro_data.py`, `macro_trigger_holdout.py`). The first signal *not*
derived from this strategy's own price/return history at all — pulled from
FRED (free, no API key). VIX (`VIXCLS`) has full history back to 1990. The
obvious credit-spread choice, ICE BofA US High Yield OAS
(`BAMLH0A0HYM2`), turned out to be capped by FRED/ICE's licensing terms to
a rolling ~3-year window regardless of the requested start date (confirmed
empirically — returns the same ~795 rows starting 2023-07 no matter what
`cosd` is passed), which would silently drop the entire pre-2020 selection
window and the COVID crash. Used Moody's Baa corporate bond yield minus
the 10-year Treasury (`BAA10Y`) instead — an unrestricted, standard
credit-spread stress proxy with full history back to 1986.

Rerun at `top_q=0.15`, same selected thresholds both times (VIX=40, BAA10Y=4.0):

| | Sharpe | CAGR | max DD | Calmar |
|---|---|---|---|---|
| Baseline, holdout | **0.531** | 8.63% | -31.0% | **0.278** |
| VIX (threshold=40, honestly selected), holdout | 0.465 | 7.04% | -31.0% | 0.227 |
| BAA10Y (threshold=4.0, honestly selected), holdout | 0.531 | 8.63% | -31.0% | 0.278 |

The BAA10Y config is identical to baseline to four decimal places — the
honestly-selected threshold barely got touched even at the COVID peak (the
spread topped out at 4.31 for essentially one day), so it never fired on a
rebalance date. The VIX config did fire — and cost return — but **the max
drawdown is unchanged to the decimal even though it fired.** That's the key
finding: a genuinely independent, real-time, well-known stress indicator
still can't help here, because the strategy only rebalances every 20
trading days. The COVID crash and its sharpest rebound days (2020-04-03,
2020-05-04) sat only 2-6 weeks apart — once a rebalance date cuts exposure,
that decision is locked in for the full next holding period regardless of
what any signal does three days later. No overlay can react faster than the
schedule that acts on it, which is the structural reason all five designs —
however different conceptually — converged on the same result. Rejected.

**Conclusion.** This isn't five unlucky parameter choices; it's a
consistent finding across price-trend, volatility, price-drawdown, and two
independent macro signals. On this data, with a 20-trading-day rebalance
cadence, there's no timing/exposure-reduction mechanism that reliably
reduces drawdown without giving back more than it saves. Reducing exposure
in a bad market, if it happens at all, is a decision to make at the
account-allocation level, not something to encode into the model.

## Phases completed

- [x] **2a** Point-in-time S&P 500 universe (Wikipedia change log, backward reconstruction)
- [x] **2b** Price ingestion + coverage audit (`src/data/prices.py`)
- [x] **2c** Cleaning & membership masking (`src/data/clean.py`)
- [x] **3a** Cross-sectional features — momentum, volatility, RSI, dollar-vol (`src/features/`)
- [x] **3b** Forward return labels with cross-sectional rank normalisation (`src/labels/`)
- [x] **3c** Walk-forward LightGBM model — annual re-fit, no hyperparameter tuning on test data (`src/models/`)
- [x] **3d** Long-only backtest + equity curve report (`src/backtest/`, `src/report/`)
- [x] **4a** SEC EDGAR fundamental data — point-in-time observations, 2007–2026 (`src/data/edgar.py`)
- [x] **4b** Feature pruning; sector-neutral construction (available, not default — costs 0.135 Sharpe on the current model, see Sector exposure)
- [x] **5a** Annual attribution tearsheet + sector exposure + worst/best periods (`src/report/attribution.py`)
- [x] **5b** Sensitivity analysis — top_q, cost assumptions, regime window (`src/backtest/sensitivity.py`)
- [x] **5c** Live signal generation — current ranked portfolio (`src/signal/`)
- [x] **5d** Documentation (this file)
- [x] **6a** Fixed EDGAR filing-date bug — median point-in-time lag was 406 days (89% > 180 days) due to later filings' comparative-period restatements overwriting the original filing date; now 36 days median
- [x] **6b** Trimmed to 5 features with real IC, regularized the LightGBM model — reduced overfitting to the dev-mode ticker subset (train/test gap 2.7x → 1.2x on a matched liquidity slice)
- [x] **6c** Fixed regime index (price-weighted → equal-weight return) and dropped the regime filter given the decomposition above
- [x] **6d** Fixed live signal generation always lagging ~1 rebalance cycle behind reality regardless of data freshness (`predict_latest()` in `src/models/model.py`)
- [x] **6e** Alpaca paper-trading integration (`src/trading/`) — dry-run by default, equal-weight rebalance to the model's top-N
- [x] **6f** Fixed walk-forward embargo gap — see Key design decisions
- [x] **6g** Dev-mode robustness test against random 100-ticker subsets — see Dev mode section
- [x] **6h** Momentum-only baseline and permutation null test — see Statistical significance
- [x] **6i** Investigated in-sample feature pruning via held-out selection window — see Statistical significance
- [x] **6j** Multi-strategy/multi-account paper trading — momentum-only baseline running in parallel with the full model, on separate Alpaca accounts (`--strategy`/`--account` in `src/trading/`)
- [x] **6k** Built and tested an insider-trading (Form 4) feature; rejected after held-out validation — see Statistical significance
- [x] **6l** Added turnover metric, equity curve chart, CI, and reproduction note (results table, `docs/equity_curve.png`, `.github/workflows/`)
- [x] **6m** Full-strategy Sharpe permutation null (1,000 random rankings, exact portfolio construction + costs) — see Statistical significance
- [x] **6n** Fama-French 5 + momentum factor regression — see Factor regression
- [x] **6o** Block-bootstrap CI on the headline excess-return/Sharpe-gap claims — see Statistical significance
- [x] **6p** Sector exposure table + re-verified sector-neutral Sharpe cost (0.54 -> 0.135 on the current model) — see Sector exposure
- [x] **6q** Benchmark hit rate, precise cost-drag math, log-scale equity curve, table of contents
- [x] **6r** Pre-registered the live paper-trading evaluation protocol (horizon, metrics, confirmation/failure criteria) before any checkpoint was assessed
- [x] **6s** Tested five market-timing/drawdown-reduction overlay designs (vol-target scaling, tunable partial trend filter, fast tail-drawdown trigger, VIX/credit-spread triggers) with honest pre-2020-selection/2020-2026-holdout validation on each; all either reduced Sharpe outright or failed holdout validation — see "Market-timing overlay" section. Root cause is structural (20-day rebalance cadence), not signal quality; none adopted, live accounts unaffected
- [x] **6t** Found `top_q=0.20` was an in-sample pick (same leak class as feature selection, just never checked for this parameter) — `top_q_holdout.py` selected `0.15` on pre-2020 data only, confirmed on the 2020-2026 holdout (Sharpe 0.531 vs 0.478), block-bootstrapped the holdout gap (95% CI [-0.004, +0.156], P(>0)=96.2% — close to but short of conventional significance). Adopted as the new default (`config/config.yaml`, `portfolio.top_q`)
- [x] **6u** LightGBM vs. a plain Ridge model on identical features/labels/walk-forward split, to test whether the nonlinear model earns its complexity (`src/backtest/model_comparison.py`). First pass was confounded (Ridge can't fit through NaN features the way LightGBM's native missing-value splits can, and dropping them lost 63% of the panel); rerun with LightGBM also restricted to the same row-matched subset showed a +0.041 Sharpe edge for LightGBM, but a block-bootstrap CI on that gap spans zero (95% CI [-0.124, +0.217], P(>0)=72.6%, `model_comparison_bootstrap_ci.py`) — not distinguishable from noise at 169 periods. Third independent complexity-vs-simplicity comparison in this project (after momentum-only-baseline and buy-and-hold) to reach the same conclusion
- [x] **6v** Regenerated the entire document at `top_q=0.15`: Decomposition, Factor regression (core finding survives — alpha still significantly negative, R² still ~94%; HML became significant, MOM weakened), Sector exposure (sector-neutral cost *increased* to 0.194 from 0.135 — the fix leans more on sector positioning, not less, worth sitting with rather than glossing over), Sensitivity (flagged that its own top_q sweep would lead you right back into the in-sample trap if read naively), all five market-timing overlays (same conclusions throughout; one caught-and-fixed bug along the way where a rerun only updated the baseline call and silently left the swept configs on the old `top_q`), and reconstructed three analyses whose original code was never committed to this repo — `momentum_baseline.py` (gap widened to +0.081 Sharpe but bootstrap CI still spans zero, P(>0)=76.1%), `sharpe_permutation.py` (same conclusion, includes a turnover sanity check that would have caught this test's own previously-documented NaN-padding bug had it recurred), `feature_selection_holdout.py` (IC t-stats exactly unchanged as expected, only Sharpe moved). Two items deliberately left unrerun and flagged individually: the dev-mode/random-100 comparison (needs a full data-pipeline rerun under a different `universe.mode`) and the embargo-purge effect size (needs retraining with the purge reverted)

## Feature set (5 features, all rank-normalised cross-sectionally)

Started at 8; `mom_3`, `mom_1`, `rsi_14` were excluded from training from the
start (negative individual IC). `mom_6_1`, `high_52w`, `vol_21` were later
dropped after measuring ~zero individual IC — see **Statistical
significance** ("Feature selection was in-sample") for why that specific
decision has a methodology caveat of its own.

| Feature | Type | IC | t-stat |
|---|---|---|---|
| `mom_12_1` | Price — 12-1 month momentum | +0.012 | 0.85 |
| `dollar_vol_60` | Price — 60-day avg dollar volume | +0.006 | 0.85 |
| `gross_prof` | Fundamental — gross profit / assets (Novy-Marx) | +0.002 | 0.21 |
| `roe` | Fundamental — net income / equity | +0.011 | 1.34 |
| `ep_ratio` | Fundamental — earnings / market cap | -0.004 | -0.44 |

None of these individually clears a conventional significance bar, and with
5+ features tested, `roe`'s t=1.34 shouldn't be read as a standout finding —
it's what you'd expect to see somewhere in this set by chance alone. The
model's IC (t=0.48, combining all features nonlinearly via LightGBM) is the
number that actually matters, and it's modest.

## Sensitivity summary

Rerun at the new `top_q=0.15` baseline:

| Dimension | Sharpe range | Baseline |
|---|---|---|
| top_q (0.10 – 0.30) | 0.632 – 0.749 | 0.719 (top_q=0.15) |
| costs_bps (5 – 20) | 0.687 – 0.735 | 0.719 (10 bps) |
| regime window (100 – 250d, vs. no filter) | 0.410 – 0.719 | **0.719 (no filter — best of all options)** |

Unlike the pre-fix version of this table, the baseline (no regime filter) is
no longer sitting at a suspicious optimum within its own sweep — adding *any*
regime window makes things worse, monotonically. Cost sensitivity is a
smooth, unremarkable curve around the baseline.

**top_q sensitivity is not smooth, and reading it naively would walk you
straight back into the trap this whole exercise was about.** In this
full-2013-2026-sample sweep, `top_q=0.10` shows Sharpe **0.749** — higher
than `0.15`'s 0.719. That's the identical in-sample-selection failure mode
that made `0.20` look fine for years: this table is swept on the same data
it's reported over, so its own "best cell" isn't trustworthy, regardless of
which value currently sits there. `top_q_holdout.py` already ran the honest
version of this exact question — select using only pre-2020 data, confirm
on the 2020-2026 holdout — and it preferred `0.15` over `0.10` there (pre-2020
Sharpe 0.981 vs 0.953; see Results). `0.15` is the adopted default *because*
of that holdout check, not because of this table. Don't re-derive a "better"
top_q from this sweep alone.

**This table is itself the in-sample sweep that `top_q_holdout.py` later
showed picked a slightly wrong `top_q`** (see Results, top). `top_q=0.20`'s
baseline here comes from exactly the same full-2013-2026-sample sweep that
was already flagged as a leak risk for feature selection — it just hadn't
been checked for this parameter until now. **Update**: this whole document
has since been regenerated at the honestly-selected `top_q=0.15` —
Decomposition, Factor regression, Sector exposure, the full Statistical
significance section (bootstrap CI, momentum-only baseline, both
permutation nulls, the feature-selection holdout table), Sensitivity, and
all five market-timing overlay holdouts. Three items are the deliberate
exceptions, flagged individually where they appear rather than silently
skipped: the dev-mode/random-100-ticker comparison (needs the entire data
pipeline rerun under `universe.mode: dev`, not just a portfolio-
construction parameter) and the embargo-purge effect size (needs retraining
with the purge temporarily reverted, a code change). The IC permutation
null and the insider-trading feature check needed no changes at all — both
are pure rank-correlation measures, unaffected by `top_q` by construction.

## Quickstart

```bash
pip install -r requirements.txt
pytest                                        # 73 tests

# Run the full pipeline (requires network access):
py -m src.data.build_universe
py -m src.data.build_prices
py -m src.data.build_clean
py -m src.data.build_fundamentals            # ~10 min, SEC EDGAR
py -m src.data.build_insider                 # optional -- built & tested, not used by the model (see Statistical significance)
py -m src.features.build_features
py -m src.labels.build_labels
py -m src.models.build_model
py -m src.backtest.build_backtest            # prints attribution tearsheet
py -m src.backtest.sensitivity               # parameter sensitivity table
py -m src.backtest.build_factor_regression   # Fama-French 5 + momentum regression
py -m src.backtest.build_bootstrap_ci        # block-bootstrap CI on excess return / Sharpe gap
py -m src.backtest.vol_scaling               # vol-target exposure scaling sweep
py -m src.backtest.drawdown_cap              # tunable partial trend-filter sweep
py -m src.backtest.drawdown_cap_holdout      # honest pre-2020/2020-holdout validation of the above
py -m src.backtest.tail_trigger_holdout      # fast tail-drawdown trigger, holdout-validated
py -m src.backtest.macro_trigger_holdout     # VIX / credit-spread trigger, holdout-validated (fetches FRED data)
py -m src.backtest.top_q_holdout             # honest pre-2020/2020-holdout selection of top_q
py -m src.backtest.top_q_bootstrap_ci        # block-bootstrap CI on the top_q=0.15 vs 0.20 gap
py -m src.backtest.model_comparison          # LightGBM vs. Ridge, row-matched, same features/labels
py -m src.backtest.model_comparison_bootstrap_ci  # block-bootstrap CI on the LightGBM-vs-Ridge gap
py -m src.backtest.momentum_baseline         # momentum-only vs. full model, identical dates/costs
py -m src.backtest.feature_selection_holdout # honest pre-2020/2020-holdout validation of feature selection
py -m src.backtest.sharpe_permutation        # full-strategy Sharpe permutation null (1,000 reshuffles)
py -m src.signal.build_live_signal           # today's portfolio holdings
py -m src.signal.build_momentum_signal       # momentum-only baseline signal

# Paper-trade via Alpaca (dry run by default; --strategy model|momentum):
py -m src.trading.rebalance                  # prints order plan only
py -m src.trading.rebalance --execute        # submits orders
```

## Key design decisions

**Point-in-time universe**: membership reconstructed backward from Wikipedia's
S&P 500 change log. Prices outside a ticker's membership window are masked to
NaN, preventing survivorship bias.

**Walk-forward evaluation**: model re-fits annually on an expanding training
window. No hyperparameters are tuned on test data — the biggest silent risk in
rolling-window backtests. The rebalance grid is spaced exactly `horizon_days`
trading days apart, so the single most recent pre-cutoff training date has a
label that resolves at approximately the same time as the first test date of
the following year — near-zero embargo at each of the 13 annual boundaries.
That date is now purged from each year's training set (`model.py`,
`walk_forward_predict`). Effect was real but small: Sharpe 0.705 → 0.665,
CAGR 10.57% → 10.0% (measured at the `top_q=0.20` default current at the
time; not rerun at `0.15` — doing so means retraining with the purge
temporarily reverted, a code change rather than a config one, out of scope
for this pass. No reason to expect the *qualitative* finding — purging
this one date costs a little Sharpe but removes a real leak — depends on
`top_q`).

**Ranked training labels**: forward returns are converted to cross-sectional
percentile ranks [0, 1] for model training, giving a stable target distribution
across volatile and quiet markets. Raw log returns are used for backtest P&L.

**No regime/market-timing filter**: tested a 200-day SMA cash overlay; once
built on a correct equal-weight return index it reduced Sharpe in every
configuration (see Decomposition). Not used by default.

**EDGAR fundamentals**: quarterly GAAP data fetched from the public SEC EDGAR
XBRL API (no key required). Filing date used as the availability date, taking
the *earliest* filing that reports each period (a later 10-K's five-year
comparative table can otherwise overwrite the true filing date with a much
later one — this was a real bug, see Phase 6a). The API only returns a filing
*date*, not time-of-day, so same-day after-hours filings aren't pushed to the
next trading day; given the 20-trading-day rebalance grid this has low
practical impact, but it's a known, undodged limitation of the data source.

**Dev mode** (`universe.mode: dev`): restricts each date's universe to the
top-100 tickers by trailing dollar volume, for fast iteration. Its metrics
run meaningfully hotter than full-universe. Tested whether this was
overfitting to that specific liquid subset vs. a general small-sample effect
by rerunning on three genuinely random 100-ticker draws:

| Universe | IC t-stat | Sharpe | CAGR |
|---|---|---|---|
| Dev-mode (top-100-by-liquidity) | 0.87 | 0.753 | 13.7% |
| Random-100 (seed=42) | 1.03 | 0.542 | 7.9% |
| Random-100 (seed=123) | 1.57 | 0.564 | 9.6% |
| Random-100 (seed=7) | 0.76 | 0.885 | 14.5% |
| **Full universe (607, honest number)** | **0.48** | **0.719** | **11.0%** |

The four dev-mode/random-100 rows above are **not** rerun at `top_q=0.15`
— unlike everything else in this document, they need the entire data
pipeline re-executed under `universe.mode: dev` (different features/labels/
predictions entirely, not just a portfolio-construction parameter), which
is a materially heavier job than the rest of this pass. Only the full-
universe row, which reuses this document's existing headline numbers, has
been updated. Treat the dev-mode comparison qualitatively (the spread
across random 100-ticker draws, not the exact Sharpe values) until it's
rerun.

IC doesn't vanish on random subsets — comparable to or higher than the
liquidity-selected one — which rules out the narrowest overfitting concern
(features/model curve-fit specifically to that original recurring ~100-name
set). But every 100-ticker sample, however chosen, shows noisier IC and a
nearly 2x Sharpe spread (0.54–0.89) purely from which 100 companies happen to
be in the sample. Dev mode was never a stable estimate of anything, for any
100-ticker subset — full mode is the only credible number to report; dev mode
is for fast local iteration only.

## Live paper-trading evaluation protocol (pre-registered)

Two Alpaca paper accounts have been running since **2026-07-08**: one
tracking the full model's top-20, one tracking the momentum-only baseline
(`src/trading/`, `--strategy model|momentum`). Writing the evaluation
criteria down now, before any live checkpoint has been assessed, so the
conclusion isn't fitted to whatever the data happens to show.

**Evaluation horizon**: 12 months, **2026-07-08 to 2027-07-08** (~13
rebalance cycles at the 20-trading-day schedule).

**Metrics tracked**: cumulative return spread (model account vs. momentum
account — the direct, paired, same-market-conditions comparison), live IC
(model's predicted score vs. realized forward return, computed each
rebalance), realized turnover (sanity check against the backtest's 40.2%),
and any operational failures (data pipeline breaks, execution slippage).

**What would count as confirmation, and what wouldn't.** Given everything
above — factor-regression alpha significantly *negative* (t=-4.79), the
excess-return and Sharpe-gap bootstrap CIs both spanning zero on 169
backtest periods — the honest prior going into this live test is that the
model has **no expected edge over the momentum baseline**. With only ~13
live periods (a fraction of the 169 that still couldn't rule out noise),
this test cannot by itself confirm or refute a stock-picking edge with any
statistical confidence — that would need years of live data, not one. So:

- **The model account outperforming the momentum account over 12 months is
  NOT, by itself, evidence of skill.** Given the small sample, a wide
  performance gap in either direction is well within plausible noise.
- **What this live run can actually check**: does execution behave as the
  backtest predicts (turnover near 40%, no operational surprises), and is
  the qualitative pattern (which account does better, by how much) at least
  *consistent with* — not contradicted by — the backtest's own conclusion
  of no significant edge. A dramatic, sustained divergence either way would
  be worth investigating, not immediately believing.
- **Success for this project was never "the model account makes more
  money."** It's an honest, well-calibrated backtest whose live behavior
  doesn't contradict what the backtest itself already concluded. If the two
  accounts track closely, or diverge in ways too small to distinguish from
  noise, that's the expected, confirming outcome — not a null result.

## Survivorship bias handling

1. Change-log completeness trusted only back to `universe.floor_date` (2010-01-01).
2. Ticker renames may appear as spurious remove/add pairs — inspect coverage audit.
3. Some delisted tickers have no retrievable price history (177 empty of 814 tickers).
