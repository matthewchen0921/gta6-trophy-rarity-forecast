# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/matthewchen0921/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/matthewchen0921/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A refresh-aware earners-over-players forecaster for GTA VI's PS5 trophy earn
rates on Exophase. It reads the house's recorded reads of the trophy page up
to a case's checkpoint and forecasts each trophy's earn rate at the target as
five quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

## Why not persistence

Exophase refreshes a game's counts in jumps, hours apart, and a refresh
*dilutes* every rate: newly tracked players hold fewer trophies than the
players tracked so far (launch night: 638 → 3,868 players, 943 → 4,276
trophies earned, every rate but the 100% one fell). Saying "rates stay where
they are" is wrong at every refresh — the house baseline scores ~1.51 on the
practice while the refresh-aware leaders score 0.035–0.078.

## How it works

1. **Compress the history into refresh snapshots.** Consecutive identical
   reads are one refresh; the page is frozen between refreshes.
2. **Arrivals' earn rate.** At each refresh, trophy `i`'s arrivals earned at
   `u_i = (E_new − E_old) / (N_new − N_old)`, with `E = rate × N / 100`.
   This absorbs both new players' trophies and the old cohort's newly earned
   ones, which only surface at refreshes.
3. **Fit and step forward.** `N(t)` is fitted as a saturating exponential
   (a log-log fit overshoots 3× on a synthetic final); `u_i(t)` is fitted
   per trophy on log-time with shrinkage toward its tier's fit, capped at
   2.5× the highest arrivals rate observed. The model steps through the
   expected refreshes up to the last refresh before the target — the answer
   is the page as of that refresh, not the target itself — and reads off
   `r_i = E_i / N`.
4. **Spread.** From the level's standard error (fit residuals over
   √observations) plus horizon growth and refresh-lag drift, so bands widen
   over the 4.5-day final horizon and stay tight over a 90-minute practice
   horizon.
5. **Flat history (the practice cases).** No refresh observed yet, so it
   falls back to the published launch-night first refresh: the 100% trophy
   stayed 100%, so new non-100 earners were only 103 over 3,230 new
   players — arrivals earn each trophy at κ ≈ 0.067 of the tracked cohort's
   rate, and every other rate falls to ~0.22×.

## Local checks

Against synthetic practice answers derived from the published refresh
totals, this scores a pinball loss of 0.024 (common-κ answer) to 0.030
(tier-varying κ), vs 1.40 for the house baseline; on three synthetic finals
it scores 0.046–0.075 vs the baseline's 0.22–0.26. Standard library only,
deterministic, ~60 ms a case.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` with one `<id>.in` case in
`/input` and reads `/output/<id>.out`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=yg6b1b3z&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
