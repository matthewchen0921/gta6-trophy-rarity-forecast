"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles (5/25/50/75/95%),
from the house's recorded reads of the trophy page up to the case's
checkpoint.  Standard library only, no network, deterministic.

Why this shape, not the house baseline (persistence + a random-walk spread,
practice loss ~1.51): Exophase does not update its counts continuously.  It
refreshes a game's page in jumps, hours apart, and a refresh *dilutes* every
rate: the newly tracked players hold fewer trophies than the players tracked
so far (launch night: 638 -> 3,868 players, 943 -> 4,276 trophies earned,
every rate but the 100% one fell).  A forecaster that says "rates stay where
they are" is therefore wrong at every refresh, and the practice board shows
it: the leaders (loss 0.035-0.078 vs the baselines' ~1.51) all model the
refresh as an earners-over-players (E/N) decomposition with a dilution
factor.  This program does the same, fitted from the case's own history:

  1. Compress the history into refresh snapshots (consecutive identical
     reads are one refresh; the page is frozen between refreshes).
  2. At each refresh, the arrivals' earn rate for trophy i is
     u_i = (E_i_new - E_i_old) / (N_new - N_old), where E_i = rate_i * N / 100.
     u_i is the sufficient statistic: it absorbs both new players' trophies
     and the old cohort's newly earned ones, which only surface at refreshes.
  3. Fit N(t) (players tracked) as a saturating exponential and u_i(t)
     (arrivals' earn rate, per trophy, shrunk toward its tier's fit) against
     log-time since release, then step forward through the expected refreshes
     up to the last refresh before the target (the answer is the page as of
     that refresh, not the target itself) and read off r_i = E_i / N.
  4. Quantile spread from the level's standard error (fit residuals over
     sqrt(observations)) + horizon + refresh-lag uncertainty, so the bands
     widen over the 4.5-day final horizon and stay tight over a 90-minute
     practice horizon.

  5. Flat history (the practice cases: every read identical, no refresh yet
     observed): fall back to the published launch-night first refresh
     (RULES.txt: 638 -> 3,868 players, 943 -> 4,276 trophies earned).  The
     100% trophy stayed 100%, so the new players' non-100 earners are only
     4276 - 3868 - (943 - 638) = 103 over 3,230 new players: arrivals earn
     each other trophy at kappa ~= 0.067 of the tracked cohort's rate, and
     every non-100 rate falls to ~0.22x.  (The totals, not the rate sum,
     pin kappa: counting the 100% trophy's earners would give 0.70, which
     is wrong by 10x and scores ~1.12 instead of ~0.03 on the practice.)

The grader runs `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim: one case at a time, /input holding only <id>.in, 8 s a run,
two runs byte-identical.  MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in
for /input and /output outside the container.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release day begins 2026-11-19T00:00:00Z; log-time is measured from here.
T0 = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Published launch-night first refresh (RULES.txt): the fallback when a case's
# history is flat and no refresh has been observed yet.
FALLBACK_N_MULT = 3868.0 / 638.0
# New players at that refresh: 3868 - 638 = 3230.  New NON-100 earners:
# 4276 - 3868 (the 100% trophy's earners) - (943 - 638) (old non-100
# earners) = 103.  kappa is derived at run time from the case's own rates:
#   kappa = 103 / ((sum of non-100 last rates / 100) * 3230)
# which comes out ~= 0.067: the arrivals had earned almost nothing yet.
FALLBACK_NEW_PLAYERS = 3868.0 - 638.0
FALLBACK_NEW_NON100_EARNERS = 4276.0 - 3868.0 - (943.0 - 638.0)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(t):
    return max(0.0, (t - T0).total_seconds() / 3600.0)


def log_time(t):
    """log(1 + hours since release): the clock the trends are fitted on."""
    return math.log1p(hours_since_release(t))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as JavaScript's Math.round does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def least_squares(xs, ys):
    """Return (intercept, slope) of the least-squares line, or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return (my - slope * mx, slope)


def median(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def compress_history(case):
    """Collapse consecutive identical reads into refresh snapshots.

    Returns a list of dicts {t, n, earners, rates} in time order, one per
    distinct page state, plus the per-trophy forward-filled last rates.
    """
    trophies = case["trophies"]
    nt = len(trophies)
    snaps = []
    last_rates = [None] * nt
    for read in case["history"]:
        rates = list(read["rates"])
        for i, r in enumerate(rates):
            if r is not None:
                last_rates[i] = r
        filled = [last_rates[i] if last_rates[i] is not None else 0.0
                  for i in range(nt)]
        n = read["players_tracked"]
        key = (n, tuple(filled))
        if snaps and snaps[-1]["key"] == key:
            continue
        earners = [filled[i] * n / 100.0 for i in range(nt)]
        snaps.append({"t": parse_time(read["at"]), "n": n,
                      "earners": earners, "rates": filled, "key": key})
    return snaps, last_rates


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    nt = len(trophies)
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    snaps, last_rates = compress_history(case)
    last = snaps[-1]

    # Refresh cadence: median gap between distinct snapshots, in hours.
    gaps = [(snaps[j]["t"] - snaps[j - 1]["t"]).total_seconds() / 3600.0
            for j in range(1, len(snaps))]
    gap_h = median(gaps) if gaps else 0.0

    if len(snaps) >= 2 and gap_h > 0:
        medians, spreads = forecast_from_refreshes(
            case, snaps, target, horizon_days, gap_h)
    else:
        medians, spreads = forecast_flat_fallback(
            case, last, target, horizon_days)

    rows = []
    for i, trophy in enumerate(trophies):
        m = medians[i]
        s = spreads[i]
        if m >= 100.0:
            # A trophy every tracked player holds stays at 100%.
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        lg = logit(m)
        qs = []
        for z in Z:
            q = inv_logit(lg + z * s)
            qs.append(round4(min(100.0, max(0.0, q))))
        # Keep the quantiles non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def forecast_from_refreshes(case, snaps, target, horizon_days, gap_h):
    """Forecast from an observed refresh history (the final's regime).

    Fits N(t) as a saturating exponential and each trophy's arrivals earn
    rate u_i(t) with shrinkage toward its tier, then steps through the
    expected refreshes from the last snapshot to the expected last refresh
    before the target.
    """
    trophies = case["trophies"]
    nt = len(trophies)
    tiers = [t["tier"] for t in trophies]

    # --- players tracked N(t): saturating exponential, fitted by grid search
    # over the time constant.  A log-log fit overshoots badly here: tracked
    # players grow fast at launch and saturate (in a synthetic final the
    # log-log fit put N(target) at 3x the truth), which would inflate every
    # trophy's earners through the arrivals term below. ---
    def fit_n_saturating(snaps):
        hs = [hours_since_release(s["t"]) for s in snaps]
        ns = [float(s["n"]) for s in snaps]
        best = None
        for tau_h in (3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0):
            xs = [math.exp(-h / tau_h) for h in hs]
            # N = Ninf - A * x, linear least squares in (Ninf, A).
            fit = least_squares(xs, ns)
            if fit is None:
                continue
            # least_squares gives y = inter + slope * x: Ninf = inter, A = -slope.
            ninf, a = fit[0], -fit[1]
            if a < 0:
                continue  # accelerating growth: not a saturating fit
            sse = sum((n - (ninf - a * x)) ** 2 for x, n in zip(xs, ns))
            if best is None or sse < best[0]:
                best = (sse, ninf, a, tau_h)
        if best is None:
            return None
        _, ninf, a, tau_h = best
        return (max(ninf, ns[-1]), a, tau_h)

    n_fit = fit_n_saturating(snaps)
    if n_fit is not None:
        n_inf, n_a, n_tau = n_fit

        def n_at(t):
            v = n_inf - n_a * math.exp(-hours_since_release(t) / n_tau)
            return min(max(float(snaps[-1]["n"]), v),
                       5.0 * float(snaps[-1]["n"]))
    else:
        xs = [log_time(s["t"]) for s in snaps]
        ys = [math.log(max(1.0, s["n"])) for s in snaps]
        fit_n = least_squares(xs, ys)
        if fit_n is None:
            n_inter, n_slope = ys[-1], 0.0
        else:
            n_inter, n_slope = fit_n
            n_slope = max(0.0, n_slope)

        def n_at(t):
            v = math.exp(n_inter + n_slope * log_time(t))
            return min(max(float(snaps[-1]["n"]), v),
                       3.0 * float(snaps[-1]["n"]))

    # --- arrivals' earn rate u_i at each observed refresh. ---
    # u[j][i]: refresh j (j >= 1), trophy i.  None where the jump is too
    # small to read (quantisation noise dominates).
    u_obs = []
    for j in range(1, len(snaps)):
        dn = snaps[j]["n"] - snaps[j - 1]["n"]
        row = []
        for i in range(nt):
            if dn <= 0:
                row.append(None)
            else:
                u = (snaps[j]["earners"][i] - snaps[j - 1]["earners"][i]) / dn
                row.append(min(1.0, max(0.0, u)))
        u_obs.append(row)
    refresh_times = [snaps[j]["t"] for j in range(1, len(snaps))]

    # Tier-level fits of logit(u) on log-time (pooled over the tier's
    # trophies), for shrinkage and for trophies with too few observations.
    tier_fit = {}
    tier_level = {}
    for tier in sorted(set(tiers)):
        idxs = [i for i in range(nt) if tiers[i] == tier]
        tx, ty = [], []
        for j, t in enumerate(refresh_times):
            for i in idxs:
                u = u_obs[j][i]
                if u is None:
                    continue
                tx.append(log_time(t))
                ty.append(logit(100.0 * u))
        f = least_squares(tx, ty)
        tier_fit[tier] = f
        tier_level[tier] = (sum(ty) / len(ty)) if ty else logit(0.5)

    # Per-trophy level (shrunk toward the tier) and slope (tier's slope,
    # or 0 when the tier has no fit).
    u_level = []
    u_slope = []
    for i in range(nt):
        tier = tiers[i]
        vals = [logit(100.0 * u_obs[j][i]) for j in range(len(u_obs))
                if u_obs[j][i] is not None]
        k = 2.0  # shrinkage weight: about two refreshes' worth of evidence
        own = sum(vals) / len(vals) if vals else tier_level[tier]
        w = len(vals) / (len(vals) + k) if vals else 0.0
        u_level.append(w * own + (1.0 - w) * tier_level[tier])
        f = tier_fit.get(tier)
        u_slope.append(f[1] if f else 0.0)

    # Residual scale of the u fits, for the spread (shrunk toward 0.35).
    resid = []
    for j, t in enumerate(refresh_times):
        lt = log_time(t)
        for i in range(nt):
            u = u_obs[j][i]
            if u is None:
                continue
            pred = u_level[i] + u_slope[i] * (lt - log_time(refresh_times[-1]))
            resid.append(logit(100.0 * u) - pred)
    resid_sd = math.sqrt(sum(r * r for r in resid) / len(resid)) if resid else 0.35
    resid_sd = min(1.0, max(0.15, resid_sd))

    # Cap each trophy's extrapolated arrivals rate: it cannot plausibly run
    # far past the highest arrivals rate the history has shown (late trophies
    # do keep rising as players reach them, so allow 2.5x, not 1x).
    u_cap = []
    for i in range(nt):
        seen = [u_obs[j][i] for j in range(len(u_obs))
                if u_obs[j][i] is not None]
        if seen:
            u_cap.append(min(0.95, 2.5 * max(seen)))
        else:
            u_cap.append(min(0.95, 2.5 * inv_logit(u_level[i]) / 100.0))

    # --- step forward through the expected refreshes. ---
    # The answer is the page as of the last refresh before the target:
    # expect that refresh half a cadence before the target.
    t_eff_seconds = target.timestamp() - 0.5 * gap_h * 3600.0
    t_eff = datetime.fromtimestamp(t_eff_seconds, tz=timezone.utc)
    t_last = snaps[-1]["t"]
    if t_eff <= t_last:
        t_eff = t_last
    n_cur = float(snaps[-1]["n"])
    e_cur = list(snaps[-1]["earners"])
    t_cur = t_last
    step_seconds = gap_h * 3600.0
    while True:
        t_next = datetime.fromtimestamp(t_cur.timestamp() + step_seconds,
                                        tz=timezone.utc)
        if t_next > t_eff:
            break
        n_next = n_at(t_next)
        dn = max(0.0, n_next - n_cur)
        lt = log_time(t_next)
        lt_ref = log_time(refresh_times[-1])
        for i in range(nt):
            # The platinum-scale trophy (100% already) stays 100%.
            if snaps[-1]["rates"][i] >= 100.0:
                e_cur[i] = n_next  # every tracked player holds it
                continue
            u = inv_logit(u_level[i] + u_slope[i] * (lt - lt_ref)) / 100.0
            u = min(u, u_cap[i])
            e_cur[i] += u * dn
        n_cur = n_next
        t_cur = t_next

    medians = []
    spreads = []
    last_rates = snaps[-1]["rates"]
    for i in range(nt):
        if last_rates[i] >= 100.0:
            medians.append(100.0)
            spreads.append(0.02)
            continue
        r = 100.0 * e_cur[i] / n_cur if n_cur > 0 else last_rates[i]
        medians.append(min(99.99, max(0.01, r)))
        # Spread: the u-fit residuals are mostly measurement noise (rates
        # are printed to 2 decimals and each refresh's u is read off a small
        # arrivals count), so the level's uncertainty is the residual scale
        # divided by sqrt(observations) -- not the raw residual scale, which
        # would triple-count the noise.  Plus horizon growth and the drift
        # over half a refresh cadence near the target (the answer is the
        # last refresh before the target, not the target itself).
        n_u = sum(1 for j in range(len(u_obs))
                  if u_obs[j][i] is not None)
        s_level = resid_sd / math.sqrt(max(1, n_u))
        lag_drift = abs(u_slope[i]) * 0.5 * math.log1p(
            gap_h / (1.0 + hours_since_release(t_eff)))
        s = math.sqrt(0.10 ** 2 + (0.08 * math.sqrt(horizon_days)) ** 2
                      + lag_drift ** 2 + s_level ** 2)
        spreads.append(min(0.8, max(0.12, s)))
    return medians, spreads


def forecast_flat_fallback(case, last, target, horizon_days):
    """Flat history: no refresh observed yet (the practice cases).

    Assume one refresh before the target, of the published launch-night
    size: arrivals earn each trophy at kappa of the tracked cohort's rate,
    kappa derived from the published totals (see FALLBACK_* above) and the
    case's own non-100 rate sum.  The 100% trophy stays 100%.
    """
    nt = len(case["trophies"])
    n0 = float(last["n"])
    n1 = n0 * FALLBACK_N_MULT
    sum_non100 = sum(r for r in last["rates"] if r < 100.0)
    kappa = FALLBACK_NEW_NON100_EARNERS / ((sum_non100 / 100.0)
                                           * FALLBACK_NEW_PLAYERS)
    kappa = min(1.0, max(0.0, kappa))
    medians = []
    spreads = []
    for i in range(nt):
        r0 = last["rates"][i]
        if r0 >= 100.0:
            medians.append(100.0)
            spreads.append(0.02)
            continue
        e0 = r0 * n0 / 100.0
        e1 = e0 + kappa * (r0 / 100.0) * (n1 - n0)
        medians.append(min(99.99, max(0.01, 100.0 * e1 / n1)))
        # The mechanism is near-exact on the practice board, but keep a
        # modest spread for per-trophy variation in kappa.
        spreads.append(0.12)
    return medians, spreads


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            # Never crash a case: fall back to persistence with a wide band.
            last = case["history"][-1] if case["history"] else None
            rows = []
            for i, trophy in enumerate(case["trophies"]):
                rate = 0.01
                if last is not None:
                    r = last["rates"][i]
                    rate = 0.01 if r is None else r
                lg = logit(rate)
                spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0) + 0.3
                qs = [round4(min(100.0, max(0.0, inv_logit(lg + z * spread))))
                      for z in Z]
                for k in range(1, 5):
                    if qs[k] < qs[k - 1]:
                        qs[k] = qs[k - 1]
                rows.append({"id": trophy["id"], "q": qs})
            forecast = {"trophies": rows}
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
