#!/usr/bin/env python3
"""
Day 20 AM: Time-to-fill distribution modeling for maker-first execution.

Goal:
Move from a single expected fill probability proxy to an explicit time-to-fill
view, then test whether walk-forward tuning of order lifetime improves maker-
first outcomes on the existing BTC funding-regime signal.

Key components:
1) Build the same yearly OOS signal sample as Day 16-19.
2) Pull 1-minute mark-price windows around each signal timestamp.
3) Estimate empirical fill-time distribution for a passive buy quote.
4) Compare two fill-probability models:
   - Legacy sqrt-time range proxy (Day 18)
   - Brownian first-passage approximation
5) Run a walk-forward backtest that chooses order lifetime L (minutes)
   using only prior years, then applies to test year.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import requests

BASE_FR = "https://fapi.binance.com/fapi/v1/fundingRate"
BASE_MP = "https://fapi.binance.com/fapi/v1/markPriceKlines"
EIGHT_HOURS_MS = 8 * 60 * 60 * 1000
ONE_MIN_MS = 60 * 1000
START_UTC = dt.datetime(2022, 1, 1, tzinfo=dt.timezone.utc)

# Signal baseline (unchanged)
Z_THRESHOLD = 1.0
VOL_Q = 0.75

# Execution settings
QUOTE_DIST = 0.0006  # 6 bps passive buy quote below entry reference
QUEUE_PRIORITY = 0.55
LIFE_CANDIDATES = [1, 3, 5, 10, 15, 20, 30]
DEFAULT_LIFE = 5
MAX_LIFE = max(LIFE_CANDIDATES)

# Round-trip cost assumptions (conservative)
FILL_COST_RT = 0.0007   # maker+taker
CHASE_COST_RT = 0.0010  # taker+taker fallback
TAKER_COST_RT = 0.0010

# Minute-window extraction
PRE_VOL_MIN = 60
POST_MIN = MAX_LIFE + 2

# Bootstrap settings
BOOT_ITERS = 5000
BOOT_SEED = 20260305
AVG_BLOCK_LEN = 5

POST_DIR = Path(__file__).resolve().parent
OUT_JSON = POST_DIR / "day20-am-time-to-fill-results.json"
OUT_CDF = POST_DIR / "day20-am-filltime-cdf.png"
OUT_EQ = POST_DIR / "day20-am-lifetime-equity.png"
OUT_BRIER = POST_DIR / "day20-am-fill-calibration.png"


@dataclass
class SignalRec:
    t: int
    year: int
    z: float | None
    vol: float | None
    entry_8h: float
    exit_8h: float
    funding_next: float
    next_range: float


@dataclass
class TradeObs:
    t: int
    year: int
    entry_px: float
    exit_px: float
    funding_next: float
    next_range: float
    sigma_1m: float
    fill_time_min: int | None
    chase_open_by_life: dict[int, float]
    ret_by_life: dict[int, float]
    taker_ret: float
    brownian_pfill_by_life: dict[int, float]
    legacy_pfill_by_life: dict[int, float]


def fetch_paginated(url: str, params: dict, max_limit: int) -> list:
    out = []
    cursor = params["startTime"]
    end_time = params["endTime"]

    while True:
        p = dict(params)
        p["startTime"] = cursor
        p["limit"] = max_limit

        data = requests.get(url, params=p, timeout=30).json()
        if not data or isinstance(data, dict):
            break

        out.extend(data)
        last = int(data[-1][0]) if isinstance(data[-1], list) else int(data[-1]["fundingTime"])
        if len(data) < max_limit or last >= end_time:
            break
        cursor = last + 1
        time.sleep(0.03)

    return out


def load_rows_8h() -> list[tuple[int, float, float, float]]:
    """Return list of (t8, funding_rate, open_price, bar_range)."""
    start_ms = int(START_UTC.timestamp() * 1000)
    end_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)

    fr_raw = fetch_paginated(
        BASE_FR,
        {"symbol": "BTCUSDT", "startTime": start_ms, "endTime": end_ms},
        max_limit=1000,
    )
    mp_raw = fetch_paginated(
        BASE_MP,
        {"symbol": "BTCUSDT", "interval": "8h", "startTime": start_ms, "endTime": end_ms},
        max_limit=1500,
    )

    mark = {}
    for k in mp_raw:
        t = int(k[0])
        o = float(k[1])
        h = float(k[2])
        l = float(k[3])
        rng = (h - l) / o if o > 0 else 0.0
        mark[t] = (o, rng)

    rows = []
    for x in fr_raw:
        t = int(x["fundingTime"])
        fr = float(x["fundingRate"])
        t8 = t - (t % EIGHT_HOURS_MS)
        if t8 in mark:
            o, rng = mark[t8]
            rows.append((t8, fr, o, rng))

    rows.sort(key=lambda r: r[0])

    # De-duplicate by timestamp (keep latest)
    dedup = []
    for t, fr, p, rng in rows:
        if dedup and dedup[-1][0] == t:
            dedup[-1] = (t, fr, p, rng)
        else:
            dedup.append((t, fr, p, rng))

    return dedup


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty quantile input")
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    w = pos - lo
    return s[lo] * (1 - w) + s[hi] * w


def rolling_zscore(values: list[float], lookback: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    for i in range(lookback, len(values)):
        w = values[i - lookback : i]
        m = sum(w) / lookback
        var = sum((x - m) ** 2 for x in w) / lookback
        sd = math.sqrt(var)
        if sd > 1e-12:
            out[i] = (values[i] - m) / sd
    return out


def rolling_vol(prices: list[float], lookback: int) -> list[float | None]:
    out: list[float | None] = [None] * len(prices)
    lr = [None] + [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
    for i in range(lookback + 1, len(prices)):
        w = lr[i - lookback : i]
        m = sum(w) / len(w)
        var = sum((x - m) ** 2 for x in w) / len(w)
        out[i] = math.sqrt(var)
    return out


def build_signal_recs() -> tuple[list[SignalRec], list[dict], list[int], int]:
    rows = load_rows_8h()

    t = [r[0] for r in rows]
    fr = [r[1] for r in rows]
    p = [r[2] for r in rows]
    bar_rng = [r[3] for r in rows]

    z = rolling_zscore(fr, lookback=90)
    vol = rolling_vol(p, lookback=21)

    recs: list[SignalRec] = []
    for i in range(90, len(rows) - 1):
        year = dt.datetime.fromtimestamp(t[i] / 1000, dt.timezone.utc).year
        recs.append(
            SignalRec(
                t=t[i],
                year=year,
                z=z[i],
                vol=vol[i],
                entry_8h=p[i],
                exit_8h=p[i + 1],
                funding_next=fr[i + 1],
                next_range=bar_rng[i + 1],
            )
        )

    years = sorted({r.year for r in recs})
    test_years = years[1:]

    selected: list[SignalRec] = []
    yearly_counts = []

    for y in test_years:
        train = [r for r in recs if r.year < y and r.vol is not None]
        test = [r for r in recs if r.year == y]

        vol_vals = [r.vol for r in train if r.vol is not None]
        if not vol_vals:
            yearly_counts.append({"year": y, "trades": 0, "vol_threshold": None})
            continue

        v_thr = quantile(vol_vals, VOL_Q)
        before = len(selected)
        for r in test:
            if r.z is None or r.vol is None:
                continue
            if r.z < -Z_THRESHOLD and r.vol > v_thr:
                selected.append(r)

        yearly_counts.append(
            {
                "year": y,
                "trades": len(selected) - before,
                "vol_threshold": v_thr,
            }
        )

    return selected, yearly_counts, test_years, len(rows)


def fetch_minute_window(start_ms: int, end_ms: int) -> list[list]:
    return fetch_paginated(
        BASE_MP,
        {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "startTime": start_ms,
            "endTime": end_ms,
        },
        max_limit=1500,
    )


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def legacy_fill_probability(next_range: float, life_min: int) -> float:
    # Day 18 proxy
    scale = math.sqrt(life_min / (8 * 60))
    micro_excursion = 0.5 * next_range * scale
    touch_prob = clamp(micro_excursion / QUOTE_DIST, 0.0, 1.0)
    return clamp(touch_prob * QUEUE_PRIORITY, 0.0, 1.0)


def brownian_fill_probability(sigma_1m: float, life_min: int) -> float:
    # Driftless Brownian first-passage approximation:
    # P(tau <= L) = erfc(a / (sigma * sqrt(2L))), barrier a > 0 in log-price units.
    if sigma_1m <= 1e-12 or life_min <= 0:
        return 0.0
    a = QUOTE_DIST
    z = a / (sigma_1m * math.sqrt(2.0 * life_min))
    return clamp(math.erfc(z), 0.0, 1.0)


def stddev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return statistics.pstdev(xs)


def enrich_trade(rec: SignalRec) -> TradeObs | None:
    start_ms = rec.t - PRE_VOL_MIN * ONE_MIN_MS
    end_ms = rec.t + POST_MIN * ONE_MIN_MS
    raw = fetch_minute_window(start_ms, end_ms)
    time.sleep(0.02)

    bars = {}
    for k in raw:
        ts = int(k[0])
        bars[ts] = {
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
        }

    if rec.t not in bars:
        return None

    entry_px = bars[rec.t]["open"]
    quote_px = entry_px * (1.0 - QUOTE_DIST)

    # Pre-entry 1-minute volatility from log returns over last PRE_VOL_MIN minutes.
    pre_times = sorted(ts for ts in bars if ts <= rec.t)
    pre_opens = [bars[ts]["open"] for ts in pre_times]
    if len(pre_opens) < PRE_VOL_MIN + 1:
        return None

    pre_slice = pre_opens[-(PRE_VOL_MIN + 1) :]
    lrs = [math.log(pre_slice[i] / pre_slice[i - 1]) for i in range(1, len(pre_slice)) if pre_slice[i - 1] > 0]
    sigma_1m = stddev(lrs)

    # First-touch time to passive quote.
    fill_time_min = None
    for m in range(1, MAX_LIFE + 1):
        bar_start = rec.t + (m - 1) * ONE_MIN_MS
        bar = bars.get(bar_start)
        if bar is None:
            break
        if bar["low"] <= quote_px:
            fill_time_min = m
            break

    chase_open_by_life: dict[int, float] = {}
    for l in LIFE_CANDIDATES:
        ts = rec.t + l * ONE_MIN_MS
        if ts not in bars:
            return None
        chase_open_by_life[l] = bars[ts]["open"]

    # Returns by lifetime policy.
    ret_by_life = {}
    r_fill = rec.exit_8h / quote_px - 1.0 - rec.funding_next - FILL_COST_RT

    for l in LIFE_CANDIDATES:
        if fill_time_min is not None and fill_time_min <= l:
            ret = r_fill
        else:
            chase_entry = chase_open_by_life[l]
            r_chase = rec.exit_8h / chase_entry - 1.0 - rec.funding_next - CHASE_COST_RT
            ret = r_chase
        ret_by_life[l] = ret

    taker_ret = rec.exit_8h / entry_px - 1.0 - rec.funding_next - TAKER_COST_RT

    brownian_p = {l: brownian_fill_probability(sigma_1m, l) for l in LIFE_CANDIDATES}
    legacy_p = {l: legacy_fill_probability(rec.next_range, l) for l in LIFE_CANDIDATES}

    return TradeObs(
        t=rec.t,
        year=rec.year,
        entry_px=entry_px,
        exit_px=rec.exit_8h,
        funding_next=rec.funding_next,
        next_range=rec.next_range,
        sigma_1m=sigma_1m,
        fill_time_min=fill_time_min,
        chase_open_by_life=chase_open_by_life,
        ret_by_life=ret_by_life,
        taker_ret=taker_ret,
        brownian_pfill_by_life=brownian_p,
        legacy_pfill_by_life=legacy_p,
    )


def summarize(xs: list[float]) -> dict:
    if not xs:
        return {
            "n": 0,
            "avg": 0.0,
            "avg_bp": 0.0,
            "win_rate": 0.0,
            "equity": 1.0,
            "median_bp": 0.0,
            "p10_bp": 0.0,
            "p90_bp": 0.0,
        }

    eq = 1.0
    wins = 0
    for r in xs:
        eq *= 1 + r
        if r > 0:
            wins += 1

    s = sorted(xs)
    return {
        "n": len(xs),
        "avg": mean(xs),
        "avg_bp": mean(xs) * 1e4,
        "win_rate": wins / len(xs),
        "equity": eq,
        "median_bp": quantile(s, 0.5) * 1e4,
        "p10_bp": quantile(s, 0.1) * 1e4,
        "p90_bp": quantile(s, 0.9) * 1e4,
    }


def stationary_bootstrap_mean_ci(xs: list[float], iters: int, seed: int, avg_block_len: float) -> dict:
    if not xs:
        return {
            "mean": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "p_gt_0": 0.0,
            "avg_block_len": avg_block_len,
            "restart_p": 0.0,
        }

    rng = random.Random(seed)
    n = len(xs)
    p = 1.0 / avg_block_len
    means = []

    for _ in range(iters):
        idx = rng.randrange(n)
        s = 0.0
        for j in range(n):
            if j > 0:
                if rng.random() < p:
                    idx = rng.randrange(n)
                else:
                    idx = (idx + 1) % n
            s += xs[idx]
        means.append(s / n)

    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    p_gt_0 = sum(1 for m in means if m > 0) / len(means)

    return {
        "mean": mean(xs),
        "ci95_low": lo,
        "ci95_high": hi,
        "p_gt_0": p_gt_0,
        "avg_block_len": avg_block_len,
        "restart_p": p,
    }


def compute_lifetime_returns(trades: list[TradeObs], life: int) -> list[float]:
    out = []
    for tr in trades:
        out.append(tr.ret_by_life[life])
    return out


def run() -> dict:
    selected, yearly_counts, test_years, rows_total = build_signal_recs()

    enriched: list[TradeObs] = []
    dropped = 0
    for rec in selected:
        e = enrich_trade(rec)
        if e is None:
            dropped += 1
        else:
            enriched.append(e)

    enriched.sort(key=lambda x: x.t)

    # Fill-time distribution diagnostics
    fill_times = [tr.fill_time_min for tr in enriched if tr.fill_time_min is not None]

    cdf_by_min = []
    hazard_by_min = []
    for m in range(1, MAX_LIFE + 1):
        cdf = mean([1.0 if (tr.fill_time_min is not None and tr.fill_time_min <= m) else 0.0 for tr in enriched])
        cdf_by_min.append({"minute": m, "cdf": cdf})

        at_risk = sum(1 for tr in enriched if tr.fill_time_min is None or tr.fill_time_min >= m)
        events = sum(1 for tr in enriched if tr.fill_time_min == m)
        hazard = (events / at_risk) if at_risk > 0 else 0.0
        hazard_by_min.append({"minute": m, "events": events, "at_risk": at_risk, "hazard": hazard})

    # Calibration at candidate lifetimes
    calibration = {}
    for l in LIFE_CANDIDATES:
        y = [1.0 if (tr.fill_time_min is not None and tr.fill_time_min <= l) else 0.0 for tr in enriched]
        p_brownian = [tr.brownian_pfill_by_life[l] for tr in enriched]
        p_legacy = [tr.legacy_pfill_by_life[l] for tr in enriched]

        brier_b = mean([(pb - yy) ** 2 for pb, yy in zip(p_brownian, y)])
        brier_l = mean([(pl - yy) ** 2 for pl, yy in zip(p_legacy, y)])

        calibration[l] = {
            "empirical_fill_rate": mean(y),
            "brownian_mean_p": mean(p_brownian),
            "legacy_mean_p": mean(p_legacy),
            "brownian_brier": brier_b,
            "legacy_brier": brier_l,
        }

    # Walk-forward order-lifetime selection
    wf_rets = []
    wf_times = []
    fixed5_rets = []
    fixed5_times = []
    taker_rets = []
    taker_times = []

    wf_yearly = []

    for y in test_years:
        train = [tr for tr in enriched if tr.year < y]
        test = [tr for tr in enriched if tr.year == y]

        if not test:
            wf_yearly.append({"year": y, "trades": 0, "selected_life": None})
            continue

        if len(train) < 20:
            selected_life = DEFAULT_LIFE
            mode = "fallback_low_data"
        else:
            leaderboard = []
            for l in LIFE_CANDIDATES:
                r = mean([tr.ret_by_life[l] for tr in train])
                leaderboard.append((l, r))
            leaderboard.sort(key=lambda x: (x[1], -x[0]), reverse=True)
            selected_life = leaderboard[0][0]
            mode = "argmax_train_mean_return"

        test_life_rets = [tr.ret_by_life[selected_life] for tr in test]
        test_fixed_rets = [tr.ret_by_life[DEFAULT_LIFE] for tr in test]
        test_taker_rets = [tr.taker_ret for tr in test]

        wf_rets.extend(test_life_rets)
        wf_times.extend([tr.t for tr in test])

        fixed5_rets.extend(test_fixed_rets)
        fixed5_times.extend([tr.t for tr in test])

        taker_rets.extend(test_taker_rets)
        taker_times.extend([tr.t for tr in test])

        # For diagnostics only
        test_best = max((mean([tr.ret_by_life[l] for tr in test]), l) for l in LIFE_CANDIDATES)

        wf_yearly.append(
            {
                "year": y,
                "trades": len(test),
                "selected_life": selected_life,
                "selection_mode": mode,
                "train_means_bp": {str(l): mean([tr.ret_by_life[l] for tr in train]) * 1e4 if train else 0.0 for l in LIFE_CANDIDATES},
                "test_mean_bp_selected": mean(test_life_rets) * 1e4,
                "test_mean_bp_fixed5": mean(test_fixed_rets) * 1e4,
                "test_mean_bp_taker": mean(test_taker_rets) * 1e4,
                "test_best_life_oracle": test_best[1],
                "test_best_life_oracle_bp": test_best[0] * 1e4,
            }
        )

    strategies = {
        "always_taker": {"label": "Always taker"},
        "maker_fixed_5m": {"label": "Maker-first (fixed 5m life)"},
        "maker_wf_life": {"label": "Maker-first (WF tuned life)"},
    }

    ret_map = {
        "always_taker": taker_rets,
        "maker_fixed_5m": fixed5_rets,
        "maker_wf_life": wf_rets,
    }
    time_map = {
        "always_taker": taker_times,
        "maker_fixed_5m": fixed5_times,
        "maker_wf_life": wf_times,
    }

    scenario_stats = {}
    curves = {}
    for i, k in enumerate(strategies):
        rets = ret_map[k]
        ts_raw = time_map[k]

        scenario_stats[k] = {
            "label": strategies[k]["label"],
            "summary": summarize(rets),
            "bootstrap_stationary": stationary_bootstrap_mean_ci(
                rets,
                BOOT_ITERS,
                BOOT_SEED + i,
                avg_block_len=AVG_BLOCK_LEN,
            ),
        }

        eq = 1.0
        xs = []
        ys = []
        for tt, rr in sorted(zip(ts_raw, rets), key=lambda x: x[0]):
            eq *= 1 + rr
            xs.append(dt.datetime.fromtimestamp(tt / 1000, dt.timezone.utc))
            ys.append(eq)
        curves[k] = {"ts": xs, "eq": ys}

    # ---- Plots ----

    # Fill CDF + hazard
    mins = [x["minute"] for x in cdf_by_min]
    emp_cdf = [x["cdf"] for x in cdf_by_min]
    brownian_cdf = [mean([brownian_fill_probability(tr.sigma_1m, m) for tr in enriched]) for m in mins]
    legacy_cdf = [mean([legacy_fill_probability(tr.next_range, m) for tr in enriched]) for m in mins]
    hazard = [x["hazard"] for x in hazard_by_min]

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.6))
    ax[0].plot(mins, emp_cdf, label="Empirical CDF")
    ax[0].plot(mins, brownian_cdf, label="Brownian first-passage")
    ax[0].plot(mins, legacy_cdf, label="Legacy sqrt-time proxy")
    ax[0].set_title("Passive-quote fill CDF")
    ax[0].set_xlabel("Order lifetime (minutes)")
    ax[0].set_ylabel("P(fill by L)")
    ax[0].set_ylim(0, 1.0)
    ax[0].legend()

    ax[1].bar(mins, hazard)
    ax[1].set_title("Minute-level fill hazard")
    ax[1].set_xlabel("Minute")
    ax[1].set_ylabel("Hazard")

    plt.tight_layout()
    plt.savefig(OUT_CDF, dpi=170)
    plt.close()

    # Equity curves
    plt.figure(figsize=(10.2, 5.1))
    for k in strategies:
        ts = curves[k]["ts"]
        eq = curves[k]["eq"]
        if eq:
            plt.plot(ts, eq, label=strategies[k]["label"])
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Day 20 AM: Lifetime-aware maker-first execution (OOS)")
    plt.xlabel("UTC time")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_EQ, dpi=170)
    plt.close()

    # Calibration/Brier
    lvals = LIFE_CANDIDATES
    emp = [calibration[l]["empirical_fill_rate"] * 100 for l in lvals]
    pb = [calibration[l]["brownian_mean_p"] * 100 for l in lvals]
    pl = [calibration[l]["legacy_mean_p"] * 100 for l in lvals]
    bb = [calibration[l]["brownian_brier"] for l in lvals]
    bl = [calibration[l]["legacy_brier"] for l in lvals]

    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.6))
    ax[0].plot(lvals, emp, marker="o", label="Empirical")
    ax[0].plot(lvals, pb, marker="o", label="Brownian")
    ax[0].plot(lvals, pl, marker="o", label="Legacy")
    ax[0].set_title("Fill-rate calibration by lifetime")
    ax[0].set_xlabel("L (minutes)")
    ax[0].set_ylabel("Fill rate (%)")
    ax[0].legend()

    ax[1].plot(lvals, bb, marker="o", label="Brownian Brier")
    ax[1].plot(lvals, bl, marker="o", label="Legacy Brier")
    ax[1].set_title("Probability error (lower is better)")
    ax[1].set_xlabel("L (minutes)")
    ax[1].set_ylabel("Brier score")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(OUT_BRIER, dpi=170)
    plt.close()

    out = {
        "meta": {
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sample_start_utc": dt.datetime.fromtimestamp(selected[0].t / 1000, dt.timezone.utc).isoformat() if selected else None,
            "sample_end_utc": dt.datetime.fromtimestamp(selected[-1].t / 1000, dt.timezone.utc).isoformat() if selected else None,
            "signal": {"z_threshold": Z_THRESHOLD, "vol_q": VOL_Q},
            "walkforward_oos_years": test_years,
            "life_candidates_min": LIFE_CANDIDATES,
            "default_life_min": DEFAULT_LIFE,
            "quote_dist": QUOTE_DIST,
            "costs": {
                "fill_cost_rt": FILL_COST_RT,
                "chase_cost_rt": CHASE_COST_RT,
                "taker_cost_rt": TAKER_COST_RT,
            },
            "bootstrap_iters": BOOT_ITERS,
            "avg_block_len": AVG_BLOCK_LEN,
            "brownian_formula": "P(tau<=L)=erfc(delta/(sigma*sqrt(2L)))",
            "legacy_formula": "touch_prob=min(1, 0.5*range*sqrt(L/480)/delta); p_fill=touch_prob*queue_priority",
        },
        "counts": {
            "rows_8h_total": rows_total,
            "signals_oos_selected": len(selected),
            "minute_enriched": len(enriched),
            "dropped_missing_minute_data": dropped,
            "yearly_trade_counts": yearly_counts,
            "fill_observed_within_30m": sum(1 for tr in enriched if tr.fill_time_min is not None),
        },
        "fill_time_distribution": {
            "cdf_by_min": cdf_by_min,
            "hazard_by_min": hazard_by_min,
            "median_fill_min": quantile(fill_times, 0.5) if fill_times else None,
            "p90_fill_min": quantile(fill_times, 0.9) if fill_times else None,
        },
        "calibration_by_life": {str(k): v for k, v in calibration.items()},
        "walkforward_lifetime_selection": wf_yearly,
        "scenarios": scenario_stats,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
