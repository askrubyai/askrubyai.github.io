#!/usr/bin/env python3
"""
Day 20 PM: Volatility-conditioned quote distance under a time-to-fill framework.

Research question:
After Day 20 AM showed order-lifetime tuning is a second-order effect, does
adapting passive quote distance to volatility regime produce a larger OOS gain?

Method:
1) Keep the same funding-regime signal and yearly expanding walk-forward split.
2) Use minute-level first-touch logic (time-to-fill framework) with fixed
   order lifetime (5 minutes).
3) For each test year, fit quote-distance policy on prior years only:
   - static distance (single value)
   - dynamic distance map by sigma regime (low / mid / high)
4) Evaluate OOS returns with realistic maker-fill vs chase fallback mechanics.
"""

from __future__ import annotations

import datetime as dt
import itertools
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

# Signal definition (unchanged)
Z_THRESHOLD = 1.0
VOL_Q = 0.75

# Execution setup
LIFE_MIN = 5
QUOTE_CANDIDATES = [0.0003, 0.0005, 0.0006, 0.0008, 0.0010]  # 3/5/6/8/10 bps
BASELINE_QUOTE = 0.0006

FILL_COST_RT = 0.0007   # maker+taker
CHASE_COST_RT = 0.0010  # taker+taker fallback
TAKER_COST_RT = 0.0010

PRE_VOL_MIN = 60
POST_MIN = LIFE_MIN + 2

BOOT_ITERS = 5000
BOOT_SEED = 20260305 + 200
AVG_BLOCK_LEN = 5

POST_DIR = Path(__file__).resolve().parent
OUT_JSON = POST_DIR / "day20-pm-volquote-results.json"
OUT_EQ = POST_DIR / "day20-pm-volquote-equity.png"
OUT_BARS = POST_DIR / "day20-pm-volquote-bars.png"
OUT_TRADEOFF = POST_DIR / "day20-pm-volquote-tradeoff.png"


@dataclass
class SignalRec:
    t: int
    year: int
    z: float | None
    vol: float | None
    entry_8h: float
    exit_8h: float
    funding_next: float


@dataclass
class TradeObs:
    t: int
    year: int
    sigma_1m: float
    entry_px: float
    exit_px: float
    funding_next: float
    chase_open: float
    taker_ret: float
    fill_time_by_dist: dict[float, int | None]
    ret_by_dist: dict[float, float]
    brownian_pfill_by_dist: dict[float, float]


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


def load_rows_8h() -> list[tuple[int, float, float]]:
    """Return list of (t8, funding_rate, open_price)."""
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

    mark = {int(k[0]): float(k[1]) for k in mp_raw}

    rows = []
    for x in fr_raw:
        t = int(x["fundingTime"])
        fr = float(x["fundingRate"])
        t8 = t - (t % EIGHT_HOURS_MS)
        if t8 in mark:
            rows.append((t8, fr, mark[t8]))

    rows.sort(key=lambda r: r[0])

    # De-duplicate by timestamp
    dedup = []
    for t, fr, p in rows:
        if dedup and dedup[-1][0] == t:
            dedup[-1] = (t, fr, p)
        else:
            dedup.append((t, fr, p))

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


def stddev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return statistics.pstdev(xs)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def brownian_fill_probability(sigma_1m: float, quote_dist: float, life_min: int) -> float:
    if sigma_1m <= 1e-12 or life_min <= 0:
        return 0.0
    z = quote_dist / (sigma_1m * math.sqrt(2.0 * life_min))
    return clamp(math.erfc(z), 0.0, 1.0)


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

    # Pre-entry realized 1m sigma
    pre_times = sorted(ts for ts in bars if ts <= rec.t)
    pre_opens = [bars[ts]["open"] for ts in pre_times]
    if len(pre_opens) < PRE_VOL_MIN + 1:
        return None

    pre_slice = pre_opens[-(PRE_VOL_MIN + 1) :]
    lrs = [math.log(pre_slice[i] / pre_slice[i - 1]) for i in range(1, len(pre_slice)) if pre_slice[i - 1] > 0]
    sigma_1m = stddev(lrs)

    chase_ts = rec.t + LIFE_MIN * ONE_MIN_MS
    if chase_ts not in bars:
        return None
    chase_open = bars[chase_ts]["open"]

    fill_time_by_dist: dict[float, int | None] = {}
    ret_by_dist: dict[float, float] = {}
    pfill_brownian: dict[float, float] = {}

    for d in QUOTE_CANDIDATES:
        quote_px = entry_px * (1.0 - d)

        fill_time = None
        for m in range(1, LIFE_MIN + 1):
            bar_start = rec.t + (m - 1) * ONE_MIN_MS
            bar = bars.get(bar_start)
            if bar is None:
                break
            if bar["low"] <= quote_px:
                fill_time = m
                break

        fill_time_by_dist[d] = fill_time

        if fill_time is not None and fill_time <= LIFE_MIN:
            ret = rec.exit_8h / quote_px - 1.0 - rec.funding_next - FILL_COST_RT
        else:
            ret = rec.exit_8h / chase_open - 1.0 - rec.funding_next - CHASE_COST_RT
        ret_by_dist[d] = ret

        pfill_brownian[d] = brownian_fill_probability(sigma_1m, d, LIFE_MIN)

    taker_ret = rec.exit_8h / entry_px - 1.0 - rec.funding_next - TAKER_COST_RT

    return TradeObs(
        t=rec.t,
        year=rec.year,
        sigma_1m=sigma_1m,
        entry_px=entry_px,
        exit_px=rec.exit_8h,
        funding_next=rec.funding_next,
        chase_open=chase_open,
        taker_ret=taker_ret,
        fill_time_by_dist=fill_time_by_dist,
        ret_by_dist=ret_by_dist,
        brownian_pfill_by_dist=pfill_brownian,
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


def sigma_state(sigma: float, q1: float, q2: float) -> int:
    if sigma <= q1:
        return 0
    if sigma <= q2:
        return 1
    return 2


def evaluate_dynamic_policy(trades: list[TradeObs], q1: float, q2: float, mapping: tuple[float, float, float]) -> tuple[float, float, float, list[float]]:
    rets = []
    fills = []
    dists = []
    for tr in trades:
        st = sigma_state(tr.sigma_1m, q1, q2)
        d = mapping[st]
        rets.append(tr.ret_by_dist[d])
        fills.append(1.0 if (tr.fill_time_by_dist[d] is not None and tr.fill_time_by_dist[d] <= LIFE_MIN) else 0.0)
        dists.append(d)
    return mean(rets), mean(fills), mean(dists), rets


def evaluate_static_policy(trades: list[TradeObs], dist: float) -> tuple[float, float, list[float]]:
    rets = [tr.ret_by_dist[dist] for tr in trades]
    fills = [1.0 if (tr.fill_time_by_dist[dist] is not None and tr.fill_time_by_dist[dist] <= LIFE_MIN) else 0.0 for tr in trades]
    return mean(rets), mean(fills), rets


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

    scenarios = {
        "always_taker": {"label": "Always taker"},
        "maker_fixed_6bps": {"label": "Maker-first fixed 6 bps"},
        "maker_static_wf": {"label": "Maker-first static distance (WF tuned)"},
        "maker_dynamic_wf": {"label": "Maker-first volatility-conditioned distance (WF tuned)"},
    }

    ret_map: dict[str, list[float]] = {k: [] for k in scenarios}
    time_map: dict[str, list[int]] = {k: [] for k in scenarios}
    fill_map: dict[str, list[float]] = {k: [] for k in scenarios}
    dist_map: dict[str, list[float]] = {k: [] for k in scenarios}

    yearly = []

    for y in test_years:
        train = [tr for tr in enriched if tr.year < y]
        test = [tr for tr in enriched if tr.year == y]

        if not test:
            yearly.append({"year": y, "trades": 0})
            continue

        train_sigma = [tr.sigma_1m for tr in train]
        if len(train) < 20 or not train_sigma:
            q1 = quantile([tr.sigma_1m for tr in test], 1 / 3)
            q2 = quantile([tr.sigma_1m for tr in test], 2 / 3)
            selection_mode = "fallback_low_data"
        else:
            q1 = quantile(train_sigma, 1 / 3)
            q2 = quantile(train_sigma, 2 / 3)
            selection_mode = "argmax_train_mean"

        # Static quote-distance tuning
        static_board = []
        for d in QUOTE_CANDIDATES:
            avg_ret, avg_fill, _ = evaluate_static_policy(train, d)
            static_board.append({"dist": d, "train_avg": avg_ret, "train_avg_bp": avg_ret * 1e4, "train_fill": avg_fill})
        static_board.sort(key=lambda x: (x["train_avg"], -x["dist"]), reverse=True)
        static_best = static_board[0]["dist"] if static_board else BASELINE_QUOTE

        # Dynamic sigma-state tuning: brute force 5^3 maps.
        dynamic_board = []
        for mapping in itertools.product(QUOTE_CANDIDATES, repeat=3):
            avg_ret, avg_fill, avg_dist, _ = evaluate_dynamic_policy(train, q1, q2, mapping)
            dynamic_board.append(
                {
                    "mapping": mapping,
                    "train_avg": avg_ret,
                    "train_avg_bp": avg_ret * 1e4,
                    "train_fill": avg_fill,
                    "train_avg_dist": avg_dist,
                }
            )

        # tie-break: higher return, then higher avg dist (more maker edge if equal)
        dynamic_board.sort(key=lambda x: (x["train_avg"], x["train_avg_dist"]), reverse=True)
        dynamic_best = dynamic_board[0]["mapping"] if dynamic_board else (BASELINE_QUOTE, BASELINE_QUOTE, BASELINE_QUOTE)

        # Evaluate on test
        taker_rets = [tr.taker_ret for tr in test]
        fixed_rets = [tr.ret_by_dist[BASELINE_QUOTE] for tr in test]
        fixed_fill = [1.0 if (tr.fill_time_by_dist[BASELINE_QUOTE] is not None and tr.fill_time_by_dist[BASELINE_QUOTE] <= LIFE_MIN) else 0.0 for tr in test]

        _s_avg, s_fill, static_rets = evaluate_static_policy(test, static_best)
        d_avg, d_fill, d_avg_dist, dynamic_rets = evaluate_dynamic_policy(test, q1, q2, dynamic_best)

        # scenario accumulation
        ret_map["always_taker"].extend(taker_rets)
        time_map["always_taker"].extend([tr.t for tr in test])
        fill_map["always_taker"].extend([1.0] * len(test))
        dist_map["always_taker"].extend([0.0] * len(test))

        ret_map["maker_fixed_6bps"].extend(fixed_rets)
        time_map["maker_fixed_6bps"].extend([tr.t for tr in test])
        fill_map["maker_fixed_6bps"].extend(fixed_fill)
        dist_map["maker_fixed_6bps"].extend([BASELINE_QUOTE] * len(test))

        ret_map["maker_static_wf"].extend(static_rets)
        time_map["maker_static_wf"].extend([tr.t for tr in test])
        fill_map["maker_static_wf"].extend([
            1.0 if (tr.fill_time_by_dist[static_best] is not None and tr.fill_time_by_dist[static_best] <= LIFE_MIN) else 0.0
            for tr in test
        ])
        dist_map["maker_static_wf"].extend([static_best] * len(test))

        ret_map["maker_dynamic_wf"].extend(dynamic_rets)
        time_map["maker_dynamic_wf"].extend([tr.t for tr in test])
        fill_map["maker_dynamic_wf"].extend([
            1.0
            if (
                tr.fill_time_by_dist[
                    dynamic_best[sigma_state(tr.sigma_1m, q1, q2)]
                ]
                is not None
                and tr.fill_time_by_dist[
                    dynamic_best[sigma_state(tr.sigma_1m, q1, q2)]
                ]
                <= LIFE_MIN
            )
            else 0.0
            for tr in test
        ])
        dist_map["maker_dynamic_wf"].extend([
            dynamic_best[sigma_state(tr.sigma_1m, q1, q2)] for tr in test
        ])

        # Diagnostics: Brownian expected fill under selected policy on this year
        dyn_brownian_fill = mean([
            tr.brownian_pfill_by_dist[dynamic_best[sigma_state(tr.sigma_1m, q1, q2)]] for tr in test
        ])

        yearly.append(
            {
                "year": y,
                "trades": len(test),
                "selection_mode": selection_mode,
                "sigma_q1": q1,
                "sigma_q2": q2,
                "selected_static_dist": static_best,
                "selected_dynamic_mapping": {
                    "low_sigma": dynamic_best[0],
                    "mid_sigma": dynamic_best[1],
                    "high_sigma": dynamic_best[2],
                },
                "test_mean_bp": {
                    "always_taker": mean(taker_rets) * 1e4,
                    "maker_fixed_6bps": mean(fixed_rets) * 1e4,
                    "maker_static_wf": mean(static_rets) * 1e4,
                    "maker_dynamic_wf": d_avg * 1e4,
                },
                "test_fill_rate": {
                    "maker_fixed_6bps": mean(fixed_fill),
                    "maker_static_wf": s_fill,
                    "maker_dynamic_wf": d_fill,
                },
                "test_dynamic_avg_quote_bps": d_avg_dist * 1e4,
                "test_dynamic_brownian_expected_fill": dyn_brownian_fill,
            }
        )

    # Summaries
    scenario_stats = {}
    curves = {}

    for i, key in enumerate(scenarios):
        rets = ret_map[key]
        ts_raw = time_map[key]
        fills = fill_map[key]
        dists = dist_map[key]

        scenario_stats[key] = {
            "label": scenarios[key]["label"],
            "summary": summarize(rets),
            "bootstrap_stationary": stationary_bootstrap_mean_ci(
                rets,
                BOOT_ITERS,
                BOOT_SEED + i,
                avg_block_len=AVG_BLOCK_LEN,
            ),
            "execution_stats": {
                "empirical_fill_rate": mean(fills),
                "avg_quote_dist_bps": mean(dists) * 1e4,
            },
        }

        eq = 1.0
        xs = []
        ys = []
        for tt, rr in sorted(zip(ts_raw, rets), key=lambda x: x[0]):
            eq *= 1 + rr
            xs.append(dt.datetime.fromtimestamp(tt / 1000, dt.timezone.utc))
            ys.append(eq)
        curves[key] = {"ts": xs, "eq": ys}

    # Policy-independent trade-off diagnostics by global sigma terciles
    all_sigma = [tr.sigma_1m for tr in enriched]
    gq1 = quantile(all_sigma, 1 / 3) if all_sigma else 0.0
    gq2 = quantile(all_sigma, 2 / 3) if all_sigma else 0.0

    tradeoff = {"low": {}, "mid": {}, "high": {}}
    for d in QUOTE_CANDIDATES:
        for name, lo, hi in [
            ("low", -1.0, gq1),
            ("mid", gq1, gq2),
            ("high", gq2, float("inf")),
        ]:
            bucket = [tr for tr in enriched if (tr.sigma_1m <= hi and tr.sigma_1m > lo)]
            if not bucket:
                tradeoff[name][str(d)] = {"fill": 0.0, "avg_bp": 0.0}
                continue
            fill = mean([
                1.0 if (tr.fill_time_by_dist[d] is not None and tr.fill_time_by_dist[d] <= LIFE_MIN) else 0.0
                for tr in bucket
            ])
            avg_bp = mean([tr.ret_by_dist[d] for tr in bucket]) * 1e4
            tradeoff[name][str(d)] = {"fill": fill, "avg_bp": avg_bp}

    # ---- Plots ----

    # Equity curves
    plt.figure(figsize=(10.4, 5.3))
    for key in scenarios:
        ts = curves[key]["ts"]
        eq = curves[key]["eq"]
        if eq:
            plt.plot(ts, eq, label=scenarios[key]["label"])
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Day 20 PM: Volatility-conditioned quote distance (OOS)")
    plt.xlabel("UTC time")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_EQ, dpi=170)
    plt.close()

    # Bars: avg bps + fill rate
    order = ["always_taker", "maker_fixed_6bps", "maker_static_wf", "maker_dynamic_wf"]
    labels = [scenarios[k]["label"] for k in order]
    avg_bps = [scenario_stats[k]["summary"]["avg_bp"] for k in order]
    fill_pct = [scenario_stats[k]["execution_stats"]["empirical_fill_rate"] * 100 for k in order]

    x = list(range(len(labels)))
    w = 0.38

    plt.figure(figsize=(11.2, 5.2))
    plt.bar([i - w / 2 for i in x], avg_bps, width=w, label="Avg return (bps/trade)")
    plt.bar([i + w / 2 for i in x], fill_pct, width=w, label="Empirical fill rate (%)")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.xticks(x, labels, rotation=10)
    plt.title("Expectancy vs fill trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_BARS, dpi=170)
    plt.close()

    # Trade-off by volatility state
    d_bps = [d * 1e4 for d in QUOTE_CANDIDATES]
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.8))

    for label, color in [("low", "tab:blue"), ("mid", "tab:orange"), ("high", "tab:green")]:
        fill_line = [tradeoff[label][str(d)]["fill"] * 100 for d in QUOTE_CANDIDATES]
        ret_line = [tradeoff[label][str(d)]["avg_bp"] for d in QUOTE_CANDIDATES]
        ax[0].plot(d_bps, fill_line, marker="o", label=label, color=color)
        ax[1].plot(d_bps, ret_line, marker="o", label=label, color=color)

    ax[0].set_title("Fill-rate vs quote distance")
    ax[0].set_xlabel("Quote distance (bps)")
    ax[0].set_ylabel("Fill rate within 5m (%)")

    ax[1].set_title("Return vs quote distance")
    ax[1].set_xlabel("Quote distance (bps)")
    ax[1].set_ylabel("Avg return (bps/trade)")

    ax[0].legend(title="Sigma state")
    ax[1].legend(title="Sigma state")
    plt.tight_layout()
    plt.savefig(OUT_TRADEOFF, dpi=170)
    plt.close()

    out = {
        "meta": {
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sample_start_utc": dt.datetime.fromtimestamp(selected[0].t / 1000, dt.timezone.utc).isoformat() if selected else None,
            "sample_end_utc": dt.datetime.fromtimestamp(selected[-1].t / 1000, dt.timezone.utc).isoformat() if selected else None,
            "signal": {"z_threshold": Z_THRESHOLD, "vol_q": VOL_Q},
            "walkforward_oos_years": test_years,
            "life_min": LIFE_MIN,
            "quote_candidates": QUOTE_CANDIDATES,
            "baseline_quote": BASELINE_QUOTE,
            "costs": {
                "fill_cost_rt": FILL_COST_RT,
                "chase_cost_rt": CHASE_COST_RT,
                "taker_cost_rt": TAKER_COST_RT,
            },
            "bootstrap_iters": BOOT_ITERS,
            "avg_block_len": AVG_BLOCK_LEN,
            "brownian_formula": "P(tau<=L)=erfc(delta/(sigma*sqrt(2L)))",
            "dynamic_policy": "Map sigma regime (low/mid/high) -> quote distance; tune yearly on prior data only.",
        },
        "counts": {
            "rows_8h_total": rows_total,
            "signals_oos_selected": len(selected),
            "minute_enriched": len(enriched),
            "dropped_missing_minute_data": dropped,
            "yearly_trade_counts": yearly_counts,
        },
        "walkforward_selection": yearly,
        "tradeoff_by_sigma_state": tradeoff,
        "scenarios": scenario_stats,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
