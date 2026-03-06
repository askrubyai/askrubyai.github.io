#!/usr/bin/env python3
"""
Day 21 PM: Multi-feature toxicity routing for maker/taker execution.

Question
--------
Can a walk-forward, multi-feature toxicity score route maker vs taker better than
(1) always-maker and (2) single-feature mom5 gating?

Protocol
--------
- Same BTC funding-regime signal and expanding yearly OOS split as Day 16-21.
- Same execution stack as Day 20/21 baseline:
    quote distance = 6 bps, order lifetime = 5 min
    maker+taker cost = 7 bps RT, fallback taker+taker = 10 bps RT
- Build minute-level trade observations.
- Compare policies:
    1) always taker
    2) always maker-first (fixed baseline)
    3) walk-forward mom5 threshold gating (Day 21 AM control)
    4) walk-forward multi-feature score gating (new)
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

# Signal definition (unchanged)
Z_THRESHOLD = 1.0
VOL_Q = 0.75

# Execution constants
QUOTE_DIST = 0.0006  # 6 bps
LIFE_MIN = 5

FILL_COST_RT = 0.0007
CHASE_COST_RT = 0.0010
TAKER_COST_RT = 0.0010

# Feature windows
PRE_MIN = 90
POST_MIN = 20

BOOT_ITERS = 5000
BOOT_SEED = 20260306 + 21
AVG_BLOCK_LEN = 5

POST_DIR = Path(__file__).resolve().parent
OUT_JSON = POST_DIR / "day21-pm-multifeature-results.json"
OUT_EQ = POST_DIR / "day21-pm-multifeature-equity.png"
OUT_BARS = POST_DIR / "day21-pm-multifeature-bars.png"
OUT_SCORE = POST_DIR / "day21-pm-score-decile-edge.png"

FEATURE_COLS = [
    "mom1",
    "mom5",
    "mom15",
    "mom30",
    "sigma1m_15",
    "sigma1m_30",
    "range15",
    "range30",
]


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
    entry_px: float
    exit_px: float
    funding_next: float
    mom1: float
    mom5: float
    mom15: float
    mom30: float
    sigma1m_15: float
    sigma1m_30: float
    range15: float
    range30: float
    fill_time_min: int | None
    maker_ret: float
    taker_ret: float
    maker_minus_taker_bp: float


def fetch_paginated(url: str, params: dict, max_limit: int) -> list:
    out = []
    cursor = params["startTime"]
    end_time = params["endTime"]

    while True:
        p = dict(params)
        p["startTime"] = cursor
        p["limit"] = max_limit

        resp = requests.get(url, params=p, timeout=30)
        data = resp.json()
        if not data or isinstance(data, dict):
            break

        out.extend(data)
        last = int(data[-1][0]) if isinstance(data[-1], list) else int(data[-1]["fundingTime"])
        if len(data) < max_limit or last >= end_time:
            break

        cursor = last + 1
        time.sleep(0.03)

    return out


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


def stddev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return statistics.pstdev(xs)


def load_rows_8h() -> list[tuple[int, float, float]]:
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

    dedup = []
    for t, fr, p in rows:
        if dedup and dedup[-1][0] == t:
            dedup[-1] = (t, fr, p)
        else:
            dedup.append((t, fr, p))

    return dedup


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


def _range_frac(bars: dict[int, dict], anchor_t: int, window: int, entry_px: float) -> float:
    times = [anchor_t - i * ONE_MIN_MS for i in range(1, window + 1)]
    highs = [bars[t]["high"] for t in times]
    lows = [bars[t]["low"] for t in times]
    return (max(highs) - min(lows)) / entry_px if entry_px > 0 else 0.0


def enrich_trade(rec: SignalRec) -> TradeObs | None:
    start_ms = rec.t - PRE_MIN * ONE_MIN_MS
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

    need_past = [rec.t - i * ONE_MIN_MS for i in range(1, PRE_MIN + 1)]
    if any(ts not in bars for ts in need_past):
        return None

    need_future = [rec.t + i * ONE_MIN_MS for i in range(0, LIFE_MIN + 1)]
    if any(ts not in bars for ts in need_future):
        return None

    entry_px = bars[rec.t]["open"]

    mom1 = entry_px / bars[rec.t - 1 * ONE_MIN_MS]["open"] - 1.0
    mom5 = entry_px / bars[rec.t - 5 * ONE_MIN_MS]["open"] - 1.0
    mom15 = entry_px / bars[rec.t - 15 * ONE_MIN_MS]["open"] - 1.0
    mom30 = entry_px / bars[rec.t - 30 * ONE_MIN_MS]["open"] - 1.0

    lrs15 = []
    for i in range(1, 16):
        p0 = bars[rec.t - i * ONE_MIN_MS]["open"]
        p1 = bars[rec.t - (i - 1) * ONE_MIN_MS]["open"]
        lrs15.append(math.log(p1 / p0))

    lrs30 = []
    for i in range(1, 31):
        p0 = bars[rec.t - i * ONE_MIN_MS]["open"]
        p1 = bars[rec.t - (i - 1) * ONE_MIN_MS]["open"]
        lrs30.append(math.log(p1 / p0))

    sigma1m_15 = stddev(lrs15)
    sigma1m_30 = stddev(lrs30)
    range15 = _range_frac(bars, rec.t, 15, entry_px)
    range30 = _range_frac(bars, rec.t, 30, entry_px)

    quote_px = entry_px * (1.0 - QUOTE_DIST)

    fill_time_min = None
    for m in range(1, LIFE_MIN + 1):
        bar = bars.get(rec.t + (m - 1) * ONE_MIN_MS)
        if bar is None:
            break
        if bar["low"] <= quote_px:
            fill_time_min = m
            break

    if fill_time_min is not None and fill_time_min <= LIFE_MIN:
        maker_ret = rec.exit_8h / quote_px - 1.0 - rec.funding_next - FILL_COST_RT
    else:
        chase_open = bars[rec.t + LIFE_MIN * ONE_MIN_MS]["open"]
        maker_ret = rec.exit_8h / chase_open - 1.0 - rec.funding_next - CHASE_COST_RT

    taker_ret = rec.exit_8h / entry_px - 1.0 - rec.funding_next - TAKER_COST_RT

    return TradeObs(
        t=rec.t,
        year=rec.year,
        entry_px=entry_px,
        exit_px=rec.exit_8h,
        funding_next=rec.funding_next,
        mom1=mom1,
        mom5=mom5,
        mom15=mom15,
        mom30=mom30,
        sigma1m_15=sigma1m_15,
        sigma1m_30=sigma1m_30,
        range15=range15,
        range30=range30,
        fill_time_min=fill_time_min,
        maker_ret=maker_ret,
        taker_ret=taker_ret,
        maker_minus_taker_bp=(maker_ret - taker_ret) * 1e4,
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


def apply_threshold_policy(trades: list[TradeObs], theta: float) -> tuple[list[float], list[int]]:
    rets = []
    actions = []  # 1 maker, 0 taker
    for tr in trades:
        use_maker = tr.mom5 >= theta
        actions.append(1 if use_maker else 0)
        rets.append(tr.maker_ret if use_maker else tr.taker_ret)
    return rets, actions


def build_bin_model(train: list[TradeObs], feature: str, bins: int = 5, prior_strength: float = 5.0) -> dict:
    vals = [getattr(tr, feature) for tr in train]
    if not vals:
        return {"feature": feature, "edges": [0.0, 1.0], "bin_mean_bp": [0.0], "bin_count": [0]}

    qs = [i / bins for i in range(bins + 1)]
    edges = [quantile(vals, q) for q in qs]

    # Ensure strictly non-decreasing + small spread fallback
    for i in range(1, len(edges)):
        if edges[i] < edges[i - 1]:
            edges[i] = edges[i - 1]

    global_mean = mean([tr.maker_minus_taker_bp for tr in train])
    sums = [0.0] * bins
    counts = [0] * bins

    for tr in train:
        x = getattr(tr, feature)
        b = bin_index(x, edges)
        sums[b] += tr.maker_minus_taker_bp
        counts[b] += 1

    means = []
    for b in range(bins):
        m = (sums[b] + prior_strength * global_mean) / (counts[b] + prior_strength)
        means.append(m)

    return {
        "feature": feature,
        "edges": edges,
        "bin_mean_bp": means,
        "bin_count": counts,
    }


def bin_index(x: float, edges: list[float]) -> int:
    bins = max(1, len(edges) - 1)
    if bins == 1:
        return 0
    if x <= edges[0]:
        return 0
    if x >= edges[-1]:
        return bins - 1
    for i in range(1, len(edges)):
        if x <= edges[i]:
            return i - 1
    return bins - 1


def score_trade(tr: TradeObs, models: dict[str, dict]) -> float:
    vals = []
    for f, m in models.items():
        b = bin_index(getattr(tr, f), m["edges"])
        vals.append(m["bin_mean_bp"][b])
    return mean(vals)


def apply_score_policy(trades: list[TradeObs], models: dict[str, dict], theta_bp: float) -> tuple[list[float], list[int], list[float]]:
    rets = []
    actions = []
    scores = []
    for tr in trades:
        s = score_trade(tr, models)
        use_maker = s >= theta_bp
        actions.append(1 if use_maker else 0)
        rets.append(tr.maker_ret if use_maker else tr.taker_ret)
        scores.append(s)
    return rets, actions, scores


def run() -> dict:
    selected, yearly_counts, test_years, rows_total = build_signal_recs()

    enriched: list[TradeObs] = []
    dropped = 0
    for rec in selected:
        tr = enrich_trade(rec)
        if tr is None:
            dropped += 1
        else:
            enriched.append(tr)

    enriched.sort(key=lambda x: x.t)

    scenarios = {
        "always_taker": {"label": "Always taker"},
        "maker_fixed": {"label": "Maker-first fixed (6 bps, 5m)"},
        "maker_mom5_threshold_wf": {"label": "Maker gated by mom5 threshold (WF)"},
        "maker_multifeature_score_wf": {"label": "Maker gated by multi-feature toxicity score (WF)"},
    }

    ret_map: dict[str, list[float]] = {k: [] for k in scenarios}
    time_map: dict[str, list[int]] = {k: [] for k in scenarios}
    maker_usage_map: dict[str, list[int]] = {k: [] for k in scenarios}
    score_trace_test: list[dict] = []
    yearly = []

    for y in test_years:
        train = [tr for tr in enriched if tr.year < y]
        test = [tr for tr in enriched if tr.year == y]

        if not test:
            yearly.append({"year": y, "trades": 0})
            continue

        taker_rets = [tr.taker_ret for tr in test]
        maker_rets = [tr.maker_ret for tr in test]

        # --- Control: mom5 threshold WF ---
        if len(train) < 20:
            theta_mom5 = -1e9
            mom5_mode = "fallback_low_data"
        else:
            train_mom = [tr.mom5 for tr in train]
            candidates = sorted(set([quantile(train_mom, q) for q in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]] + [-1e9, 1e9]))
            board = []
            for th in candidates:
                rr, aa = apply_threshold_policy(train, th)
                board.append(
                    {
                        "theta": th,
                        "train_avg": mean(rr),
                        "maker_usage": mean([float(a) for a in aa]),
                    }
                )
            board.sort(key=lambda z: (z["train_avg"], z["maker_usage"]), reverse=True)
            theta_mom5 = board[0]["theta"]
            mom5_mode = "argmax_train_mean"

        mom5_test_rets, mom5_test_actions = apply_threshold_policy(test, theta_mom5)

        # --- New: multi-feature score WF ---
        if len(train) < 25:
            feat_models = {f: {"feature": f, "edges": [0.0, 1.0], "bin_mean_bp": [0.0], "bin_count": [0]} for f in FEATURE_COLS}
            theta_score_bp = -1e9
            score_mode = "fallback_low_data"
            train_scores = [0.0 for _ in train]
        else:
            feat_models = {f: build_bin_model(train, f, bins=5, prior_strength=5.0) for f in FEATURE_COLS}
            train_scores = [score_trade(tr, feat_models) for tr in train]
            cands = sorted(set([quantile(train_scores, q) for q in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]] + [-1e9, 1e9]))

            board = []
            for th in cands:
                rr, aa, _ = apply_score_policy(train, feat_models, th)
                board.append(
                    {
                        "theta_score_bp": th,
                        "train_avg": mean(rr),
                        "maker_usage": mean([float(a) for a in aa]),
                    }
                )
            board.sort(key=lambda z: (z["train_avg"], z["maker_usage"]), reverse=True)
            theta_score_bp = board[0]["theta_score_bp"]
            score_mode = "argmax_train_mean"

        score_test_rets, score_test_actions, score_test_vals = apply_score_policy(test, feat_models, theta_score_bp)

        if score_mode != "fallback_low_data":
            for tr, sv in zip(test, score_test_vals):
                score_trace_test.append(
                    {
                        "t": tr.t,
                        "year": tr.year,
                        "score_bp": sv,
                        "maker_minus_taker_bp": tr.maker_minus_taker_bp,
                    }
                )

        # accumulate
        ret_map["always_taker"].extend(taker_rets)
        time_map["always_taker"].extend([tr.t for tr in test])
        maker_usage_map["always_taker"].extend([0] * len(test))

        ret_map["maker_fixed"].extend(maker_rets)
        time_map["maker_fixed"].extend([tr.t for tr in test])
        maker_usage_map["maker_fixed"].extend([1] * len(test))

        ret_map["maker_mom5_threshold_wf"].extend(mom5_test_rets)
        time_map["maker_mom5_threshold_wf"].extend([tr.t for tr in test])
        maker_usage_map["maker_mom5_threshold_wf"].extend(mom5_test_actions)

        ret_map["maker_multifeature_score_wf"].extend(score_test_rets)
        time_map["maker_multifeature_score_wf"].extend([tr.t for tr in test])
        maker_usage_map["maker_multifeature_score_wf"].extend(score_test_actions)

        # feature diagnostics (spread of bin means)
        feat_spread = {}
        for f in FEATURE_COLS:
            bm = feat_models[f]["bin_mean_bp"]
            feat_spread[f] = {
                "min_bin_mean_bp": min(bm),
                "max_bin_mean_bp": max(bm),
                "spread_bp": max(bm) - min(bm),
            }

        yearly.append(
            {
                "year": y,
                "trades": len(test),
                "mom5_threshold_policy": {
                    "selection_mode": mom5_mode,
                    "theta": theta_mom5,
                    "theta_bp": theta_mom5 * 1e4,
                    "test_maker_usage": mean([float(a) for a in mom5_test_actions]),
                    "test_mean_bp": mean(mom5_test_rets) * 1e4,
                },
                "multifeature_policy": {
                    "selection_mode": score_mode,
                    "theta_score_bp": theta_score_bp,
                    "test_maker_usage": mean([float(a) for a in score_test_actions]),
                    "test_mean_bp": mean(score_test_rets) * 1e4,
                    "train_score_p10_bp": quantile(train_scores, 0.10) if train_scores else 0.0,
                    "train_score_p50_bp": quantile(train_scores, 0.50) if train_scores else 0.0,
                    "train_score_p90_bp": quantile(train_scores, 0.90) if train_scores else 0.0,
                },
                "test_baselines_mean_bp": {
                    "always_taker": mean(taker_rets) * 1e4,
                    "maker_fixed": mean(maker_rets) * 1e4,
                },
                "feature_bin_spread_bp": feat_spread,
            }
        )

    scenario_stats = {}
    curves = {}

    for i, key in enumerate(scenarios):
        rets = ret_map[key]
        ts_raw = time_map[key]
        maker_usage = maker_usage_map[key]

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
                "maker_usage_rate": mean([float(x) for x in maker_usage]),
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

    # Score calibration deciles (test points only)
    score_vals = [x["score_bp"] for x in score_trace_test]
    score_deciles = []
    if score_vals:
        edges = [quantile(score_vals, q / 10) for q in range(11)]
        for d in range(10):
            lo = edges[d]
            hi = edges[d + 1]
            if d == 0:
                bucket = [x for x in score_trace_test if lo <= x["score_bp"] <= hi]
            else:
                bucket = [x for x in score_trace_test if lo < x["score_bp"] <= hi]
            score_deciles.append(
                {
                    "decile": d + 1,
                    "score_lo_bp": lo,
                    "score_hi_bp": hi,
                    "n": len(bucket),
                    "realized_maker_minus_taker_bp": mean([x["maker_minus_taker_bp"] for x in bucket]) if bucket else 0.0,
                }
            )

    # Global diagnostics
    filled_all = [tr for tr in enriched if tr.fill_time_min is not None and tr.fill_time_min <= LIFE_MIN]
    global_diag = {
        "maker_fill_rate": mean([1.0 if tr.fill_time_min is not None and tr.fill_time_min <= LIFE_MIN else 0.0 for tr in enriched]),
        "maker_minus_taker_avg_bp": mean([tr.maker_minus_taker_bp for tr in enriched]),
        "n_filled": len(filled_all),
    }

    # --- Plots ---
    plt.figure(figsize=(10.8, 5.3))
    for key in scenarios:
        ts = curves[key]["ts"]
        eq = curves[key]["eq"]
        if eq:
            plt.plot(ts, eq, label=scenarios[key]["label"])
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Day 21 PM: Multi-feature toxicity routing (OOS)")
    plt.xlabel("UTC time")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_EQ, dpi=170)
    plt.close()

    order = ["always_taker", "maker_fixed", "maker_mom5_threshold_wf", "maker_multifeature_score_wf"]
    labels = [scenarios[k]["label"] for k in order]
    avg_bps = [scenario_stats[k]["summary"]["avg_bp"] for k in order]
    maker_usage = [scenario_stats[k]["execution_stats"]["maker_usage_rate"] * 100 for k in order]

    x = list(range(len(labels)))
    w = 0.38

    plt.figure(figsize=(11.8, 5.3))
    plt.bar([i - w / 2 for i in x], avg_bps, width=w, label="Avg return (bps/trade)")
    plt.bar([i + w / 2 for i in x], maker_usage, width=w, label="Maker usage (%)")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.xticks(x, labels, rotation=8)
    plt.title("Expectancy vs maker exposure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_BARS, dpi=170)
    plt.close()

    if score_deciles:
        plt.figure(figsize=(10.6, 4.8))
        plt.bar([str(x["decile"]) for x in score_deciles], [x["realized_maker_minus_taker_bp"] for x in score_deciles])
        plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        plt.title("Realized maker-minus-taker by toxicity-score decile (test years)")
        plt.xlabel("Score decile (1 = most toxic / lowest score)")
        plt.ylabel("Maker - Taker (bps/trade)")
        plt.tight_layout()
        plt.savefig(OUT_SCORE, dpi=170)
        plt.close()

    out = {
        "meta": {
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sample_start_utc": dt.datetime.fromtimestamp(selected[0].t / 1000, dt.timezone.utc).isoformat() if selected else None,
            "sample_end_utc": dt.datetime.fromtimestamp(selected[-1].t / 1000, dt.timezone.utc).isoformat() if selected else None,
            "signal": {"z_threshold": Z_THRESHOLD, "vol_q": VOL_Q},
            "walkforward_oos_years": test_years,
            "execution": {
                "quote_dist": QUOTE_DIST,
                "life_min": LIFE_MIN,
                "fill_cost_rt": FILL_COST_RT,
                "chase_cost_rt": CHASE_COST_RT,
                "taker_cost_rt": TAKER_COST_RT,
            },
            "features": FEATURE_COLS,
            "bootstrap_iters": BOOT_ITERS,
            "avg_block_len": AVG_BLOCK_LEN,
        },
        "counts": {
            "rows_8h_total": rows_total,
            "signals_oos_selected": len(selected),
            "minute_enriched": len(enriched),
            "dropped_missing_minute_data": dropped,
            "yearly_trade_counts": yearly_counts,
        },
        "global_diagnostics": global_diag,
        "score_decile_calibration": score_deciles,
        "walkforward_selection": yearly,
        "scenarios": scenario_stats,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
