#!/usr/bin/env python3
"""
Day 21 AM: Adverse-selection-aware maker gating.

Research question
-----------------
Can we improve maker-first execution by *conditionally* using passive orders only
in states where maker edge is positive versus immediate taker entry?

Method
------
- Keep the exact BTC funding-regime signal + expanding yearly OOS splits from Day 16-20.
- Keep execution constants from Day 20: quote distance = 6 bps, order lifetime = 5m.
- Build minute-level trade observations around each signal.
- Compare four policies:
  1) Always taker
  2) Always maker-first (6 bps quote, 5m life)
  3) Walk-forward momentum-threshold gating (maker iff mom5 >= theta)
  4) Walk-forward momentum-state gating (maker/taker per mom5 tercile)
- Evaluate OOS expectancy and stationary-bootstrap confidence intervals.
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

# Execution constants (same baseline stack)
QUOTE_DIST = 0.0006  # 6 bps
LIFE_MIN = 5

FILL_COST_RT = 0.0007   # maker+taker
CHASE_COST_RT = 0.0010  # taker+taker fallback after timeout
TAKER_COST_RT = 0.0010

# Feature windows
PRE_MIN = 60
POST_MIN = 20

BOOT_ITERS = 5000
BOOT_SEED = 20260306
AVG_BLOCK_LEN = 5

POST_DIR = Path(__file__).resolve().parent
OUT_JSON = POST_DIR / "day21-am-adverse-gating-results.json"
OUT_EQ = POST_DIR / "day21-am-adverse-equity.png"
OUT_BARS = POST_DIR / "day21-am-adverse-bars.png"
OUT_DELTA = POST_DIR / "day21-am-maker-minus-taker-by-mom-decile.png"


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
    mom5: float
    mom15: float
    sigma1m_15: float
    fill_time_min: int | None
    maker_ret: float
    taker_ret: float
    maker_minus_taker_bp: float
    post_fill_drift_3m: float | None


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

    entry_px = bars[rec.t]["open"]

    # Need enough pre-entry bars for features.
    pre_times = sorted(ts for ts in bars if ts <= rec.t)
    pre_opens = [bars[ts]["open"] for ts in pre_times]
    if len(pre_opens) < PRE_MIN + 1:
        return None

    # Need all post bars for maker fallback and post-fill drift estimate.
    need_times = [rec.t + m * ONE_MIN_MS for m in range(0, POST_MIN + 1)]
    if any(ts not in bars for ts in need_times[: LIFE_MIN + 6]):
        return None

    mom5 = entry_px / bars[rec.t - 5 * ONE_MIN_MS]["open"] - 1.0
    mom15 = entry_px / bars[rec.t - 15 * ONE_MIN_MS]["open"] - 1.0

    # 1-minute realized volatility over last 15 minutes.
    lrs = []
    for i in range(1, 16):
        p0 = bars[rec.t - i * ONE_MIN_MS]["open"]
        p1 = bars[rec.t - (i - 1) * ONE_MIN_MS]["open"]
        if p0 > 0:
            lrs.append(math.log(p1 / p0))
    sigma1m_15 = stddev(lrs)

    # Maker-first mechanics.
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

    # Short-horizon post-fill drift proxy (adverse selection diagnostic).
    post_fill_drift_3m = None
    if fill_time_min is not None:
        drift_ts = rec.t + min(fill_time_min + 3, POST_MIN) * ONE_MIN_MS
        if drift_ts in bars:
            post_fill_drift_3m = bars[drift_ts]["open"] / quote_px - 1.0

    return TradeObs(
        t=rec.t,
        year=rec.year,
        entry_px=entry_px,
        exit_px=rec.exit_8h,
        funding_next=rec.funding_next,
        mom5=mom5,
        mom15=mom15,
        sigma1m_15=sigma1m_15,
        fill_time_min=fill_time_min,
        maker_ret=maker_ret,
        taker_ret=taker_ret,
        maker_minus_taker_bp=(maker_ret - taker_ret) * 1e4,
        post_fill_drift_3m=post_fill_drift_3m,
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


def mom_state(x: float, q1: float, q2: float) -> int:
    if x <= q1:
        return 0
    if x <= q2:
        return 1
    return 2


def choose_state_actions(train: list[TradeObs], q1: float, q2: float) -> dict[int, int]:
    # action: 1 maker, 0 taker
    out = {}
    for s in [0, 1, 2]:
        bucket = [tr for tr in train if mom_state(tr.mom5, q1, q2) == s]
        if not bucket:
            out[s] = 1
            continue
        maker_mu = mean([tr.maker_ret for tr in bucket])
        taker_mu = mean([tr.taker_ret for tr in bucket])
        out[s] = 1 if maker_mu >= taker_mu else 0
    return out


def apply_state_policy(trades: list[TradeObs], q1: float, q2: float, actions: dict[int, int]) -> tuple[list[float], list[int]]:
    rets = []
    acts = []
    for tr in trades:
        s = mom_state(tr.mom5, q1, q2)
        use_maker = actions.get(s, 1) == 1
        acts.append(1 if use_maker else 0)
        rets.append(tr.maker_ret if use_maker else tr.taker_ret)
    return rets, acts


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
        "maker_mom_threshold_wf": {"label": "Maker gated by mom5 threshold (WF)"},
        "maker_mom_state_wf": {"label": "Maker gated by mom5 state map (WF)"},
    }

    ret_map: dict[str, list[float]] = {k: [] for k in scenarios}
    time_map: dict[str, list[int]] = {k: [] for k in scenarios}
    maker_usage_map: dict[str, list[int]] = {k: [] for k in scenarios}

    yearly = []

    for y in test_years:
        train = [tr for tr in enriched if tr.year < y]
        test = [tr for tr in enriched if tr.year == y]

        if not test:
            yearly.append({"year": y, "trades": 0})
            continue

        # Baselines in test year
        taker_rets = [tr.taker_ret for tr in test]
        maker_rets = [tr.maker_ret for tr in test]

        # --- WF threshold policy on mom5 ---
        if len(train) < 20:
            theta = -1e9  # fallback => always maker
            threshold_mode = "fallback_low_data"
        else:
            train_mom = [tr.mom5 for tr in train]
            candidate_q = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
            candidates = sorted(set([quantile(train_mom, q) for q in candidate_q] + [-1e9, 1e9]))

            board = []
            for th in candidates:
                tr_rets, tr_actions = apply_threshold_policy(train, th)
                board.append(
                    {
                        "theta": th,
                        "train_avg": mean(tr_rets),
                        "train_avg_bp": mean(tr_rets) * 1e4,
                        "maker_usage": mean([float(a) for a in tr_actions]),
                    }
                )

            board.sort(key=lambda z: (z["train_avg"], z["maker_usage"]), reverse=True)
            theta = board[0]["theta"]
            threshold_mode = "argmax_train_mean"

        thr_test_rets, thr_test_actions = apply_threshold_policy(test, theta)

        # --- WF state-map policy on mom5 terciles ---
        if len(train) < 20:
            q1 = quantile([tr.mom5 for tr in test], 1 / 3)
            q2 = quantile([tr.mom5 for tr in test], 2 / 3)
            actions = {0: 1, 1: 1, 2: 1}
            state_mode = "fallback_low_data"
        else:
            train_mom = [tr.mom5 for tr in train]
            q1 = quantile(train_mom, 1 / 3)
            q2 = quantile(train_mom, 2 / 3)
            actions = choose_state_actions(train, q1, q2)
            state_mode = "argmax_per_state_train_mean"

        state_test_rets, state_test_actions = apply_state_policy(test, q1, q2, actions)

        # Accumulate scenarios
        ret_map["always_taker"].extend(taker_rets)
        time_map["always_taker"].extend([tr.t for tr in test])
        maker_usage_map["always_taker"].extend([0] * len(test))

        ret_map["maker_fixed"].extend(maker_rets)
        time_map["maker_fixed"].extend([tr.t for tr in test])
        maker_usage_map["maker_fixed"].extend([1] * len(test))

        ret_map["maker_mom_threshold_wf"].extend(thr_test_rets)
        time_map["maker_mom_threshold_wf"].extend([tr.t for tr in test])
        maker_usage_map["maker_mom_threshold_wf"].extend(thr_test_actions)

        ret_map["maker_mom_state_wf"].extend(state_test_rets)
        time_map["maker_mom_state_wf"].extend([tr.t for tr in test])
        maker_usage_map["maker_mom_state_wf"].extend(state_test_actions)

        # Diagnostics for this test year
        filled = [tr for tr in test if tr.fill_time_min is not None and tr.fill_time_min <= LIFE_MIN]
        avg_postfill_3m_bp = mean([tr.post_fill_drift_3m for tr in filled if tr.post_fill_drift_3m is not None]) * 1e4 if filled else 0.0

        yearly.append(
            {
                "year": y,
                "trades": len(test),
                "threshold_policy": {
                    "selection_mode": threshold_mode,
                    "theta": theta,
                    "theta_bp": theta * 1e4,
                    "test_maker_usage": mean([float(a) for a in thr_test_actions]),
                    "test_mean_bp": mean(thr_test_rets) * 1e4,
                },
                "state_policy": {
                    "selection_mode": state_mode,
                    "q1": q1,
                    "q2": q2,
                    "actions": {
                        "low_mom": "maker" if actions.get(0, 1) == 1 else "taker",
                        "mid_mom": "maker" if actions.get(1, 1) == 1 else "taker",
                        "high_mom": "maker" if actions.get(2, 1) == 1 else "taker",
                    },
                    "test_maker_usage": mean([float(a) for a in state_test_actions]),
                    "test_mean_bp": mean(state_test_rets) * 1e4,
                },
                "test_baselines_mean_bp": {
                    "always_taker": mean(taker_rets) * 1e4,
                    "maker_fixed": mean(maker_rets) * 1e4,
                },
                "test_adverse_diag": {
                    "fill_rate": mean([
                        1.0 if (tr.fill_time_min is not None and tr.fill_time_min <= LIFE_MIN) else 0.0
                        for tr in test
                    ]),
                    "avg_post_fill_drift_3m_bp": avg_postfill_3m_bp,
                },
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

    # Additional diagnostics: maker-minus-taker delta by mom5 decile.
    mom_vals = [tr.mom5 for tr in enriched]
    decile_edges = [quantile(mom_vals, q / 10) for q in range(0, 11)] if mom_vals else [0.0] * 11

    decile_stats = []
    for d in range(10):
        lo = decile_edges[d]
        hi = decile_edges[d + 1]
        if d == 0:
            bucket = [tr for tr in enriched if lo <= tr.mom5 <= hi]
        else:
            bucket = [tr for tr in enriched if lo < tr.mom5 <= hi]

        decile_stats.append(
            {
                "decile": d + 1,
                "mom5_lo": lo,
                "mom5_hi": hi,
                "n": len(bucket),
                "maker_minus_taker_bp": mean([tr.maker_minus_taker_bp for tr in bucket]) if bucket else 0.0,
                "maker_bp": mean([tr.maker_ret for tr in bucket]) * 1e4 if bucket else 0.0,
                "taker_bp": mean([tr.taker_ret for tr in bucket]) * 1e4 if bucket else 0.0,
            }
        )

    # Global adverse-selection diagnostics.
    filled_all = [tr for tr in enriched if tr.fill_time_min is not None and tr.fill_time_min <= LIFE_MIN]
    global_diag = {
        "maker_fill_rate": mean([
            1.0 if (tr.fill_time_min is not None and tr.fill_time_min <= LIFE_MIN) else 0.0
            for tr in enriched
        ]),
        "avg_post_fill_drift_3m_bp": mean([
            tr.post_fill_drift_3m for tr in filled_all if tr.post_fill_drift_3m is not None
        ]) * 1e4 if filled_all else 0.0,
        "maker_minus_taker_avg_bp": mean([tr.maker_minus_taker_bp for tr in enriched]),
    }

    # ---- Plots ----
    plt.figure(figsize=(10.6, 5.3))
    for key in scenarios:
        ts = curves[key]["ts"]
        eq = curves[key]["eq"]
        if eq:
            plt.plot(ts, eq, label=scenarios[key]["label"])
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Day 21 AM: Adverse-selection-aware maker gating (OOS)")
    plt.xlabel("UTC time")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_EQ, dpi=170)
    plt.close()

    order = ["always_taker", "maker_fixed", "maker_mom_threshold_wf", "maker_mom_state_wf"]
    labels = [scenarios[k]["label"] for k in order]
    avg_bps = [scenario_stats[k]["summary"]["avg_bp"] for k in order]
    maker_usage = [scenario_stats[k]["execution_stats"]["maker_usage_rate"] * 100 for k in order]

    x = list(range(len(labels)))
    w = 0.38

    plt.figure(figsize=(11.4, 5.3))
    plt.bar([i - w / 2 for i in x], avg_bps, width=w, label="Avg return (bps/trade)")
    plt.bar([i + w / 2 for i in x], maker_usage, width=w, label="Maker usage (%)")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.xticks(x, labels, rotation=10)
    plt.title("Expectancy vs maker exposure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_BARS, dpi=170)
    plt.close()

    plt.figure(figsize=(10.8, 4.8))
    plt.bar([str(x["decile"]) for x in decile_stats], [x["maker_minus_taker_bp"] for x in decile_stats])
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Maker minus taker edge by pre-entry 5m momentum decile")
    plt.xlabel("mom5 decile (1 = most negative momentum)")
    plt.ylabel("Maker - Taker (bps/trade)")
    plt.tight_layout()
    plt.savefig(OUT_DELTA, dpi=170)
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
            "feature": "mom5 = P_t / P_{t-5m} - 1",
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
        "global_adverse_diagnostics": global_diag,
        "maker_minus_taker_by_mom_decile": decile_stats,
        "walkforward_selection": yearly,
        "scenarios": scenario_stats,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
