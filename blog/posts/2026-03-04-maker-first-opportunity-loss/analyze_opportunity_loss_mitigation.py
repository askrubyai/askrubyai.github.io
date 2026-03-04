#!/usr/bin/env python3
"""
Day 19 PM: Maker-first opportunity-loss mitigation.

Question:
Can we improve Day 18's maker-first/chase expectancy by only chasing unfilled
orders on stronger signals (selected out-of-sample via yearly walk-forward)?

Method summary:
- Same BTC funding-regime signal and yearly OOS walk-forward protocol.
- Same maker-first queue model as Day 18.
- New mitigation: for each test year, choose a strength quantile threshold
  using only prior-year data; chase unfilled orders only when signal strength
  exceeds that threshold. Otherwise, skip the unfilled branch.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import requests

BASE_FR = "https://fapi.binance.com/fapi/v1/fundingRate"
BASE_MP = "https://fapi.binance.com/fapi/v1/markPriceKlines"
EIGHT_HOURS_MS = 8 * 60 * 60 * 1000
START_UTC = dt.datetime(2022, 1, 1, tzinfo=dt.timezone.utc)

Z_THRESHOLD = 1.0
VOL_Q = 0.75

BOOT_ITERS = 5000
BOOT_SEED = 20260304
AVG_BLOCK_LEN = 5

# Baseline maker-first assumptions carried from Day 18.
MAKER_CFG = {
    "quote_dist": 0.0006,  # 6 bps from touch
    "queue_priority": 0.55,
    "order_life_min": 5,
    "fill_cost_rt": 0.0007,   # maker+taker
    "fill_latency_k": 0.03,
    "unfilled_capture": 0.60,
    "chase_cost_rt": 0.0010,  # taker+taker fallback
    "chase_latency_k": 0.10,
}

TAKER_CFG = {
    "cost_rt": 0.0010,
    "latency_k": 0.05,
}

# Candidate quantiles for yearly chase gating.
Q_CANDIDATES = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

POST_DIR = Path(__file__).resolve().parent
OUT_JSON = POST_DIR / "day19-pm-opportunity-loss-results.json"
OUT_CURVE = POST_DIR / "day19-pm-opportunity-loss-equity.png"
OUT_BAR = POST_DIR / "day19-pm-opportunity-loss-bars.png"


@dataclass
class Rec:
    t: int
    year: int
    z: float | None
    vol: float | None
    gross: float
    next_range: float


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


def load_rows() -> list[tuple[int, float, float, float]]:
    """Returns list of (t8, funding_rate, open_price, next_bar_range)."""
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

    def q(v: float) -> float:
        return quantile(s, v)

    return {
        "n": len(xs),
        "avg": mean(xs),
        "avg_bp": mean(xs) * 1e4,
        "win_rate": wins / len(xs),
        "equity": eq,
        "median_bp": q(0.5) * 1e4,
        "p10_bp": q(0.1) * 1e4,
        "p90_bp": q(0.9) * 1e4,
    }


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def maker_fill_probability(next_range: float, quote_dist: float, queue_priority: float, order_life_min: float) -> float:
    # 8h bars overstate micro fill opportunities. Approximate order-lifetime
    # one-sided excursion by sqrt-time scaling from 8h to order_life_min.
    scale = math.sqrt(order_life_min / (8 * 60))
    micro_excursion = 0.5 * next_range * scale
    touch_prob = clamp(micro_excursion / quote_dist, 0.0, 1.0)
    return clamp(touch_prob * queue_priority, 0.0, 1.0)


def maker_first_components(gross: float, rng: float, cfg: dict) -> tuple[float, float, float]:
    """
    Returns (r_fill, r_chase, p_fill).
    - r_fill: realized return if maker fills
    - r_chase: realized return if unfilled but chased
    - p_fill: expected maker fill probability
    """
    p_fill = maker_fill_probability(
        next_range=rng,
        quote_dist=cfg["quote_dist"],
        queue_priority=cfg["queue_priority"],
        order_life_min=cfg["order_life_min"],
    )
    r_fill = gross - cfg["fill_cost_rt"] - cfg["fill_latency_k"] * rng
    r_chase = cfg["unfilled_capture"] * gross - cfg["chase_cost_rt"] - cfg["chase_latency_k"] * rng
    return r_fill, r_chase, p_fill


def score_trade(rec: Rec, mode: str, chase_threshold: float | None = None) -> tuple[float, float, float, int]:
    """
    Returns (ret, p_fill, capture_ratio, chase_flag).
    chase_flag is 1 when strategy chooses to chase on unfilled branch.
    """
    if mode == "taker_taker":
        r = rec.gross - TAKER_CFG["cost_rt"] - TAKER_CFG["latency_k"] * rec.next_range
        return r, 1.0, 1.0, 1

    r_fill, r_chase, p_fill = maker_first_components(rec.gross, rec.next_range, MAKER_CFG)

    if mode == "maker_first_always_chase":
        r = p_fill * r_fill + (1.0 - p_fill) * r_chase
        cap = p_fill + (1.0 - p_fill) * MAKER_CFG["unfilled_capture"]
        return r, p_fill, cap, 1

    if mode == "maker_only_passive":
        r = p_fill * r_fill
        return r, p_fill, p_fill, 0

    if mode == "maker_first_strength_gated":
        strength = -rec.z if rec.z is not None else 0.0
        chase = 1 if (chase_threshold is not None and strength >= chase_threshold) else 0
        unfilled_branch = r_chase if chase == 1 else 0.0
        r = p_fill * r_fill + (1.0 - p_fill) * unfilled_branch
        cap = p_fill + (1.0 - p_fill) * (MAKER_CFG["unfilled_capture"] if chase else 0.0)
        return r, p_fill, cap, chase

    raise ValueError(f"unknown mode: {mode}")


def simulate_for_quantile(recs: list[Rec], q: float) -> tuple[float, float, float, float, float]:
    """Returns avg_ret, threshold, avg_fill, avg_capture, unfilled_chase_rate."""
    if not recs:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    strengths = [-r.z for r in recs if r.z is not None]
    if not strengths:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    thr = quantile(strengths, q)

    rets = []
    fills = []
    captures = []
    chase_weights = []
    unfilled_weights = []

    for r in recs:
        ret, p_fill, cap, chase = score_trade(r, "maker_first_strength_gated", chase_threshold=thr)
        rets.append(ret)
        fills.append(p_fill)
        captures.append(cap)
        unfilled = 1.0 - p_fill
        unfilled_weights.append(unfilled)
        chase_weights.append(unfilled * chase)

    denom = sum(unfilled_weights)
    unfilled_chase_rate = (sum(chase_weights) / denom) if denom > 1e-12 else 0.0

    return mean(rets), thr, mean(fills), mean(captures), unfilled_chase_rate


def select_quantile(train_recs: list[Rec]) -> dict:
    if len(train_recs) < 25:
        # Low-data fallback: moderate gate.
        q = 0.70
        avg_ret, thr, avg_fill, avg_cap, unfilled_chase_rate = simulate_for_quantile(train_recs, q)
        return {
            "selected_q": q,
            "selected_threshold": thr,
            "train_avg_ret": avg_ret,
            "train_avg_bp": avg_ret * 1e4,
            "train_avg_fill": avg_fill,
            "train_avg_capture": avg_cap,
            "train_unfilled_chase_rate": unfilled_chase_rate,
            "selection_mode": "fallback_low_data",
        }

    leaderboard = []
    for q in Q_CANDIDATES:
        avg_ret, thr, avg_fill, avg_cap, unfilled_chase_rate = simulate_for_quantile(train_recs, q)
        leaderboard.append(
            {
                "q": q,
                "threshold": thr,
                "avg_ret": avg_ret,
                "avg_bp": avg_ret * 1e4,
                "avg_fill": avg_fill,
                "avg_capture": avg_cap,
                "unfilled_chase_rate": unfilled_chase_rate,
            }
        )

    # maximize training avg return; tie-break toward higher q (more selective / safer)
    leaderboard.sort(key=lambda x: (x["avg_ret"], x["q"]), reverse=True)
    best = leaderboard[0]

    return {
        "selected_q": best["q"],
        "selected_threshold": best["threshold"],
        "train_avg_ret": best["avg_ret"],
        "train_avg_bp": best["avg_bp"],
        "train_avg_fill": best["avg_fill"],
        "train_avg_capture": best["avg_capture"],
        "train_unfilled_chase_rate": best["unfilled_chase_rate"],
        "selection_mode": "argmax_train_avg_ret",
        "leaderboard": leaderboard,
    }


def run() -> dict:
    rows = load_rows()

    t = [r[0] for r in rows]
    fr = [r[1] for r in rows]
    p = [r[2] for r in rows]
    bar_rng = [r[3] for r in rows]

    z = rolling_zscore(fr, lookback=90)
    vol = rolling_vol(p, lookback=21)
    ret_next = [p[i + 1] / p[i] - 1 for i in range(len(p) - 1)]

    recs: list[Rec] = []
    for i in range(90, len(rows) - 1):
        year = dt.datetime.fromtimestamp(t[i] / 1000, dt.timezone.utc).year
        recs.append(
            Rec(
                t=t[i],
                year=year,
                z=z[i],
                vol=vol[i],
                gross=ret_next[i] - fr[i + 1],
                next_range=bar_rng[i + 1],
            )
        )

    years = sorted({r.year for r in recs})
    test_years = years[1:]

    strategies = {
        "taker_taker": {"label": "Always taker"},
        "maker_first_always_chase": {"label": "Maker-first, always chase"},
        "maker_first_strength_gated": {"label": "Maker-first, selective chase (WF tuned)"},
        "maker_only_passive": {"label": "Maker-only passive"},
    }

    all_rets: dict[str, list[float]] = {k: [] for k in strategies}
    all_times: dict[str, list[int]] = {k: [] for k in strategies}
    all_fill: dict[str, list[float]] = {k: [] for k in strategies}
    all_capture: dict[str, list[float]] = {k: [] for k in strategies}
    all_chase_flag: dict[str, list[int]] = {k: [] for k in strategies}

    yearly = []

    for y in test_years:
        train = [r for r in recs if r.year < y and r.vol is not None]
        test = [r for r in recs if r.year == y]

        vol_vals = [r.vol for r in train if r.vol is not None]
        if not vol_vals:
            yearly.append({"year": y, "trades": 0, "vol_threshold": None})
            continue

        v_thr = quantile(vol_vals, VOL_Q)

        train_selected = [r for r in train if (r.z is not None and r.vol is not None and r.z < -Z_THRESHOLD and r.vol > v_thr)]
        test_selected = [r for r in test if (r.z is not None and r.vol is not None and r.z < -Z_THRESHOLD and r.vol > v_thr)]

        gate_info = select_quantile(train_selected)
        thr = gate_info["selected_threshold"]

        for r in test_selected:
            for mode in strategies:
                if mode == "maker_first_strength_gated":
                    ret, pf, cap, chase = score_trade(r, mode, chase_threshold=thr)
                else:
                    ret, pf, cap, chase = score_trade(r, mode)

                all_rets[mode].append(ret)
                all_times[mode].append(r.t)
                all_fill[mode].append(pf)
                all_capture[mode].append(cap)
                all_chase_flag[mode].append(chase)

        # Year-level diagnostics on selective strategy only.
        gated_test_rets = []
        gated_unfilled_chase_weights = []
        gated_unfilled_weights = []
        for r in test_selected:
            rr, p_fill, _cap, chase = score_trade(r, "maker_first_strength_gated", chase_threshold=thr)
            gated_test_rets.append(rr)
            unfilled = 1.0 - p_fill
            gated_unfilled_weights.append(unfilled)
            gated_unfilled_chase_weights.append(unfilled * chase)

        den = sum(gated_unfilled_weights)
        test_unfilled_chase_rate = (sum(gated_unfilled_chase_weights) / den) if den > 1e-12 else 0.0

        yearly.append(
            {
                "year": y,
                "trades": len(test_selected),
                "vol_threshold": v_thr,
                "gating": gate_info,
                "test_selective_avg_bp": mean(gated_test_rets) * 1e4 if gated_test_rets else 0.0,
                "test_selective_unfilled_chase_rate": test_unfilled_chase_rate,
            }
        )

    scenario_stats = {}
    curves = {}

    for i, mode in enumerate(strategies):
        rets = all_rets[mode]
        times = all_times[mode]

        summary = summarize(rets)
        boot = stationary_bootstrap_mean_ci(
            rets,
            BOOT_ITERS,
            BOOT_SEED + i,
            avg_block_len=AVG_BLOCK_LEN,
        )

        avg_fill = mean(all_fill[mode])
        avg_capture = mean(all_capture[mode])

        # Unfilled chase rate is defined where an unfilled branch exists.
        # Weighted by (1 - p_fill), so it represents probability of chasing
        # conditional on being unfilled.
        unfilled_weights = [1.0 - pf for pf in all_fill[mode]]
        chase_weights = [u * c for u, c in zip(unfilled_weights, all_chase_flag[mode])]
        den = sum(unfilled_weights)
        unfilled_chase_rate = (sum(chase_weights) / den) if den > 1e-12 else 0.0

        scenario_stats[mode] = {
            "label": strategies[mode]["label"],
            "summary": summary,
            "bootstrap_stationary": boot,
            "fill_stats": {
                "avg_fill_prob": avg_fill,
                "expected_skip_fraction": 1.0 - avg_fill,
                "avg_gross_capture": avg_capture,
                "unfilled_chase_rate": unfilled_chase_rate,
            },
        }

        eq = 1.0
        ts = []
        ys = []
        for tt, rr in sorted(zip(times, rets), key=lambda x: x[0]):
            eq *= 1 + rr
            ts.append(dt.datetime.fromtimestamp(tt / 1000, dt.timezone.utc))
            ys.append(eq)
        curves[mode] = {"ts": ts, "eq": ys}

    # Equity plot
    plt.figure(figsize=(10, 5.2))
    for mode in strategies:
        ts = curves[mode]["ts"]
        eq = curves[mode]["eq"]
        if eq:
            plt.plot(ts, eq, label=strategies[mode]["label"])
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Day 19 PM: Opportunity-loss mitigation under maker-first logic")
    plt.xlabel("UTC time")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_CURVE, dpi=170)
    plt.close()

    # Bar comparison: mean bps + capture
    modes = list(strategies.keys())
    labels = [strategies[m]["label"] for m in modes]
    avg_bps = [scenario_stats[m]["summary"]["avg_bp"] for m in modes]
    capture = [scenario_stats[m]["fill_stats"]["avg_gross_capture"] * 100 for m in modes]

    x = list(range(len(labels)))
    width = 0.38

    plt.figure(figsize=(10.5, 5.2))
    plt.bar([i - width / 2 for i in x], avg_bps, width=width, label="Avg return (bps/trade)")
    plt.bar([i + width / 2 for i in x], capture, width=width, label="Expected gross-alpha capture (%)")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.xticks(x, labels, rotation=10)
    plt.title("Opportunity-loss trade-off: expectancy vs captured alpha")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_BAR, dpi=170)
    plt.close()

    out = {
        "meta": {
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sample_start_utc": dt.datetime.fromtimestamp(rows[0][0] / 1000, dt.timezone.utc).isoformat(),
            "sample_end_utc": dt.datetime.fromtimestamp(rows[-1][0] / 1000, dt.timezone.utc).isoformat(),
            "signal": {"z_threshold": Z_THRESHOLD, "vol_q": VOL_Q},
            "walkforward_oos_years": test_years,
            "bootstrap_iters": BOOT_ITERS,
            "avg_block_len": AVG_BLOCK_LEN,
            "maker_cfg": MAKER_CFG,
            "taker_cfg": TAKER_CFG,
            "gating": {
                "q_candidates": Q_CANDIDATES,
                "selection_rule": "For each test year y: pick q maximizing prior-years avg return; chase unfilled only if strength (-z) >= quantile_q(train_strength).",
            },
        },
        "counts": {
            "rows_total": len(rows),
            "oos_trades": len(all_rets["maker_first_always_chase"]),
            "yearly": yearly,
        },
        "scenarios": scenario_stats,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
