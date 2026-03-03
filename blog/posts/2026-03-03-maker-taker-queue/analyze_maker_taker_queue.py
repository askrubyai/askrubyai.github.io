#!/usr/bin/env python3
"""
Day 18: Explicit maker/taker queue assumptions on the BTC funding-regime edge.

Builds directly on Day 17:
- same signal family (funding z-score + volatility gate)
- same OOS expanding yearly walk-forward protocol
- replaces coarse fill-factor scenarios with explicit order-type logic
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

# Signal baseline (unchanged from Day 16/17)
Z_THRESHOLD = 1.0
VOL_Q = 0.75

BOOT_ITERS = 5000
BOOT_SEED = 20260303
AVG_BLOCK_LEN = 5

SCENARIOS = {
    "taker_taker": {
        "label": "Always taker (enter+exit)",
        "mode": "taker",
        "cost_rt": 0.0010,  # 10 bps roundtrip
        "latency_k": 0.05,
        "gross_capture_unfilled": 1.0,
    },
    "maker_first_then_chase": {
        "label": "Maker-first, chase if unfilled",
        "mode": "maker_chase",
        "quote_dist": 0.0006,  # 6 bps from touch
        "queue_priority": 0.55,
        "order_life_min": 5,
        "fill_cost_rt": 0.0007,  # maker+taker
        "fill_latency_k": 0.03,
        "unfilled_capture": 0.60,
        "chase_cost_rt": 0.0010,  # taker+taker fallback
        "chase_latency_k": 0.10,
    },
    "maker_only_passive": {
        "label": "Maker-only (skip if no fill)",
        "mode": "maker_only",
        "quote_dist": 0.0008,
        "queue_priority": 0.45,
        "order_life_min": 3,
        "fill_cost_rt": 0.0004,  # maker+maker
        "fill_latency_k": 0.04,
        "gross_capture_unfilled": 0.0,
    },
}

POST_DIR = Path(__file__).resolve().parent
OUT_JSON = POST_DIR / "day18-maker-taker-queue-results.json"
OUT_CURVE = POST_DIR / "day18-queue-equity-curves.png"
OUT_FILL = POST_DIR / "day18-fill-capture-bars.png"


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


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def maker_fill_probability(
    next_range: float,
    quote_dist: float,
    queue_priority: float,
    order_life_min: float,
) -> float:
    # 8h bars overstate micro fill opportunities. Approximate order-lifetime
    # one-sided excursion by sqrt-time scaling from 8h to order_life_min.
    scale = math.sqrt(order_life_min / (8 * 60))
    micro_excursion = 0.5 * next_range * scale
    touch_prob = clamp(micro_excursion / quote_dist, 0.0, 1.0)
    return clamp(touch_prob * queue_priority, 0.0, 1.0)


def apply_scenario(gross: float, rng: float, scenario: dict) -> tuple[float, float, float]:
    """
    Returns (expected_return, fill_prob, gross_capture_ratio).
    gross_capture_ratio describes expected fraction of gross alpha captured.
    """
    mode = scenario["mode"]

    if mode == "taker":
        r = gross - scenario["cost_rt"] - scenario["latency_k"] * rng
        return r, 1.0, 1.0

    if mode == "maker_chase":
        p_fill = maker_fill_probability(
            next_range=rng,
            quote_dist=scenario["quote_dist"],
            queue_priority=scenario["queue_priority"],
            order_life_min=scenario["order_life_min"],
        )
        r_fill = gross - scenario["fill_cost_rt"] - scenario["fill_latency_k"] * rng
        r_chase = (
            scenario["unfilled_capture"] * gross
            - scenario["chase_cost_rt"]
            - scenario["chase_latency_k"] * rng
        )
        r = p_fill * r_fill + (1.0 - p_fill) * r_chase
        gross_capture = p_fill + (1.0 - p_fill) * scenario["unfilled_capture"]
        return r, p_fill, gross_capture

    if mode == "maker_only":
        p_fill = maker_fill_probability(
            next_range=rng,
            quote_dist=scenario["quote_dist"],
            queue_priority=scenario["queue_priority"],
            order_life_min=scenario["order_life_min"],
        )
        r_fill = gross - scenario["fill_cost_rt"] - scenario["fill_latency_k"] * rng
        r = p_fill * r_fill  # unfilled orders are skipped
        return r, p_fill, p_fill

    raise ValueError(f"unknown mode: {mode}")


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

    selected: list[Rec] = []
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

    times = [r.t for r in selected]
    gross = [r.gross for r in selected]
    ranges = [r.next_range for r in selected]

    scenario_stats = {}
    curves = {}

    for i, (key, s) in enumerate(SCENARIOS.items()):
        rets = []
        fill_probs = []
        capture_ratios = []

        for g, rg in zip(gross, ranges):
            rr, pf, cap = apply_scenario(g, rg, s)
            rets.append(rr)
            fill_probs.append(pf)
            capture_ratios.append(cap)

        scenario_stats[key] = {
            "assumptions": s,
            "summary": summarize(rets),
            "fill_stats": {
                "avg_fill_prob": mean(fill_probs),
                "expected_skip_fraction": 1.0 - mean(fill_probs),
                "avg_gross_capture": mean(capture_ratios),
            },
            "bootstrap_stationary": stationary_bootstrap_mean_ci(
                rets,
                BOOT_ITERS,
                BOOT_SEED + i,
                avg_block_len=AVG_BLOCK_LEN,
            ),
        }

        eq = 1.0
        ts = []
        ys = []
        for tt, rr in sorted(zip(times, rets), key=lambda x: x[0]):
            eq *= 1 + rr
            ts.append(dt.datetime.fromtimestamp(tt / 1000, dt.timezone.utc))
            ys.append(eq)
        curves[key] = {"ts": ts, "eq": ys}

    # Equity curve plot
    plt.figure(figsize=(10, 5.2))
    for key, s in SCENARIOS.items():
        ts = curves[key]["ts"]
        eq = curves[key]["eq"]
        if eq:
            plt.plot(ts, eq, label=s["label"])
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Funding-regime edge with explicit order-type assumptions")
    plt.xlabel("UTC time")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_CURVE, dpi=170)
    plt.close()

    # Fill/capture bar plot
    labels = [SCENARIOS[k]["label"] for k in SCENARIOS]
    fill_vals = [scenario_stats[k]["fill_stats"]["avg_fill_prob"] * 100 for k in SCENARIOS]
    cap_vals = [scenario_stats[k]["fill_stats"]["avg_gross_capture"] * 100 for k in SCENARIOS]

    x = list(range(len(labels)))
    width = 0.35

    plt.figure(figsize=(10, 5.0))
    plt.bar([i - width / 2 for i in x], fill_vals, width=width, label="Expected fill probability (%)")
    plt.bar([i + width / 2 for i in x], cap_vals, width=width, label="Expected gross-alpha capture (%)")
    plt.ylim(0, 105)
    plt.ylabel("Percent")
    plt.title("Queue assumptions: fill vs captured alpha")
    plt.xticks(x, labels, rotation=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FILL, dpi=170)
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
            "queue_fill_proxy": "touch_prob = min(1, 0.5*range*sqrt(order_life_min/480)/quote_dist); fill_prob = touch_prob*queue_priority",
        },
        "counts": {
            "rows_total": len(rows),
            "oos_trades": len(selected),
            "yearly_trade_counts": yearly_counts,
        },
        "scenarios": scenario_stats,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
