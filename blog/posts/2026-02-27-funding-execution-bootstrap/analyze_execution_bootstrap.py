#!/usr/bin/env python3
"""
Day 17: Execution realism + stationary bootstrap on BTC funding-regime edge.

Goal:
- Keep yesterday's fixed signal family (z_funding + vol gate)
- Evaluate OOS-only trades in expanding yearly walk-forward form
- Stress returns with execution scenarios (fees/spread, latency slippage proxy, partial fills)
- Estimate confidence with stationary bootstrap (serial-dependence-aware)
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

# Fixed baseline from prior work
Z_THRESHOLD = 1.0
VOL_Q = 0.75

BOOT_ITERS = 5000
BOOT_SEED = 20260227
AVG_BLOCK_LEN = 5  # stationary bootstrap expected block length in trades

SCENARIOS = {
    "ideal_4bps_full_fill": {
        "label": "Ideal (4bps, full fill)",
        "cost": 0.0004,
        "latency_k": 0.0,
        "fill": 1.00,
    },
    "realistic_7bps_85fill_10pct_range": {
        "label": "Realistic (7bps, 85% fill, 10% range latency)",
        "cost": 0.0007,
        "latency_k": 0.10,
        "fill": 0.85,
    },
    "stressed_10bps_70fill_20pct_range": {
        "label": "Stressed (10bps, 70% fill, 20% range latency)",
        "cost": 0.0010,
        "latency_k": 0.20,
        "fill": 0.70,
    },
}

POST_DIR = Path(__file__).resolve().parent
OUT_JSON = POST_DIR / "day17-execution-bootstrap-results.json"
OUT_CURVE = POST_DIR / "day17-execution-equity-curves.png"
OUT_CI = POST_DIR / "day17-bootstrap-ci.png"


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
    """Returns list of (t8, funding_rate, open_price, high_low_range)."""
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


def lag1_autocorr(xs: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    m = mean(xs)
    num = sum((xs[i] - m) * (xs[i - 1] - m) for i in range(1, len(xs)))
    den = sum((x - m) ** 2 for x in xs)
    if abs(den) < 1e-12:
        return 0.0
    return num / den


def iid_bootstrap_mean_ci(xs: list[float], iters: int, seed: int) -> dict:
    if not xs:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p_gt_0": 0.0}

    rng = random.Random(seed)
    n = len(xs)
    means = []
    for _ in range(iters):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))

    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    p_gt_0 = sum(1 for m in means if m > 0) / len(means)

    return {
        "mean": mean(xs),
        "ci95_low": lo,
        "ci95_high": hi,
        "p_gt_0": p_gt_0,
    }


def stationary_bootstrap_mean_ci(
    xs: list[float],
    iters: int,
    seed: int,
    avg_block_len: float,
) -> dict:
    """
    Stationary bootstrap (Politis-Romano style):
    - with restart prob p = 1/L, start a new random block
    - otherwise continue previous index + 1 (circular)
    """
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


def apply_execution(gross: float, rng: float, cost: float, latency_k: float, fill: float) -> float:
    # fill fraction participates in PnL; unfilled fraction sits in cash (0 return)
    executed = gross - cost - latency_k * rng
    return fill * executed


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

    # OOS baseline selection (expanding yearly walk-forward, fixed z and q)
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

    # Scenario evaluation
    scenario_stats = {}
    curves = {}

    for i, (key, s) in enumerate(SCENARIOS.items()):
        rets = [
            apply_execution(g, rg, cost=s["cost"], latency_k=s["latency_k"], fill=s["fill"])
            for g, rg in zip(gross, ranges)
        ]

        scenario_stats[key] = {
            "assumptions": s,
            "summary": summarize(rets),
            "lag1_autocorr": lag1_autocorr(rets),
            "bootstrap_iid": iid_bootstrap_mean_ci(rets, BOOT_ITERS, BOOT_SEED + i),
            "bootstrap_stationary": stationary_bootstrap_mean_ci(
                rets,
                BOOT_ITERS,
                BOOT_SEED + 5000 + i,
                avg_block_len=AVG_BLOCK_LEN,
            ),
        }

        # curve
        eq = 1.0
        ts = []
        ys = []
        for tt, rr in sorted(zip(times, rets), key=lambda x: x[0]):
            eq *= 1 + rr
            ts.append(dt.datetime.fromtimestamp(tt / 1000, dt.timezone.utc))
            ys.append(eq)
        curves[key] = {"ts": ts, "eq": ys}

    # Plot equity curves
    plt.figure(figsize=(10, 5.3))
    for key, s in SCENARIOS.items():
        ts = curves[key]["ts"]
        eq = curves[key]["eq"]
        if eq:
            plt.plot(ts, eq, label=s["label"])
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("BTC funding-regime edge: OOS equity under execution scenarios")
    plt.xlabel("UTC time")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_CURVE, dpi=170)
    plt.close()

    # Plot stationary-bootstrap CIs (mean return per trade)
    keys = list(SCENARIOS.keys())
    labels = [SCENARIOS[k]["label"] for k in keys]
    means_bp = [scenario_stats[k]["bootstrap_stationary"]["mean"] * 1e4 for k in keys]
    lo_bp = [scenario_stats[k]["bootstrap_stationary"]["ci95_low"] * 1e4 for k in keys]
    hi_bp = [scenario_stats[k]["bootstrap_stationary"]["ci95_high"] * 1e4 for k in keys]

    x = list(range(len(keys)))
    yerr = [
        [means_bp[i] - lo_bp[i] for i in range(len(keys))],
        [hi_bp[i] - means_bp[i] for i in range(len(keys))],
    ]

    plt.figure(figsize=(9.5, 4.8))
    plt.errorbar(x, means_bp, yerr=yerr, fmt="o", capsize=5)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(x, labels, rotation=10)
    plt.ylabel("Mean return per trade (bps)")
    plt.title(f"Stationary bootstrap 95% CI (avg block length = {AVG_BLOCK_LEN} trades)")
    plt.tight_layout()
    plt.savefig(OUT_CI, dpi=170)
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
