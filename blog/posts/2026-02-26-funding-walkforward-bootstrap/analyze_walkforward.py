#!/usr/bin/env python3
"""
Day 16 (afternoon): walk-forward validation + bootstrap CI
for BTC funding-crowding long-only regime strategy.

Strategy family:
  long when z_funding < -z_threshold and realized_vol > quantile(vol, q)

Validation:
  - Expanding yearly walk-forward (train on years < y, test on year y)
  - Parameter grid search on train only
  - Compare vs fixed baseline params (z=1.0, q=0.75)
  - Bootstrap CI on out-of-sample mean return per trade
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
FEE_ROUNDTRIP = 0.0004

Z_GRID = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
BASELINE = {"z": 1.0, "q": 0.75}
MIN_TRAIN_TRADES = 40
BOOTSTRAP_ITERS = 5000
RNG_SEED = 42

POST_DIR = Path(__file__).resolve().parent
OUT_JSON = POST_DIR / "day16b-results.json"
OUT_CURVE = POST_DIR / "day16b-oos-curves.png"
OUT_YEAR = POST_DIR / "day16b-oos-yearly-bps.png"


@dataclass
class Rec:
    t: int
    year: int
    z: float | None
    vol: float | None
    net_long: float


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
        last = int(data[-1]["fundingTime"]) if isinstance(data[-1], dict) else int(data[-1][0])
        if len(data) < max_limit or last >= end_time:
            break
        cursor = last + 1
        time.sleep(0.03)

    return out


def load_btc_rows() -> list[tuple[int, float, float]]:
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

    mark_open = {int(k[0]): float(k[1]) for k in mp_raw}

    rows = []
    for x in fr_raw:
        t = int(x["fundingTime"])
        fr = float(x["fundingRate"])
        t8 = t - (t % EIGHT_HOURS_MS)
        if t8 in mark_open:
            rows.append((t8, fr, mark_open[t8]))

    rows.sort(key=lambda r: r[0])

    dedup = []
    for t, fr, p in rows:
        if dedup and dedup[-1][0] == t:
            dedup[-1] = (t, fr, p)
        else:
            dedup.append((t, fr, p))

    return dedup


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


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("quantile() got empty list")
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "avg": 0.0, "avg_bp": 0.0, "win_rate": 0.0, "equity": 1.0}
    eq = 1.0
    wins = 0
    for r in xs:
        eq *= 1 + r
        if r > 0:
            wins += 1
    return {
        "n": len(xs),
        "avg": mean(xs),
        "avg_bp": mean(xs) * 1e4,
        "win_rate": wins / len(xs),
        "equity": eq,
    }


def select_trades(recs: list[Rec], z_thr: float, vol_q: float, fit_recs: list[Rec], ret_with_time: bool = False):
    vol_vals = [r.vol for r in fit_recs if r.vol is not None]
    if not vol_vals:
        return []
    v_thr = quantile(vol_vals, vol_q)

    picked = []
    for r in recs:
        if r.z is None or r.vol is None:
            continue
        if r.z < -z_thr and r.vol > v_thr:
            picked.append((r.t, r.net_long) if ret_with_time else r.net_long)
    return picked


def bootstrap_mean_ci(xs: list[float], iters: int = BOOTSTRAP_ITERS, seed: int = RNG_SEED) -> dict:
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


def run() -> dict:
    rows = load_btc_rows()
    t = [r[0] for r in rows]
    fr = [r[1] for r in rows]
    p = [r[2] for r in rows]

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
                net_long=ret_next[i] - fr[i + 1] - FEE_ROUNDTRIP,
            )
        )

    years = sorted({r.year for r in recs})
    test_years = years[1:]  # expanding train -> first year only train

    wf_all: list[tuple[int, float]] = []
    base_all: list[tuple[int, float]] = []
    yearly_rows = []

    for y in test_years:
        train = [r for r in recs if r.year < y]
        test = [r for r in recs if r.year == y]

        best = None
        for z_thr in Z_GRID:
            for q in Q_GRID:
                tr = select_trades(train, z_thr=z_thr, vol_q=q, fit_recs=train, ret_with_time=False)
                if len(tr) < MIN_TRAIN_TRADES:
                    continue
                score = mean(tr)
                if best is None or score > best["score"]:
                    best = {"z": z_thr, "q": q, "score": score, "n_train": len(tr)}

        if best is None:
            best = {
                "z": BASELINE["z"],
                "q": BASELINE["q"],
                "score": 0.0,
                "n_train": 0,
            }

        wf_trades = select_trades(test, z_thr=best["z"], vol_q=best["q"], fit_recs=train, ret_with_time=True)
        base_trades = select_trades(
            test,
            z_thr=BASELINE["z"],
            vol_q=BASELINE["q"],
            fit_recs=train,
            ret_with_time=True,
        )

        wf_all.extend(wf_trades)
        base_all.extend(base_trades)

        wf_rets = [x[1] for x in wf_trades]
        base_rets = [x[1] for x in base_trades]

        yearly_rows.append(
            {
                "year": y,
                "selected_params": {"z": best["z"], "vol_q": best["q"]},
                "train_avg_bp": best["score"] * 1e4,
                "train_trades": best["n_train"],
                "wf_oos": summarize(wf_rets),
                "baseline_oos": summarize(base_rets),
            }
        )

    wf_rets_all = [x[1] for x in wf_all]
    base_rets_all = [x[1] for x in base_all]

    # Plot OOS cumulative curves
    def curve(points: list[tuple[int, float]]):
        eq = 1.0
        ts = []
        vals = []
        for tt, rr in sorted(points, key=lambda x: x[0]):
            eq *= 1 + rr
            ts.append(dt.datetime.fromtimestamp(tt / 1000, dt.timezone.utc))
            vals.append(eq)
        return ts, vals

    ts_wf, eq_wf = curve(wf_all)
    ts_b, eq_b = curve(base_all)

    plt.figure(figsize=(10, 5.2))
    if eq_wf:
        plt.plot(ts_wf, eq_wf, label="Walk-forward tuned")
    if eq_b:
        plt.plot(ts_b, eq_b, label="Fixed baseline (z=1.0, q=0.75)")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("BTC funding-regime strategy: out-of-sample equity only")
    plt.xlabel("UTC time")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_CURVE, dpi=170)
    plt.close()

    # Plot yearly OOS avg bps
    yrs = [r["year"] for r in yearly_rows]
    wf_bps = [r["wf_oos"]["avg_bp"] for r in yearly_rows]
    b_bps = [r["baseline_oos"]["avg_bp"] for r in yearly_rows]

    x = list(range(len(yrs)))
    w = 0.38
    plt.figure(figsize=(9.5, 4.8))
    plt.bar([i - w / 2 for i in x], wf_bps, width=w, label="Walk-forward tuned")
    plt.bar([i + w / 2 for i in x], b_bps, width=w, label="Fixed baseline")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(x, [str(y) for y in yrs])
    plt.ylabel("Average return per trade (bps)")
    plt.title("Yearly OOS expectancy (BTC)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_YEAR, dpi=170)
    plt.close()

    res = {
        "meta": {
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sample_start_utc": dt.datetime.fromtimestamp(rows[0][0] / 1000, dt.timezone.utc).isoformat(),
            "sample_end_utc": dt.datetime.fromtimestamp(rows[-1][0] / 1000, dt.timezone.utc).isoformat(),
            "fee_roundtrip": FEE_ROUNDTRIP,
            "grid": {"z": Z_GRID, "vol_q": Q_GRID},
            "bootstrap_iters": BOOTSTRAP_ITERS,
            "min_train_trades": MIN_TRAIN_TRADES,
        },
        "overall": {
            "walkforward_oos": summarize(wf_rets_all),
            "baseline_oos": summarize(base_rets_all),
            "walkforward_bootstrap": bootstrap_mean_ci(wf_rets_all),
            "baseline_bootstrap": bootstrap_mean_ci(base_rets_all),
        },
        "yearly": yearly_rows,
        "counts": {
            "btc_rows": len(rows),
            "oos_test_years": test_years,
            "walkforward_trades_total": len(wf_rets_all),
            "baseline_trades_total": len(base_rets_all),
        },
    }

    OUT_JSON.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2))
