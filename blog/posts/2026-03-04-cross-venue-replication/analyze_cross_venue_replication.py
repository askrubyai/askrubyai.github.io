#!/usr/bin/env python3
"""
Day 19: Cross-venue replication of the BTC funding-regime signal.

Question:
Does the Day 16-18 funding/volatility edge survive when we port the exact same
signal and walk-forward protocol across multiple venues?

Venues:
- Binance BTCUSDT perp
- Bybit BTCUSDT linear perp
- OKX BTC-USDT-SWAP

Method:
- Build 8h-aligned series from exchange funding + hourly candles
- Same signal: funding z-score + realized-volatility gate
- Same protocol: expanding yearly OOS walk-forward
- Same execution baseline for comparability: taker/taker + latency drag
- Confidence: stationary bootstrap over trade returns
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

START_UTC = dt.datetime(2022, 1, 1, tzinfo=dt.timezone.utc)
START_MS = int(START_UTC.timestamp() * 1000)
HOUR_MS = 60 * 60 * 1000
EIGHT_HOURS_MS = 8 * HOUR_MS

Z_THRESHOLD = 1.0
VOL_Q = 0.75

COST_RT = 0.0010  # 10 bps roundtrip taker baseline
LATENCY_K = 0.05

BOOT_ITERS = 5000
BOOT_SEED = 20260304
AVG_BLOCK_LEN = 5

POST_DIR = Path(__file__).resolve().parent
OUT_JSON = POST_DIR / "day19-cross-venue-results.json"
OUT_CURVE = POST_DIR / "day19-cross-venue-equity.png"
OUT_CI = POST_DIR / "day19-cross-venue-ci.png"


@dataclass
class Obs:
    t: int
    fr: float
    open_px: float
    next_range: float


@dataclass
class Rec:
    t: int
    year: int
    z: float | None
    vol: float | None
    gross: float
    next_range: float


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


def fetch_json(url: str, params: dict, timeout: int = 30) -> dict | list:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_binance() -> tuple[list[tuple[int, float]], dict[int, tuple[float, float, float, float]]]:
    end_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)

    funding: list[tuple[int, float]] = []
    cursor = START_MS
    while True:
        j = fetch_json(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            {
                "symbol": "BTCUSDT",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        if not j:
            break
        for x in j:
            funding.append((int(x["fundingTime"]), float(x["fundingRate"])))
        last = int(j[-1]["fundingTime"])
        if len(j) < 1000 or last >= end_ms:
            break
        if last < cursor:
            break
        cursor = last + 1
        time.sleep(0.03)

    candles: dict[int, tuple[float, float, float, float]] = {}
    cursor = START_MS
    while True:
        j = fetch_json(
            "https://fapi.binance.com/fapi/v1/markPriceKlines",
            {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
            },
        )
        if not j:
            break
        for k in j:
            t = int(k[0])
            candles[t] = (float(k[1]), float(k[2]), float(k[3]), float(k[4]))
        last = int(j[-1][0])
        if len(j) < 1500 or last >= end_ms:
            break
        if last < cursor:
            break
        cursor = last + 1
        time.sleep(0.03)

    funding = sorted({t: fr for t, fr in funding}.items())
    return funding, candles


def fetch_bybit() -> tuple[list[tuple[int, float]], dict[int, tuple[float, float, float, float]]]:
    end_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)

    # funding history (descending pages)
    funding_raw: list[tuple[int, float]] = []
    cursor_end = end_ms
    while True:
        j = fetch_json(
            "https://api.bybit.com/v5/market/funding/history",
            {
                "category": "linear",
                "symbol": "BTCUSDT",
                "endTime": cursor_end,
                "limit": 200,
            },
        )
        data = j.get("result", {}).get("list", [])
        if not data:
            break
        ts = []
        for x in data:
            t = int(x["fundingRateTimestamp"])
            ts.append(t)
            if START_MS <= t <= end_ms:
                funding_raw.append((t, float(x["fundingRate"])))
        oldest = min(ts)
        if oldest <= START_MS:
            break
        if oldest >= cursor_end:
            break
        cursor_end = oldest - 1
        time.sleep(0.03)

    # 1h kline (descending pages)
    candles: dict[int, tuple[float, float, float, float]] = {}
    cursor_end = end_ms
    while True:
        j = fetch_json(
            "https://api.bybit.com/v5/market/kline",
            {
                "category": "linear",
                "symbol": "BTCUSDT",
                "interval": "60",
                "end": cursor_end,
                "limit": 1000,
            },
        )
        data = j.get("result", {}).get("list", [])
        if not data:
            break
        ts = []
        for row in data:
            t = int(row[0])
            ts.append(t)
            if t < START_MS:
                continue
            candles[t] = (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
        oldest = min(ts)
        if oldest <= START_MS:
            break
        if oldest >= cursor_end:
            break
        cursor_end = oldest - 1
        time.sleep(0.03)

    funding = sorted({t: fr for t, fr in funding_raw}.items())
    return funding, candles


def fetch_okx() -> tuple[list[tuple[int, float]], dict[int, tuple[float, float, float, float]]]:
    end_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)

    # funding history (descending pages via after)
    funding_raw: list[tuple[int, float]] = []
    cursor_after: str | None = None
    while True:
        params = {"instId": "BTC-USDT-SWAP", "limit": "100"}
        if cursor_after is not None:
            params["after"] = cursor_after
        j = fetch_json("https://www.okx.com/api/v5/public/funding-rate-history", params)
        data = j.get("data", [])
        if not data:
            break
        ts = []
        for x in data:
            t = int(x["fundingTime"])
            ts.append(t)
            if START_MS <= t <= end_ms:
                funding_raw.append((t, float(x["fundingRate"])))
        oldest = min(ts)
        if oldest <= START_MS:
            break
        if cursor_after is not None and int(cursor_after) <= oldest:
            break
        cursor_after = str(oldest)
        time.sleep(0.04)

    # 1H candles (descending pages via after)
    candles: dict[int, tuple[float, float, float, float]] = {}
    cursor_after = None
    while True:
        params = {"instId": "BTC-USDT-SWAP", "bar": "1H", "limit": "100"}
        if cursor_after is not None:
            params["after"] = cursor_after
        j = fetch_json("https://www.okx.com/api/v5/market/history-candles", params)
        data = j.get("data", [])
        if not data:
            break
        ts = []
        for row in data:
            t = int(row[0])
            ts.append(t)
            if t < START_MS:
                continue
            candles[t] = (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
        oldest = min(ts)
        if oldest <= START_MS:
            break
        if cursor_after is not None and int(cursor_after) <= oldest:
            break
        cursor_after = str(oldest)
        time.sleep(0.04)

    funding = sorted({t: fr for t, fr in funding_raw}.items())
    return funding, candles


def align_to_8h(
    funding: list[tuple[int, float]],
    candles_1h: dict[int, tuple[float, float, float, float]],
) -> list[Obs]:
    opens = {t: v[0] for t, v in candles_1h.items()}
    highs = {t: v[1] for t, v in candles_1h.items()}
    lows = {t: v[2] for t, v in candles_1h.items()}

    rows: list[Obs] = []
    for t, fr in funding:
        if t % EIGHT_HOURS_MS != 0:
            continue
        if t not in opens or (t + EIGHT_HOURS_MS) not in opens:
            continue

        hour_marks = [t + h * HOUR_MS for h in range(8)]
        if any(hm not in highs or hm not in lows for hm in hour_marks):
            continue

        o = opens[t]
        hi = max(highs[hm] for hm in hour_marks)
        lo = min(lows[hm] for hm in hour_marks)
        rng = (hi - lo) / o if o > 0 else 0.0
        rows.append(Obs(t=t, fr=fr, open_px=o, next_range=rng))

    rows.sort(key=lambda r: r.t)

    # de-dup by timestamp, keep latest occurrence
    dedup = []
    for r in rows:
        if dedup and dedup[-1].t == r.t:
            dedup[-1] = r
        else:
            dedup.append(r)

    return dedup


def run_signal(rows: list[Obs], venue: str) -> dict:
    fr = [r.fr for r in rows]
    p = [r.open_px for r in rows]
    rng = [r.next_range for r in rows]

    z = rolling_zscore(fr, lookback=90)
    vol = rolling_vol(p, lookback=21)
    ret_next = [p[i + 1] / p[i] - 1 for i in range(len(p) - 1)]

    recs: list[Rec] = []
    for i in range(90, len(rows) - 1):
        year = dt.datetime.fromtimestamp(rows[i].t / 1000, dt.timezone.utc).year
        recs.append(
            Rec(
                t=rows[i].t,
                year=year,
                z=z[i],
                vol=vol[i],
                gross=ret_next[i] - fr[i + 1],
                next_range=rng[i + 1],
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

    rets = [g - COST_RT - LATENCY_K * rg for g, rg in zip(gross, ranges)]

    # equity curve
    eq = 1.0
    curve_t = []
    curve_y = []
    for t, r in sorted(zip(times, rets), key=lambda x: x[0]):
        eq *= 1 + r
        curve_t.append(dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc))
        curve_y.append(eq)

    out = {
        "venue": venue,
        "counts": {
            "rows_total": len(rows),
            "oos_trades": len(rets),
            "yearly_trade_counts": yearly_counts,
        },
        "summary": summarize(rets),
        "bootstrap_stationary": stationary_bootstrap_mean_ci(
            rets,
            BOOT_ITERS,
            BOOT_SEED + abs(hash(venue)) % 10000,
            avg_block_len=AVG_BLOCK_LEN,
        ),
        "curve": {
            "ts": [x.isoformat() for x in curve_t],
            "equity": curve_y,
        },
    }
    return out


def plot_equity(results: dict[str, dict]) -> None:
    plt.figure(figsize=(10, 5.2))
    for venue, payload in results.items():
        ts = [dt.datetime.fromisoformat(x) for x in payload["curve"]["ts"]]
        eq = payload["curve"]["equity"]
        if eq:
            plt.plot(ts, eq, label=venue)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Day 19: OOS equity by venue (same signal + same execution)")
    plt.xlabel("UTC time")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_CURVE, dpi=170)
    plt.close()


def plot_ci(results: dict[str, dict]) -> None:
    venues = list(results.keys())
    means = [results[v]["summary"]["avg_bp"] for v in venues]
    lo = [results[v]["bootstrap_stationary"]["ci95_low"] * 1e4 for v in venues]
    hi = [results[v]["bootstrap_stationary"]["ci95_high"] * 1e4 for v in venues]

    x = list(range(len(venues)))
    yerr_low = [m - l for m, l in zip(means, lo)]
    yerr_hi = [h - m for m, h in zip(means, hi)]

    plt.figure(figsize=(9.2, 5.0))
    plt.errorbar(
        x,
        means,
        yerr=[yerr_low, yerr_hi],
        fmt="o",
        capsize=6,
        linewidth=1.4,
        markersize=6,
    )
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.xticks(x, venues)
    plt.ylabel("Mean return (bps/trade)")
    plt.title("Stationary-bootstrap CI by venue (95%)")
    plt.tight_layout()
    plt.savefig(OUT_CI, dpi=170)
    plt.close()


def run() -> dict:
    venue_loaders = {
        "Binance": fetch_binance,
        "Bybit": fetch_bybit,
        "OKX": fetch_okx,
    }

    venue_results = {}

    for venue, loader in venue_loaders.items():
        funding, candles = loader()
        rows = align_to_8h(funding, candles)
        venue_results[venue] = run_signal(rows, venue)

    plot_equity(venue_results)
    plot_ci(venue_results)

    out = {
        "meta": {
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sample_start_utc": START_UTC.isoformat(),
            "signal": {
                "z_threshold": Z_THRESHOLD,
                "vol_q": VOL_Q,
                "gross": "next_8h_return - next_funding",
                "net": f"gross - {COST_RT:.4f} - {LATENCY_K:.2f}*next_range",
            },
            "walkforward_oos": "expanding yearly (test years = all except first year)",
            "bootstrap": {
                "iters": BOOT_ITERS,
                "avg_block_len": AVG_BLOCK_LEN,
                "method": "stationary bootstrap",
            },
        },
        "venues": venue_results,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
