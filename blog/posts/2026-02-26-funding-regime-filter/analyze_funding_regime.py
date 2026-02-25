#!/usr/bin/env python3
"""
Day 16 research script:
Funding-crowding mean reversion on Binance perps, with volatility regime filter.

No pandas dependency.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import requests

BASE_FR = "https://fapi.binance.com/fapi/v1/fundingRate"
BASE_MP = "https://fapi.binance.com/fapi/v1/markPriceKlines"
EIGHT_HOURS_MS = 8 * 60 * 60 * 1000
START_UTC = dt.datetime(2022, 1, 1, tzinfo=dt.timezone.utc)
FEE_ROUNDTRIP = 0.0004  # 4 bps roundtrip (e.g., 2 bps per side maker)

POST_DIR = Path(__file__).resolve().parent
OUT_JSON = POST_DIR / "day16-results.json"
OUT_PNG_CURVE = POST_DIR / "day16-funding-regime-curves.png"
OUT_PNG_YEAR = POST_DIR / "day16-funding-yearly-bps.png"


def fetch_paginated(url: str, params: dict, max_limit: int) -> list:
    out = []
    cursor = params["startTime"]
    end_time = params["endTime"]

    while True:
        p = dict(params)
        p["startTime"] = cursor
        p["limit"] = max_limit

        data = requests.get(url, params=p, timeout=25).json()
        if not data or isinstance(data, dict):
            break

        out.extend(data)

        last = int(data[-1]["fundingTime"]) if isinstance(data[-1], dict) else int(data[-1][0])
        if len(data) < max_limit or last >= end_time:
            break

        cursor = last + 1
        time.sleep(0.03)

    return out


def load_symbol(symbol: str) -> list[tuple[int, float, float]]:
    start_ms = int(START_UTC.timestamp() * 1000)
    end_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)

    fr_raw = fetch_paginated(
        BASE_FR,
        {"symbol": symbol, "startTime": start_ms, "endTime": end_ms},
        max_limit=1000,
    )
    mp_raw = fetch_paginated(
        BASE_MP,
        {"symbol": symbol, "interval": "8h", "startTime": start_ms, "endTime": end_ms},
        max_limit=1500,
    )

    # Mark-price 8h kline open timestamps line up with funding boundaries.
    mark_open = {int(k[0]): float(k[1]) for k in mp_raw}

    rows = []
    for x in fr_raw:
        t = int(x["fundingTime"])
        fr = float(x["fundingRate"])
        t8 = t - (t % EIGHT_HOURS_MS)
        if t8 in mark_open:
            rows.append((t8, fr, mark_open[t8]))

    rows.sort(key=lambda r: r[0])

    # Deduplicate any repeated funding timestamps by keeping the latest observed row.
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


def rolling_vol_from_prices(prices: list[float], lookback: int) -> list[float | None]:
    out: list[float | None] = [None] * len(prices)
    lr = [None] + [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
    for i in range(lookback + 1, len(prices)):
        w = lr[i - lookback : i]
        m = sum(w) / len(w)
        var = sum((x - m) ** 2 for x in w) / len(w)
        out[i] = math.sqrt(var)
    return out


def summarize(rets: list[float], periods_per_year: int = 1095) -> dict:
    if not rets:
        return {"n": 0, "avg": 0.0, "avg_bp": 0.0, "sharpe": 0.0, "equity": 1.0}

    avg = sum(rets) / len(rets)
    var = sum((x - avg) ** 2 for x in rets) / len(rets)
    sd = math.sqrt(var)
    sharpe = (avg / sd) * math.sqrt(periods_per_year) if sd > 1e-12 else 0.0

    equity = 1.0
    for r in rets:
        equity *= 1 + r

    return {
        "n": len(rets),
        "avg": avg,
        "avg_bp": avg * 1e4,
        "sharpe": sharpe,
        "equity": equity,
    }


def run() -> dict:
    btc_rows = load_symbol("BTCUSDT")
    eth_rows = load_symbol("ETHUSDT")

    def prep(rows):
        t = [r[0] for r in rows]
        fr = [r[1] for r in rows]
        p = [r[2] for r in rows]
        ret_next = [p[i + 1] / p[i] - 1 for i in range(len(p) - 1)]
        z = rolling_zscore(fr, lookback=90)
        vol = rolling_vol_from_prices(p, lookback=21)
        v_nonnull = [x for x in vol if x is not None]
        v_med = statistics.median(v_nonnull)
        v_q75 = statistics.quantiles(v_nonnull, n=4)[2]
        return t, fr, p, ret_next, z, vol, v_med, v_q75

    btc_t, btc_fr, btc_p, btc_ret, btc_z, btc_vol, btc_med, btc_q75 = prep(btc_rows)
    eth_t, eth_fr, eth_p, eth_ret, eth_z, _, _, _ = prep(eth_rows)

    # Strategy A: symmetric contrarian at |z| > 1
    btc_sym_rets = []
    eth_sym_rets = []
    btc_sym_curve = []
    btc_sym_times = []

    eq = 1.0
    for i in range(90, len(btc_rows) - 1):
        zi = btc_z[i]
        if zi is None or abs(zi) <= 1:
            continue
        s = -1 if zi > 1 else 1
        r = s * btc_ret[i] - s * btc_fr[i + 1] - FEE_ROUNDTRIP
        btc_sym_rets.append(r)
        eq *= 1 + r
        btc_sym_curve.append(eq)
        btc_sym_times.append(btc_t[i])

    for i in range(90, len(eth_rows) - 1):
        zi = eth_z[i]
        if zi is None or abs(zi) <= 1:
            continue
        s = -1 if zi > 1 else 1
        r = s * eth_ret[i] - s * eth_fr[i + 1] - FEE_ROUNDTRIP
        eth_sym_rets.append(r)

    # Strategy B: BTC long-only on negative funding panic + high vol filter
    btc_filtered_rets = []
    btc_filtered_curve = []
    btc_filtered_times = []
    eq2 = 1.0

    for i in range(90, len(btc_rows) - 1):
        zi = btc_z[i]
        vi = btc_vol[i]
        if zi is None or vi is None:
            continue
        if zi < -1 and vi > btc_q75:
            s = 1
            r = s * btc_ret[i] - s * btc_fr[i + 1] - FEE_ROUNDTRIP
            btc_filtered_rets.append(r)
            eq2 *= 1 + r
            btc_filtered_curve.append(eq2)
            btc_filtered_times.append(btc_t[i])

    # Yearly bps for BTC variants
    yearly = {
        "btc_sym": defaultdict(list),
        "btc_filtered": defaultdict(list),
    }

    for i in range(90, len(btc_rows) - 1):
        zi = btc_z[i]
        if zi is None:
            continue

        year = dt.datetime.fromtimestamp(btc_t[i] / 1000, dt.timezone.utc).year

        if abs(zi) > 1:
            s = -1 if zi > 1 else 1
            r = s * btc_ret[i] - s * btc_fr[i + 1] - FEE_ROUNDTRIP
            yearly["btc_sym"][year].append(r)

        vi = btc_vol[i]
        if vi is not None and zi < -1 and vi > btc_q75:
            r2 = btc_ret[i] - btc_fr[i + 1] - FEE_ROUNDTRIP
            yearly["btc_filtered"][year].append(r2)

    yearly_bps = {
        k: {str(y): (sum(v) / len(v) * 1e4) for y, v in sorted(d.items())}
        for k, d in yearly.items()
    }

    # Plot cumulative curves
    def ms_to_dt(xs):
        return [dt.datetime.fromtimestamp(x / 1000, dt.timezone.utc) for x in xs]

    plt.figure(figsize=(10, 5.2))
    if btc_sym_curve:
        plt.plot(ms_to_dt(btc_sym_times), btc_sym_curve, label="BTC contrarian |z|>1")
    if btc_filtered_curve:
        plt.plot(ms_to_dt(btc_filtered_times), btc_filtered_curve, label="BTC long z<-1 & vol>Q75")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Funding-Crowding Strategies on BTCUSDT (net of 4 bps roundtrip)")
    plt.ylabel("Equity (1.0 = flat)")
    plt.xlabel("UTC time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG_CURVE, dpi=170)
    plt.close()

    # Plot yearly bps bars for BTC
    years = sorted({int(y) for y in yearly_bps["btc_sym"].keys()} | {int(y) for y in yearly_bps["btc_filtered"].keys()})
    sym_vals = [yearly_bps["btc_sym"].get(str(y), 0.0) for y in years]
    fil_vals = [yearly_bps["btc_filtered"].get(str(y), 0.0) for y in years]

    x = list(range(len(years)))
    w = 0.38
    plt.figure(figsize=(9.5, 4.8))
    plt.bar([i - w / 2 for i in x], sym_vals, width=w, label="BTC contrarian |z|>1")
    plt.bar([i + w / 2 for i in x], fil_vals, width=w, label="BTC long z<-1 & vol>Q75")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(x, [str(y) for y in years])
    plt.ylabel("Average return per trade (bps)")
    plt.title("Yearly trade expectancy (BTC)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG_YEAR, dpi=170)
    plt.close()

    results = {
        "meta": {
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sample_start_utc": dt.datetime.fromtimestamp(btc_t[0] / 1000, dt.timezone.utc).isoformat(),
            "sample_end_utc": dt.datetime.fromtimestamp(btc_t[-1] / 1000, dt.timezone.utc).isoformat(),
            "fee_roundtrip": FEE_ROUNDTRIP,
            "z_lookback_intervals": 90,
            "vol_lookback_intervals": 21,
            "interval_hours": 8,
        },
        "summary": {
            "btc_contrarian": summarize(btc_sym_rets),
            "eth_contrarian": summarize(eth_sym_rets),
            "btc_filtered_long_only": summarize(btc_filtered_rets),
        },
        "yearly_bps": yearly_bps,
        "counts": {
            "btc_rows": len(btc_rows),
            "eth_rows": len(eth_rows),
        },
    }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    res = run()
    print(json.dumps(res, indent=2))
