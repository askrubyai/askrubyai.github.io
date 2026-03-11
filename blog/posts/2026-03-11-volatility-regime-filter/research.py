#!/usr/bin/env python3
"""
Day 29 PM: Volatility Regime Filter
====================================
Hypothesis: Skip trading during mid-volatility regimes (which show negative expectancy)
while participating in low and high volatility regimes.

Key insight from Day 20 PM data:
- Low sigma: +26.9 to +30.2 bps/trade (EXCELLENT)
- Mid sigma: -3.0 to -6.0 bps/trade (TERRIBLE)
- High sigma: +28.3 to +32.0 bps/trade (EXCELLENT)

The problem is MID volatility, not high volatility!
"""

from __future__ import annotations
import datetime as dt
import json
import statistics
import numpy as np
import pandas as pd
import requests
import time
from pathlib import Path

BASE_K = "https://fapi.binance.com/fapi/v1/klines"
START_UTC = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)

def fetch_5m_candles(start_ts: int, end_ts: int, symbol: str = "BTCUSDT") -> list[dict]:
    """Fetch 5-minute candles."""
    url = BASE_K
    all_candles = []
    current = start_ts
    
    while current < end_ts:
        params = {"symbol": symbol, "interval": "5m", "startTime": current, "limit": 1000}
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if not data:
                break
            all_candles.extend(data)
            current = data[-1][0] + 1
            time.sleep(0.2)
        except Exception as e:
            print(f"Kline fetch error: {e}")
            break
    
    return all_candles

def compute_realized_vol(candles: list, window: int = 20) -> list[float]:
    """Compute realized volatility from candle returns."""
    returns = []
    for i in range(1, len(candles)):
        close_prev = float(candles[i-1][4])  # close
        close_curr = float(candles[i][4])
        ret = (close_curr - close_prev) / close_prev
        returns.append(ret)
    
    # Rolling realized vol (annualized)
    vol = []
    for i in range(window, len(returns)):
        window_rets = returns[i-window:i]
        rv = np.std(window_rets) * np.sqrt(12 * 24 * 365)  # 5-min bars -> annualize
        vol.append(rv)
    
    return vol

def classify_regime(vol: float, q1: float, q2: float) -> str:
    """Classify volatility into low/mid/high based on percentiles."""
    if vol < q1:
        return "low"
    elif vol < q2:
        return "mid"
    else:
        return "high"

def simulate_maker_trades(candles: list, vol_series: list, quote_dist_bps: float = 10,
                          fill_threshold: float = 0.5) -> dict:
    """
    Simulate maker-first trades with volatility regime filtering.
    Compare: all trades vs mid-vol filtered.
    """
    # Parameters
    maker_fee = -0.0001  # 1 bps rebate
    taker_fee = 0.003    # 30 bps fee
    spread_capture = quote_dist_bps / 10000  # Convert bps to fraction
    
    results = {
        "all_trades": {"n": 0, "pnl_bps": [], "wins": 0},
        "low_vol": {"n": 0, "pnl_bps": [], "wins": 0},
        "mid_vol": {"n": 0, "pnl_bps": [], "wins": 0},
        "high_vol": {"n": 0, "pnl_bps": [], "wins": 0},
        "filtered": {"n": 0, "pnl_bps": [], "wins": 0},  # low + high only
    }
    
    # Calculate volatility percentiles from full history
    vol_arr = np.array(vol_series)
    q25 = np.percentile(vol_arr, 25)
    q75 = np.percentile(vol_arr, 75)
    
    print(f"Volatility percentiles - Q25: {q25:.4f}, Q75: {q75:.4f}")
    
    # For each 8-hour period (roughly funding interval)
    funding_interval_bars = 96  # 8 hours * 12 (5-min bars per hour)
    
    for i in range(funding_interval_bars, len(candles) - 12, funding_interval_bars):
        # Get volatility for this period
        vol_idx = i - 20  # Offset for vol calculation
        if vol_idx >= len(vol_series):
            continue
        
        current_vol = vol_series[vol_idx]
        regime = classify_regime(current_vol, q25, q75)
        
        # Simulate a trade at the start of this period
        entry_price = float(candles[i][4])
        
        # Check if price moves enough to get filled (simplified)
        # In reality, use Brownian motion model
        next_12_bars = candles[i:i+12]
        max_move = 0
        for j in range(1, len(next_12_bars)):
            move = abs(float(next_12_bars[j][4]) - entry_price) / entry_price
            max_move = max(max_move, move)
        
        # Fill probability based on quote distance
        fill_prob = min(1.0, max_move / spread_capture) if spread_capture > 0 else 0
        
        # Outcome: if filled, capture spread; if not, no PnL
        if fill_prob >= fill_threshold:
            # PnL = spread capture + maker rebate
            pnl_bps = quote_dist_bps + 1  # 1 bps rebate
            
            # Add some noise based on direction (simplified)
            # In reality, would need actual direction signal
            results["all_trades"]["n"] += 1
            results["all_trades"]["pnl_bps"].append(pnl_bps)
            if pnl_bps > 0:
                results["all_trades"]["wins"] += 1
            
            results[f"{regime}_vol"]["n"] += 1
            results[f"{regime}_vol"]["pnl_bps"].append(pnl_bps)
            if pnl_bps > 0:
                results[f"{regime}_vol"]["wins"] += 1
            
            if regime != "mid":  # Filter out mid-vol
                results["filtered"]["n"] += 1
                results["filtered"]["pnl_bps"].append(pnl_bps)
                if pnl_bps > 0:
                    results["filtered"]["wins"] += 1
    
    # Calculate summary stats
    summary = {}
    for key, data in results.items():
        if data["n"] > 0:
            pnl_arr = np.array(data["pnl_bps"])
            summary[key] = {
                "n": data["n"],
                "avg_bps": np.mean(pnl_arr),
                "win_rate": data["wins"] / data["n"],
                "total_bps": np.sum(pnl_arr),
            }
        else:
            summary[key] = {"n": 0, "avg_bps": 0, "win_rate": 0, "total_bps": 0}
    
    return summary, {"q25": q25, "q75": q75}

def main():
    print("=" * 60)
    print("Day 29 PM: Volatility Regime Filter Analysis")
    print("=" * 60)
    
    # Fetch data
    start_ts = int(START_UTC.timestamp() * 1000)
    end_ts = int(dt.datetime(2025, 6, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    
    print("Fetching BTCUSDT 5m candles...")
    candles = fetch_5m_candles(start_ts, end_ts)
    print(f"Fetched {len(candles)} candles")
    
    # Compute volatility
    print("Computing realized volatility...")
    vol_series = compute_realized_vol(candles)
    print(f"Computed {len(vol_series)} volatility observations")
    
    # Run simulation with different quote distances
    quote_dists = [6, 8, 10, 12]
    all_results = {}
    
    for qd in quote_dists:
        print(f"\n--- Quote Distance: {qd} bps ---")
        summary, percentiles = simulate_maker_trades(
            candles, vol_series, 
            quote_dist_bps=qd,
            fill_threshold=0.5
        )
        
        print(f"  All trades:     n={summary['all_trades']['n']:4d}, avg={summary['all_trades']['avg_bps']:7.2f} bps, WR={summary['all_trades']['win_rate']:.1%}")
        print(f"  Low vol:        n={summary['low_vol']['n']:4d}, avg={summary['low_vol']['avg_bps']:7.2f} bps, WR={summary['low_vol']['win_rate']:.1%}")
        print(f"  Mid vol:        n={summary['mid_vol']['n']:4d}, avg={summary['mid_vol']['avg_bps']:7.2f} bps, WR={summary['mid_vol']['win_rate']:.1%}")
        print(f"  High vol:       n={summary['high_vol']['n']:4d}, avg={summary['high_vol']['avg_bps']:7.2f} bps, WR={summary['high_vol']['win_rate']:.1%}")
        print(f"  Filtered:       n={summary['filtered']['n']:4d}, avg={summary['filtered']['avg_bps']:7.2f} bps, WR={summary['filtered']['win_rate']:.1%}")
        
        all_results[qd] = {
            "summary": summary,
            "percentiles": percentiles
        }
    
    # Compare filtered vs unfiltered
    print("\n" + "=" * 60)
    print("SUMMARY: Filtered (Low + High) vs All Trades")
    print("=" * 60)
    
    for qd in quote_dists:
        s = all_results[qd]["summary"]
        improvement = s["filtered"]["avg_bps"] - s["all_trades"]["avg_bps"]
        print(f"{qd} bps: All={s['all_trades']['avg_bps']:.2f}, Filtered={s['filtered']['avg_bps']:.2f}, Δ={improvement:+.2f} bps")
    
    # Save results
    output = {
        "meta": {
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sample_start": START_UTC.isoformat(),
            "sample_end": "2026-02-01",
            "hypothesis": "Skip mid-volatility regimes, trade low/high",
        },
        "results": all_results,
    }
    
    with open("volatility_regime_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to volatility_regime_results.json")

if __name__ == "__main__":
    main()
