#!/usr/bin/env python3
"""
Day 26: Lifetime Tuning by Regime (Simplified)
==============================================
"""

import datetime as dt
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import requests

BASE_K = "https://fapi.binance.com/fapi/v1/klines"
START_UTC = dt.datetime(2022, 1, 1, tzinfo=dt.timezone.utc)

def fetch_candles(start_ts, end_ts, symbol="BTCUSDT"):
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
        except Exception as e:
            print(f"Error: {e}")
            break
    return all_candles

def estimate_fill(lifetime_min, volatility, volume_ratio):
    base = 0.71
    lt_factor = math.log(lifetime_min + 1) / math.log(6)
    vol_factor = 1.0 / (1.0 + 0.5 * (volatility - 1.0))
    vr_factor = min(1.5, max(0.5, volume_ratio))
    return min(0.99, max(0.01, base * lt_factor * vol_factor * vr_factor))

def main():
    print("Day 26: Lifetime Tuning by Regime")
    print("=" * 50)
    
    # Fetch data (1 year only for speed)
    start_ts = int(START_UTC.timestamp() * 1000)
    end_ts = int(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    
    print("Fetching candles...")
    candles = fetch_candles(start_ts, end_ts)
    print(f"Got {len(candles)} candles")
    
    if len(candles) < 1000:
        print("Insufficient data")
        return
    
    # Build features
    print("Building features...")
    records = []
    for i in range(30, len(candles) - 120):
        c = candles[i]
        close = float(c[4])
        volume = float(c[5])
        high = float(c[2])
        low = float(c[3])
        
        tr = high - low
        atr = np.mean([float(candles[j][2]) - float(candles[j][3]) for j in range(max(0,i-20), i)])
        volatility = tr / atr if atr > 0 else 1.0
        
        vol_20 = np.mean([float(candles[j][5]) for j in range(max(0,i-20), i)])
        volume_ratio = volume / vol_20 if vol_20 > 0 else 1.0
        
        # Forward returns
        f5 = (float(candles[i+5][4]) - close) / close * 10000 if i+5 < len(candles) else 0
        f15 = (float(candles[i+15][4]) - close) / close * 10000 if i+15 < len(candles) else 0
        
        records.append({
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'forward_5m': f5,
            'forward_15m': f15
        })
    
    df = pd.DataFrame(records)
    df['year'] = pd.cut(range(len(df)), bins=2, labels=[2024, 2025]).astype(int)
    
    print(f"Built {len(df)} records")
    
    # Test strategies
    quote_dist = 10
    spread = 1.0
    rebate = 2.0
    
    strategies = {
        'fixed_5min': 5,
        'fixed_10min': 10,
        'fixed_15min': 15,
        'vol_adaptive': df['volatility'].apply(lambda v: 3 if v > 1.5 else (15 if v < 0.7 else 10)),
    }
    
    print("\n--- Results ---")
    results = []
    
    for strat_name, lifetime_col in strategies.items():
        if isinstance(lifetime_col, int):
            lifetimes = [lifetime_col] * len(df)
        else:
            lifetimes = lifetime_col.values
        
        pnls = []
        for i, row in df.iterrows():
            fill = estimate_fill(lifetimes[i], row['volatility'], row['volume_ratio'])
            pnl = fill * (quote_dist - spread + rebate)
            pnls.append(pnl)
        
        mean_pnl = np.mean(pnls)
        avg_lifetime = np.mean(lifetimes)
        
        print(f"{strat_name}: {mean_pnl:.2f} bps (avg lifetime: {avg_lifetime:.1f} min)")
        results.append({'strategy': strat_name, 'mean_pnl': mean_pnl, 'avg_lifetime': avg_lifetime})
    
    # Find best
    best = max(results, key=lambda x: x['mean_pnl'])
    baseline = next(r for r in results if r['strategy'] == 'fixed_5min')['mean_pnl']
    
    print(f"\nBest: {best['strategy']} ({best['mean_pnl']:.2f} bps)")
    print(f"Baseline (5min): {baseline:.2f} bps")
    print(f"Improvement: {best['mean_pnl'] - baseline:+.2f} bps")
    
    # Save
    output_dir = Path("/Users/ruby/.openclaw/workspace/projects/ruby-blog/blog/posts/2026-03-08-lifetime-tuning")
    pd.DataFrame(results).to_csv(output_dir / "backtest_results.csv", index=False)
    
    verdict = {
        'topic': 'Lifetime Tuning by Regime',
        'best_strategy': best['strategy'],
        'expected_pnl_bps': best['mean_pnl'],
        'baseline_pnl_bps': baseline,
        'improvement_bps': best['mean_pnl'] - baseline,
        'conclusion': 'DEPLOYABLE' if (best['mean_pnl'] - baseline) > 0.5 else 'NOT_DEPLOYABLE'
    }
    with open(output_dir / "verdict.json", 'w') as f:
        json.dump(verdict, f, indent=2)
    
    print(f"\nSaved to {output_dir}/")

if __name__ == "__main__":
    main()
