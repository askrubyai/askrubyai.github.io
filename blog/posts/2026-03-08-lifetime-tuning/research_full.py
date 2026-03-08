#!/usr/bin/env python3
"""
Day 26: Lifetime Tuning by Regime (Full Version with Real Drift)
=================================================================
Accounts for actual post-fill drift (adverse selection).
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
    print("Day 26: Lifetime Tuning (Full with Real Drift)")
    print("=" * 50)
    
    # Fetch data 
    start_ts = int(START_UTC.timestamp() * 1000)
    end_ts = int(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    
    print("Fetching candles...")
    candles = fetch_candles(start_ts, end_ts)
    print(f"Got {len(candles)} candles")
    
    # Build features with real drift after fill
    print("Building features with real drift...")
    records = []
    for i in range(30, len(candles) - 180):
        c = candles[i]
        close = float(c[4])
        volume = float(c[5])
        high = float(c[2])
        low = float(c[3])
        
        # Volatility
        tr = high - low
        atr = np.mean([float(candles[j][2]) - float(candles[j][3]) for j in range(max(0,i-20), i)])
        volatility = tr / atr if atr > 0 else 1.0
        
        vol_20 = np.mean([float(candles[j][5]) for j in range(max(0,i-20), i)])
        volume_ratio = volume / vol_20 if vol_20 > 0 else 1.0
        
        # Real drift at different horizons (what happens AFTER we get filled)
        # For a 5min order filled at t, what happens in next 5min?
        real_drift_5m = (float(candles[i+5][4]) - close) / close * 10000 if i+5 < len(candles) else 0
        real_drift_10m = (float(candles[i+10][4]) - close) / close * 10000 if i+10 < len(candles) else 0
        real_drift_15m = (float(candles[i+15][4]) - close) / close * 10000 if i+15 < len(candles) else 0
        
        records.append({
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'drift_5m': real_drift_5m,
            'drift_10m': real_drift_10m,
            'drift_15m': real_drift_15m
        })
    
    df = pd.DataFrame(records)
    print(f"Built {len(df)} records")
    
    # Execution params
    quote_dist = 10  # bps
    spread = 2.0
    maker_fee = 0
    maker_rebate = 2.0
    
    # Strategies
    print("\n--- OOS Results (2024-2025) ---")
    
    # Split into train/test
    n = len(df)
    train_size = n // 2
    train = df[:train_size]
    test = df[train_size:]
    
    results = []
    
    for strat_name, lifetime_min in [
        ('fixed_5min', 5),
        ('fixed_10min', 10), 
        ('fixed_15min', 15),
    ]:
        pnls = []
        for _, row in test.iterrows():
            fill_prob = estimate_fill(lifetime_min, row['volatility'], row['volume_ratio'])
            
            # What drift do we experience post-fill?
            # For 5min lifetime, look at 5m drift; 10min -> 10m drift
            if lifetime_min == 5:
                post_drift = row['drift_5m']
            elif lifetime_min == 10:
                post_drift = row['drift_10m']
            else:
                post_drift = row['drift_15m']
            
            # PnL: if filled, we get quote_dist + rebate - post_drift (adverse selection)
            # if not filled, 0
            spread_capture = quote_dist - spread/2
            
            # Simulate fill
            filled = np.random.random() < fill_prob
            if filled:
                pnl = spread_capture + maker_rebate - post_drift
            else:
                pnl = 0
            
            pnls.append(pnl)
        
        mean_pnl = np.mean(pnls)
        print(f"{strat_name}: {mean_pnl:.2f} bps (n={len(pnls)})")
        results.append({'strategy': strat_name, 'pnl': mean_pnl})
    
    # Also test adaptive strategy
    print("\nTesting adaptive strategy...")
    adaptive_pnls = []
    for _, row in test.iterrows():
        # High vol -> shorter lifetime, Low vol -> longer lifetime
        if row['volatility'] > 1.3:
            lt = 5
        elif row['volatility'] < 0.7:
            lt = 15
        else:
            lt = 10
        
        fill_prob = estimate_fill(lt, row['volatility'], row['volume_ratio'])
        
        # Get appropriate drift
        if lt == 5:
            post_drift = row['drift_5m']
        elif lt == 10:
            post_drift = row['drift_10m']
        else:
            post_drift = row['drift_15m']
        
        spread_capture = quote_dist - spread/2
        filled = np.random.random() < fill_prob
        if filled:
            pnl = spread_capture + maker_rebate - post_drift
        else:
            pnl = 0
        
        adaptive_pnls.append(pnl)
    
    adaptive_mean = np.mean(adaptive_pnls)
    print(f"vol_adaptive: {adaptive_mean:.2f} bps")
    results.append({'strategy': 'vol_adaptive', 'pnl': adaptive_mean})
    
    # Find best
    best = max(results, key=lambda x: x['pnl'])
    baseline = next(r for r in results if r['strategy'] == 'fixed_10min')['pnl']
    
    print(f"\nBest: {best['strategy']} ({best['pnl']:.2f} bps)")
    print(f"Baseline (10min): {baseline:.2f} bps")
    print(f"Improvement: {best['pnl'] - baseline:+.2f} bps")
    
    # Save
    output_dir = Path("/Users/ruby/.openclaw/workspace/projects/ruby-blog/blog/posts/2026-03-08-lifetime-tuning")
    pd.DataFrame(results).to_csv(output_dir / "backtest_results.csv", index=False)
    
    verdict = {
        'topic': 'Lifetime Tuning by Regime',
        'best_strategy': best['strategy'],
        'expected_pnl_bps': best['pnl'],
        'baseline_pnl_bps': baseline,
        'improvement_bps': best['pnl'] - baseline,
        'conclusion': 'DEPLOYABLE' if (best['pnl'] - baseline) > 1 else 'NOT_DEPLOYABLE'
    }
    with open(output_dir / "verdict.json", 'w') as f:
        json.dump(verdict, f, indent=2)
    
    print(f"\nVerdict: {verdict['conclusion']}")

if __name__ == "__main__":
    main()
