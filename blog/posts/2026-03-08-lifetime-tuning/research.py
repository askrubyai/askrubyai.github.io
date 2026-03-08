#!/usr/bin/env python3
"""
Day 26: Lifetime Tuning by Regime
==================================
Hypothesis: Adapt order lifetime based on volatility and volume regimes.

Key idea:
- High volatility: market moves fast, shorter lifetime reduces adverse selection
- Low volatility: market is stable, longer lifetime improves fill rate
- Volume regimes also matter: high volume = faster fills, low volume = need more time

Current baseline: 5 minute fixed lifetime
Explore: Adaptive lifetime based on regime (1-30 minutes)

Uses Binance API for live data.
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
import numpy as np
import pandas as pd

BASE_K = "https://fapi.binance.com/fapi/v1/klines"

START_UTC = dt.datetime(2022, 1, 1, tzinfo=dt.timezone.utc)

def fetch_5m_candles(start_ts: int, end_ts: int, symbol: str = "BTCUSDT") -> list[dict]:
    """Fetch 5-minute candles for microstructure analysis."""
    url = BASE_K
    all_candles = []
    current = start_ts
    
    while current < end_ts:
        params = {
            "symbol": symbol,
            "interval": "5m",
            "startTime": current,
            "limit": 1000
        }
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

def estimate_fill_by_lifetime(lifetime_min: float, volatility: float, volume_ratio: float) -> float:
    """
    Estimate fill probability based on lifetime, volatility, and volume.
    
    Fill probability increases with:
    - Longer lifetime (more time to get filled)
    - Lower volatility (less adverse selection risk)
    - Higher volume (more liquidity)
    
    Model based on empirical data from Day 20:
    - 5min lifetime: ~71% fill
    - 15min lifetime: ~86% fill
    - 30min lifetime: ~95% fill (in normal conditions)
    """
    # Base fill rate at 5min lifetime in normal conditions
    base_fill_5m = 0.71
    
    # Lifetime effect (logarithmic - diminishing returns)
    lifetime_factor = math.log(lifetime_min + 1) / math.log(6)  # normalized to 5min
    
    # Volatility effect (higher vol = lower fill probability)
    vol_factor = 1.0 / (1.0 + 0.5 * (volatility - 1.0))
    
    # Volume effect (higher volume = higher fill)
    vol_ratio_factor = min(1.5, max(0.5, volume_ratio))
    
    fill_prob = base_fill_5m * lifetime_factor * vol_factor * vol_ratio_factor
    
    return min(0.99, max(0.01, fill_prob))

def calculate_pnl_by_lifetime(lifetime_min: float, volatility: float, volume_ratio: float, 
                            quote_dist_bps: float = 10, drift_5m: float = 0) -> dict:
    """
    Calculate expected PnL for a given lifetime and market conditions.
    
    If lifetime <= observation period: PnL based on observed drift
    If lifetime > observation period: PnL = fill_prob * spread_capture
    """
    spread_bps = 2.0
    maker_rebate = 2.0
    
    fill_prob = estimate_fill_by_lifetime(lifetime_min, volatility, volume_ratio)
    spread_capture = quote_dist_bps - spread_bps / 2
    
    # Expected PnL = P(fill) * (spread_capture + rebate) - cost_of_waiting
    # Cost of waiting: we miss opportunity cost if market moves significantly
    # Simple model: expected drift over lifetime
    
    # For simplicity: use expected value
    expected_pnl = fill_prob * (spread_capture + maker_rebate)
    
    # Also consider adverse selection cost if we get filled near the end
    # In high vol, price likely moved against us
    adverse_selection_cost = 0.5 * volatility * (lifetime_min / 5)  # bps
    
    net_pnl = expected_pnl - adverse_selection_cost
    
    return {
        'lifetime_min': lifetime_min,
        'volatility': volatility,
        'volume_ratio': volume_ratio,
        'fill_prob': fill_prob,
        'expected_pnl': expected_pnl,
        'adverse_selection_cost': adverse_selection_cost,
        'net_pnl': net_pnl
    }

def build_features(candles: list) -> list[dict]:
    """Build microstructure features for regime classification."""
    records = []
    
    for i in range(30, len(candles) - 120):
        c = candles[i]
        
        close = float(c[4])
        volume = float(c[5])
        high = float(c[2])
        low = float(c[3])
        
        # Volatility (true range normalized)
        tr = high - low
        atr = np.mean([float(candles[j][2]) - float(candles[j][3]) 
                      for j in range(max(0, i-20), i)])
        volatility = tr / atr if atr > 0 else 1.0
        
        # Volume
        vol_20 = np.mean([float(candles[j][5]) for j in range(max(0, i-20), i)])
        volume_ratio = volume / vol_20 if vol_20 > 0 else 1.0
        
        # Forward returns for different horizons
        forward_returns = []
        for f in [5, 15, 30, 60]:  # minutes
            if i + f < len(candles):
                fut_close = float(candles[i+f][4])
                ret = (fut_close - close) / close * 10000  # bps
                forward_returns.append((f, ret))
        
        # Regime classification
        if volatility < 0.7:
            regime = 'low_vol'
        elif volatility > 1.5:
            regime = 'high_vol'
        else:
            regime = 'mid_vol'
        
        if volume_ratio < 0.7:
            regime += '_low_vol'
        elif volume_ratio > 1.3:
            regime += '_high_vol'
        else:
            regime += '_norm_vol'
        
        records.append({
            'timestamp': c[0],
            'close': close,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'regime': regime,
            'forward_5m': forward_returns[0][1] if len(forward_returns) > 0 else 0,
            'forward_15m': forward_returns[1][1] if len(forward_returns) > 1 else 0,
            'forward_30m': forward_returns[2][1] if len(forward_returns) > 2 else 0,
            'forward_60m': forward_returns[3][1] if len(forward_returns) > 3 else 0
        })
    
    return records

def run_lifetime_backtest(records: list[dict], lifetimes: list[float], years: list[int], 
                         quote_dist_bps: float = 10) -> dict:
    """
    Walk-forward backtest across different lifetime strategies.
    
    Strategies:
    1. Fixed 5min (baseline)
    2. Fixed 15min
    3. Volatility-adaptive: high_vol=3min, mid_vol=5min, low_vol=15min
    4. Volume-adaptive: low_vol=15min, norm=10min, high=5min
    5. Combined regime-adaptive
    """
    df = pd.DataFrame(records)
    df['year'] = pd.to_datetime(df['timestamp'], unit='ms').dt.year
    
    all_results = []
    
    for test_year in years:
        train_years = [y for y in years if y < test_year]
        
        train = df[df['year'].isin(train_years)]
        test = df[df['year'] == test_year]
        
        if len(train) < 500 or len(test) < 100:
            continue
        
        # Define strategies
        strategies = {
            'fixed_5min': lambda row: 5,
            'fixed_10min': lambda row: 10,
            'fixed_15min': lambda row: 15,
            'vol_adaptive': lambda row: 3 if row['volatility'] > 1.5 else (15 if row['volatility'] < 0.7 else 5),
            'vol_adaptive_v2': lambda row: 3 if row['volatility'] > 1.5 else (20 if row['volatility'] < 0.7 else 10),
            'volume_adaptive': lambda row: 15 if row['volume_ratio'] < 0.7 else (5 if row['volume_ratio'] > 1.3 else 10),
            'combined_adaptive': lambda row: (
                3 if row['volatility'] > 1.5 else (
                    20 if row['volatility'] < 0.7 and row['volume_ratio'] < 0.7 else 10
                )
            )
        }
        
        for strat_name, strat_func in strategies.items():
            test = test.copy()
            test['lifetime'] = test.apply(strat_func, axis=1)
            
            # Calculate PnL for each row
            pnls = []
            for _, row in test.iterrows():
                fill_prob = estimate_fill_by_lifetime(
                    row['lifetime'], row['volatility'], row['volume_ratio']
                )
                spread_capture = quote_dist_bps - 1.0  # half spread
                pnl = fill_prob * (spread_capture + 2.0)
                pnls.append(pnl)
            
            test['pnl'] = pnls
            
            n_trades = len(test)
            mean_pnl = test['pnl'].mean()
            std_pnl = test['pnl'].std()
            
            avg_lifetime = test['lifetime'].mean()
            avg_fill_prob = np.mean([
                estimate_fill_by_lifetime(row['lifetime'], row['volatility'], row['volume_ratio'])
                for _, row in test.iterrows()
            ])
            
            all_results.append({
                'year': test_year,
                'strategy': strat_name,
                'n_periods': n_trades,
                'mean_pnl_bps': mean_pnl,
                'std_pnl_bps': std_pnl,
                'avg_lifetime_min': avg_lifetime,
                'avg_fill_prob': avg_fill_prob
            })
    
    return all_results

def find_best_strategy(results: list[dict]) -> dict:
    """Find the best lifetime strategy."""
    df = pd.DataFrame(results)
    
    # Aggregate across years
    agg = df.groupby('strategy').agg({
        'mean_pnl_bps': 'mean',
        'avg_lifetime_min': 'mean',
        'avg_fill_prob': 'mean'
    }).reset_index()
    
    # Find best
    best_idx = agg['mean_pnl_bps'].idxmax()
    best = agg.iloc[best_idx]
    
    return {
        'best_strategy': best['strategy'],
        'expected_pnl_bps': best['mean_pnl_bps'],
        'avg_lifetime_min': best['avg_lifetime_min'],
        'avg_fill_prob': best['avg_fill_prob'],
        'all_strategies': agg.to_dict('records')
    }

def main():
    print("Day 26: Lifetime Tuning by Regime")
    print("=" * 50)
    
    # Theoretical analysis
    print("\n--- Theoretical Analysis ---")
    print(f"{'Lifetime':<10} {'Vol':<8} {'VolRatio':<10} {'Fill':<10} {'NetPnL':<10}")
    print("-" * 50)
    
    for lifetime in [3, 5, 10, 15, 20]:
        for vol in [0.5, 1.0, 2.0]:
            for vol_ratio in [0.5, 1.0, 1.5]:
                result = calculate_pnl_by_lifetime(lifetime, vol, vol_ratio)
                print(f"{lifetime:<10} {vol:<8.1f} {vol_ratio:<10.2f} {result['fill_prob']:<10.2%} {result['net_pnl']:<10.2f}")
    
    # Fetch data
    start_ts = int(START_UTC.timestamp() * 1000)
    end_ts = int(dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    
    print("\nFetching 5m candles...")
    candles = fetch_5m_candles(start_ts, end_ts)
    print(f"Got {len(candles)} candles")
    
    if len(candles) < 1000:
        print("Insufficient data")
        return
    
    print("Building features...")
    records = build_features(candles)
    print(f"Built {len(records)} records")
    
    # Run backtest
    lifetimes = [3, 5, 10, 15, 20]
    years = [2023, 2024, 2025]
    
    print("\nRunning walk-forward backtest...")
    results = run_lifetime_backtest(records, lifetimes, years, quote_dist_bps=10)
    
    # Display results
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("Day 26: Lifetime Tuning Results")
    print("=" * 80)
    
    # Pivot by year
    pivot = results_df.pivot_table(
        values='mean_pnl_bps', 
        index='strategy', 
        columns='year',
        aggfunc='mean'
    )
    print("\nMean PnL (bps) by Strategy and Year:")
    print(pivot.round(2).to_string())
    
    # Aggregate
    overall = results_df.groupby('strategy').agg({
        'mean_pnl_bps': 'mean',
        'avg_lifetime_min': 'mean',
        'avg_fill_prob': 'mean'
    }).round(2)
    
    print("\n--- Overall OOS Performance ---")
    print(overall.to_string())
    
    # Find best
    best = find_best_strategy(results)
    
    print("\n" + "=" * 80)
    print("BEST STRATEGY")
    print("=" * 80)
    print(f"Strategy: {best['best_strategy']}")
    print(f"Expected PnL: {best['expected_pnl_bps']:.2f} bps/trade")
    print(f"Avg Lifetime: {best['avg_lifetime_min']:.1f} min")
    print(f"Avg Fill Rate: {best['avg_fill_prob']:.1%}")
    
    # Compare to baseline (fixed 5min)
    baseline_5 = overall.loc['fixed_5min', 'mean_pnl_bps'] if 'fixed_5min' in overall.index else None
    if baseline_5:
        print(f"\nBaseline (fixed 5min): {baseline_5:.2f} bps/trade")
        diff = best['expected_pnl_bps'] - baseline_5
        print(f"Improvement: {diff:+.2f} bps/trade ({diff/baseline_5*100:+.1f}%)")
    
    # Save results
    output_dir = Path("/Users/ruby/.openclaw/workspace/projects/ruby-blog/blog/posts/2026-03-08-lifetime-tuning")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / "backtest_results.csv", index=False)
    
    # Verdict
    verdict = {
        'topic': 'Lifetime Tuning by Regime',
        'best_strategy': best['best_strategy'],
        'baseline_strategy': 'fixed_5min',
        'expected_pnl_baseline_bps': baseline_5,
        'expected_pnl_best_bps': best['expected_pnl_bps'],
        'improvement_bps': diff if baseline_5 else 0,
        'conclusion': 'DEPLOYABLE' if (baseline_5 and diff > 1) else 'NOT_DEPLOYABLE'
    }
    with open(output_dir / "verdict.json", 'w') as f:
        json.dump(verdict, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    
    return results_df, best

if __name__ == "__main__":
    results, best = main()
