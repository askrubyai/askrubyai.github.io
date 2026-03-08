#!/usr/bin/env python3
"""
Day 27: Selective Participation
================================
Hypothesis: Only trade when funding regime signal is strong.
Skip trading during neutral/low-signal regimes to avoid noise.

Key insight: Not all funding periods are equally tradeable.
Some regimes have better risk/reward than others.

Uses Binance API for live data.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import statistics
import time
from pathlib import Path

import requests
import numpy as np
import pandas as pd

BASE_FR = "https://fapi.binance.com/fapi/v1/fundingRate"
BASE_K = "https://fapi.binance.com/fapi/v1/klines"

START_UTC = dt.datetime(2022, 1, 1, tzinfo=dt.timezone.utc)

def fetch_binanceFunding(start_ts: int, end_ts: int, symbol: str = "BTCUSDT") -> list[dict]:
    """Fetch funding rates."""
    url = BASE_FR
    all_rates = []
    current = start_ts
    
    while current < end_ts:
        params = {"symbol": symbol, "startTime": current, "limit": 1000}
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if not data:
                break
            all_rates.extend(data)
            current = data[-1]["fundingTime"] + 1
            time.sleep(0.2)
        except Exception as e:
            print(f"Funding fetch error: {e}")
            break
    
    return all_rates

def fetch_5m_candles(start_ts: int, end_ts: int, symbol: str = "BTCUSDT") -> list[dict]:
    """Fetch 5-minute candles."""
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

def build_regime_features(candles: list, funding_rates: list) -> list[dict]:
    """Build funding regime features using percentile-based classification."""
    
    # Create funding rate lookup
    funding_dict = {}
    for f in funding_rates:
        funding_dict[f['fundingTime']] = float(f['fundingRate'])
    
    # Get all funding rates for percentile calculation
    all_funding = [float(f['fundingRate']) for f in funding_rates]
    p25 = np.percentile(all_funding, 25)
    p75 = np.percentile(all_funding, 75)
    
    print(f"Funding percentiles - 25th: {p25*100:.4f}%, 75th: {p75*100:.4f}%")
    
    # Group candles by 8-hour funding intervals
    records = []
    funding_interval_ms = 8 * 60 * 60 * 1000
    
    for i in range(20, len(candles) - 60):
        c = candles[i]
        ts = c[0]
        
        # Find current funding period
        funding_ts = (ts // funding_interval_ms) * funding_interval_ms
        funding_rate = funding_dict.get(funding_ts, 0)
        
        close = float(c[4])
        
        # Features
        mom5 = (close - float(candles[i-5][4])) / float(candles[i-5][4])
        mom1 = (close - float(candles[i-1][4])) / float(candles[i-1][4])
        
        # Volatility
        vol_20 = np.std([float(candles[j][4]) for j in range(max(0, i-20), i)])
        vol_current = np.std([float(candles[j][4]) for j in range(max(0, i-5), i)])
        vol_ratio = vol_current / vol_20 if vol_20 > 0 else 1.0
        
        # Forward return (5m)
        if i + 5 < len(candles):
            future_close = float(candles[i+5][4])
            drift_5m = (future_close - close) / close * 10000  # bps
        else:
            drift_5m = 0
        
        # Funding regime classification (percentile-based)
        if funding_rate > p75:
            regime = 'high'
        elif funding_rate < p25:
            regime = 'low'
        else:
            regime = 'neutral'
        
        # Absolute funding strength
        funding_abs = abs(funding_rate) * 10000  # in bps
        
        records.append({
            'timestamp': ts,
            'close': close,
            'funding_rate': funding_rate,
            'funding_abs': funding_abs,
            'regime': regime,
            'mom5': mom5,
            'mom1': mom1,
            'vol_ratio': vol_ratio,
            'drift_5m': drift_5m
        })
    
    return records

def run_selective_participation_backtest(records: list[dict], years: list[int], 
                                         threshold_pct: float = 0.25) -> dict:
    """
    Test selective participation strategies.
    """
    df = pd.DataFrame(records)
    df['year'] = pd.to_datetime(df['timestamp'], unit='ms').dt.year
    
    # Quote distance: 10 bps (from Day 25)
    quote_dist = 10
    spread_bps = 2.0
    maker_rebate = 2.0
    
    # Fill probability at 10 bps
    fill_base = 1.0 / (1.0 + math.exp(0.15 * (quote_dist - 8.0)))
    
    # Adjust by volatility
    df['vol_adj'] = np.clip(df['vol_ratio'], 0.5, 2.0)
    df['fill_prob'] = fill_base / df['vol_adj']
    df['fill_prob'] = df['fill_prob'].clip(0.01, 0.99)
    
    # PnL if filled
    spread_capture = quote_dist - spread_bps / 2
    pnl_if_filled = spread_capture + maker_rebate
    
    # Expected PnL
    df['expected_pnl'] = df['fill_prob'] * pnl_if_filled
    
    # Realized PnL (simulated)
    np.random.seed(42)
    df['filled'] = np.random.random(len(df)) < df['fill_prob']
    df['realized_pnl'] = np.where(df['filled'], pnl_if_filled, 0)
    
    all_results = []
    
    for test_year in years:
        test = df[df['year'] == test_year]
        
        if len(test) < 100:
            continue
        
        # Strategy 1: Always trade
        always_pnl = test['expected_pnl'].mean()
        always_n = len(test)
        
        # Strategy 2: Only high funding (bear regime)
        high_fund = test[test['regime'] == 'high']
        high_pnl = high_fund['expected_pnl'].mean() if len(high_fund) > 0 else None
        high_part = len(high_fund) / len(test) if len(high_fund) > 0 else 0
        
        # Strategy 3: Only low funding (bull regime)
        low_fund = test[test['regime'] == 'low']
        low_pnl = low_fund['expected_pnl'].mean() if len(low_fund) > 0 else None
        low_part = len(low_fund) / len(test) if len(low_fund) > 0 else 0
        
        # Strategy 4: Skip neutral - only trade in extreme regimes
        extreme = test[test['regime'].isin(['high', 'low'])]
        extreme_pnl = extreme['expected_pnl'].mean() if len(extreme) > 0 else None
        extreme_part = len(extreme) / len(test) if len(extreme) > 0 else 0
        
        # Strategy 5: Momentum-based regime filter
        # Only trade when momentum aligns with funding regime
        aligned = test[
            ((test['regime'] == 'low') & (test['mom1'] > 0)) |
            ((test['regime'] == 'high') & (test['mom1'] < 0))
        ]
        aligned_pnl = aligned['expected_pnl'].mean() if len(aligned) > 0 else None
        aligned_part = len(aligned) / len(test) if len(aligned) > 0 else 0
        
        # Strategy 6: High volatility only
        high_vol = test[test['vol_ratio'] > 1.2]
        high_vol_pnl = high_vol['expected_pnl'].mean() if len(high_vol) > 0 else None
        high_vol_part = len(high_vol) / len(test) if len(high_vol) > 0 else 0
        
        # Strategy 7: Low volatility only
        low_vol = test[test['vol_ratio'] < 0.8]
        low_vol_pnl = low_vol['expected_pnl'].mean() if len(low_vol) > 0 else None
        low_vol_part = len(low_vol) / len(test) if len(low_vol) > 0 else 0
        
        # Always trade baseline
        all_results.append({
            'year': test_year,
            'strategy': 'always_trade',
            'n_periods': always_n,
            'expected_pnl_bps': always_pnl,
            'participation_rate': 1.0
        })
        
        if high_pnl is not None:
            all_results.append({
                'year': test_year,
                'strategy': 'high_funding_only',
                'n_periods': len(high_fund),
                'expected_pnl_bps': high_pnl,
                'participation_rate': high_part
            })
        
        if low_pnl is not None:
            all_results.append({
                'year': test_year,
                'strategy': 'low_funding_only',
                'n_periods': len(low_fund),
                'expected_pnl_bps': low_pnl,
                'participation_rate': low_part
            })
        
        if extreme_pnl is not None:
            all_results.append({
                'year': test_year,
                'strategy': 'extreme_regimes_only',
                'n_periods': len(extreme),
                'expected_pnl_bps': extreme_pnl,
                'participation_rate': extreme_part
            })
        
        if aligned_pnl is not None:
            all_results.append({
                'year': test_year,
                'strategy': 'regime_momentum_aligned',
                'n_periods': len(aligned),
                'expected_pnl_bps': aligned_pnl,
                'participation_rate': aligned_part
            })
        
        if high_vol_pnl is not None:
            all_results.append({
                'year': test_year,
                'strategy': 'high_volatility_only',
                'n_periods': len(high_vol),
                'expected_pnl_bps': high_vol_pnl,
                'participation_rate': high_vol_part
            })
        
        if low_vol_pnl is not None:
            all_results.append({
                'year': test_year,
                'strategy': 'low_volatility_only',
                'n_periods': len(low_vol),
                'expected_pnl_bps': low_vol_pnl,
                'participation_rate': low_vol_part
            })
    
    return all_results

def main():
    print("Day 27: Selective Participation")
    print("=" * 50)
    
    # Fetch data
    start_ts = int(START_UTC.timestamp() * 1000)
    end_ts = int(dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    
    print("Fetching funding rates...")
    funding_rates = fetch_binanceFunding(start_ts, end_ts)
    print(f"Got {len(funding_rates)} funding rates")
    
    print("Fetching 5m candles...")
    candles = fetch_5m_candles(start_ts, end_ts)
    print(f"Got {len(candles)} candles")
    
    if len(candles) < 1000 or len(funding_rates) < 10:
        print("Insufficient data")
        return
    
    print("Building regime features...")
    records = build_regime_features(candles, funding_rates)
    print(f"Built {len(records)} records")
    
    # Analyze funding regime distribution
    df = pd.DataFrame(records)
    print("\n--- Funding Regime Distribution ---")
    regime_dist = df['regime'].value_counts(normalize=True)
    print(regime_dist)
    
    print("\n--- Funding Rate Stats ---")
    print(f"Mean: {df['funding_rate'].mean()*100:.4f}%")
    print(f"Std: {df['funding_rate'].std()*100:.4f}%")
    print(f"Min: {df['funding_rate'].min()*100:.4f}%")
    print(f"Max: {df['funding_rate'].max()*100:.4f}%")
    
    # Run backtest
    years = [2023, 2024, 2025]
    print("\nRunning selective participation backtest...")
    results = run_selective_participation_backtest(records, years, threshold_pct=0.25)
    
    # Display results
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("Day 27: Selective Participation Results")
    print("=" * 80)
    
    # Pivot by strategy
    pivot = results_df.pivot_table(
        values='expected_pnl_bps',
        index='strategy',
        columns='year',
        aggfunc='mean'
    )
    print("\nExpected PnL (bps) by Strategy:")
    print(pivot.round(2).to_string())
    
    # Aggregate
    overall = results_df.groupby('strategy').agg({
        'expected_pnl_bps': 'mean',
        'n_periods': 'sum',
        'participation_rate': 'mean'
    }).round(2)
    
    print("\n--- Overall OOS Performance ---")
    print(overall.to_string())
    
    # Compare to baseline
    baseline = overall.loc['always_trade', 'expected_pnl_bps']
    print(f"\nBaseline (always trade): {baseline:.2f} bps/trade")
    
    print("\n--- Strategy Comparison ---")
    for strat in overall.index:
        if strat != 'always_trade':
            strat_pnl = overall.loc[strat, 'expected_pnl_bps']
            participation = overall.loc[strat, 'participation_rate']
            diff = strat_pnl - baseline
            print(f"{strat}: {strat_pnl:.2f} bps ({diff:+.2f} vs baseline), participation: {participation:.1%}")
    
    # Find best selective strategy (only if there are other strategies)
    selective_strategies = overall[overall.index != 'always_trade']
    if len(selective_strategies) > 0:
        best_selective = selective_strategies['expected_pnl_bps'].idxmax()
        best_selective_pnl = overall.loc[best_selective, 'expected_pnl_bps']
    else:
        best_selective = 'always_trade'
        best_selective_pnl = baseline
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(f"Best strategy: {best_selective}")
    print(f"Expected PnL: {best_selective_pnl:.2f} bps")
    print(f"Baseline: {baseline:.2f} bps")
    print(f"Improvement: {best_selective_pnl - baseline:+.2f} bps")
    
    if best_selective_pnl > baseline + 1:
        verdict = "DEPLOYABLE"
    else:
        verdict = "NOT_DEPLOYABLE"
    print(f"\nVerdict: {verdict}")
    
    # Save results
    output_dir = Path("/Users/ruby/.openclaw/workspace/projects/ruby-blog/blog/posts/2026-03-08-selective-participation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / "backtest_results.csv", index=False)
    
    # Save regime stats
    regime_stats = {
        'high_funding_pct': float(regime_dist.get('high', 0)),
        'low_funding_pct': float(regime_dist.get('low', 0)),
        'neutral_funding_pct': float(regime_dist.get('neutral', 0)),
        'mean_funding_bps': float(df['funding_rate'].mean() * 10000),
        'std_funding_bps': float(df['funding_rate'].std() * 10000)
    }
    with open(output_dir / "regime_stats.json", 'w') as f:
        json.dump(regime_stats, f, indent=2)
    
    verdict_data = {
        'topic': 'Selective Participation',
        'best_strategy': best_selective,
        'best_pnl_bps': float(best_selective_pnl),
        'baseline_pnl_bps': float(baseline),
        'improvement_bps': float(best_selective_pnl - baseline),
        'conclusion': verdict
    }
    with open(output_dir / "verdict.json", 'w') as f:
        json.dump(verdict_data, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    
    return results_df, overall

if __name__ == "__main__":
    results, overall = main()
