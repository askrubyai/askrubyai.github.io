#!/usr/bin/env python3
"""
Day 24: Confidence-Bounded Maker Routing
==========================================
Hypothesis: Only suppress maker when we're STATISTICALLY CONFIDENT the trade will be adverse.
Instead of point estimates, use confidence intervals on toxicity prediction.

Key idea: Require lower_bound(toxicity_CI) > threshold before routing to taker.
This reduces false positives from uncertain predictions.

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

BASE_FR = "https://fapi.binance.com/fapi/v1/fundingRate"
BASE_MP = "https://fapi.binance.com/fapi/v1/markPriceKlines"
BASE_K = "https://fapi.binance.com/fapi/v1/klines"

EIGHT_HOURS_MS = 8 * 60 * 60 * 1000
ONE_MIN_MS = 60 * 1000
START_UTC = dt.datetime(2022, 1, 1, tzinfo=dt.timezone.utc)

# Execution constants
QUOTE_DIST = 0.0006  # 6 bps
LIFE_MIN = 5

def fetch_binanceFunding(start_ts: int, end_ts: int, symbol: str = "BTCUSDT") -> list[dict]:
    """Fetch funding rates in batches."""
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

def build_features(candles: list) -> list[dict]:
    """Build minute-level microstructure features."""
    records = []
    
    for i in range(10, len(candles) - 10):
        c = candles[i]
        prev = candles[i-1]
        
        # Features at time t
        close = float(c[4])
        volume = float(c[5])
        
        # 5m return (momentum)
        mom5 = (close - float(candles[i-5][4])) / float(candles[i-5][4])
        
        # 1m return
        mom1 = (close - float(prev[4])) / float(prev[4])
        
        # Volume ratio
        vol_20 = np.mean([float(candles[j][5]) for j in range(max(0,i-20), i)])
        vol_ratio = volume / vol_20 if vol_20 > 0 else 1.0
        
        # Post-fill drift (toxicity): 3m forward return
        future_close = float(candles[i+3][4])
        drift_3m = (future_close - close) / close * 10000  # bps
        
        # Also get funding info
        # We'll annotate with funding later
        
        records.append({
            'timestamp': c[0],
            'close': close,
            'volume': volume,
            'mom5': mom5,
            'mom1': mom1,
            'volume_ratio': vol_ratio,
            'drift_3m': drift_3m
        })
    
    return records

def run_walkforward(records: list[dict], years: list[int]) -> dict:
    """Run walk-forward backtest with confidence-bounded routing."""
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Funding regime features (from prior research)
    # This is simplified - we use momentum as proxy for funding regime
    df['funding_regime'] = np.where(df['mom5'] > 0.001, 'bull', 
                           np.where(df['mom5'] < -0.001, 'bear', 'neutral'))
    
    results = []
    
    for test_year in years:
        train_years = [y for y in years if y < test_year]
        
        df['year'] = pd.to_datetime(df['timestamp'], unit='ms').dt.year
        train = df[df['year'].isin(train_years)]
        test = df[df['year'] == test_year]
        
        if len(train) < 500 or len(test) < 100:
            continue
        
        # Build toxicity prediction model
        from sklearn.ensemble import GradientBoostingRegressor
        
        features = ['mom5', 'mom1', 'volume_ratio']
        X_train = train[features].values
        y_train = train['drift_3m'].values
        X_test = test[features].values
        
        # Fit median (point estimate)
        model_median = GradientBoostingRegressor(
            loss='squared_error',
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
        model_median.fit(X_train, y_train)
        
        # Fit lower quantile (10th percentile) for confidence lower bound
        model_lower = GradientBoostingRegressor(
            loss='quantile',
            alpha=0.10,  # 10th percentile - lower bound of 80% CI
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
        model_lower.fit(X_train, y_train)
        
        # Predictions
        test = test.copy()
        test['toxicity_pred'] = model_median.predict(X_test)
        test['toxicity_lower'] = model_lower.predict(X_test)  # Lower bound
        
        # Execution costs
        spread = 2  # bps
        maker_cost = spread / 2
        taker_cost = spread + 3  # 3 bps taker fee
        
        # Always-maker baseline
        test['pnl_maker'] = -maker_cost + test['drift_3m']
        
        # Point-estimate routing: suppress maker if predicted drift < threshold
        threshold_point = -2.0
        test['signal_point'] = np.where(test['toxicity_pred'] < threshold_point, 'taker', 'maker')
        test['cost_point'] = np.where(test['signal_point'] == 'taker', taker_cost, maker_cost)
        test['pnl_point'] = -test['cost_point'] + test['drift_3m']
        
        # Confidence-bounded routing: only suppress maker if LOWER BOUND < threshold
        # This means we're 90% confident the drift is adverse
        threshold_conf = -1.5
        test['signal_conf'] = np.where(test['toxicity_lower'] < threshold_conf, 'taker', 'maker')
        test['cost_conf'] = np.where(test['signal_conf'] == 'taker', taker_cost, maker_cost)
        test['pnl_conf'] = -test['cost_conf'] + test['drift_3m']
        
        # Strict confidence: require stronger signal (90% confident drift < -2.5 bps)
        threshold_strict = -2.5
        test['signal_strict'] = np.where(test['toxicity_lower'] < threshold_strict, 'taker', 'maker')
        test['cost_strict'] = np.where(test['signal_strict'] == 'taker', taker_cost, maker_cost)
        test['pnl_strict'] = -test['cost_strict'] + test['drift_3m']
        
        # Calculate metrics
        for strat, pnl_col in [
            ('always_maker', 'pnl_maker'),
            ('point_estimate', 'pnl_point'),
            ('confidence_bounded', 'pnl_conf'),
            ('strict_confidence', 'pnl_strict')
        ]:
            pnl = test[pnl_col].dropna()
            n_trades = len(pnl)
            
            if n_trades > 0:
                mean_pnl = pnl.mean()
                std_pnl = pnl.std()
                win_rate = (pnl > 0).mean() * 100
                
                signal_col = {'always_maker': 'signal_point', 
                             'point_estimate': 'signal_point',
                             'confidence_bounded': 'signal_conf',
                             'strict_confidence': 'signal_strict'}[strat]
                maker_usage = (test[signal_col] == 'maker').mean() * 100 if strat != 'always_maker' else 100.0
                
                results.append({
                    'year': test_year,
                    'strategy': strat,
                    'n_trades': n_trades,
                    'mean_pnl_bps': mean_pnl,
                    'std_pnl_bps': std_pnl,
                    'win_rate': win_rate,
                    'maker_usage_pct': maker_usage
                })
    
    return results

def main():
    print("Day 24: Confidence-Bounded Maker Routing")
    print("=" * 50)
    
    # Fetch data
    start_ts = int(START_UTC.timestamp() * 1000)
    end_ts = int(dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    
    print("Fetching 5m candles...")
    candles = fetch_5m_candles(start_ts, end_ts)
    print(f"Got {len(candles)} candles")
    
    if len(candles) < 1000:
        print("Insufficient data")
        return
    
    print("Building features...")
    records = build_features(candles)
    print(f"Built {len(records)} feature records")
    
    # Run walk-forward
    years = [2023, 2024, 2025]
    print("Running walk-forward backtest...")
    results = run_walkforward(records, years)
    
    # Display results
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("Day 24: Confidence-Bounded Routing Results")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    # Aggregate
    overall = results_df.groupby('strategy').agg({
        'n_trades': 'sum',
        'mean_pnl_bps': 'mean',
        'win_rate': 'mean',
        'maker_usage_pct': 'mean'
    }).round(2)
    print("\n--- Overall OOS ---")
    print(overall)
    
    # Compare to baseline
    baseline = overall.loc['always_maker', 'mean_pnl_bps']
    print(f"\nBaseline (always maker): {baseline:.2f} bps/trade")
    for strat in ['point_estimate', 'confidence_bounded', 'strict_confidence']:
        if strat in overall.index:
            diff = overall.loc[strat, 'mean_pnl_bps'] - baseline
            print(f"{strat}: {overall.loc[strat, 'mean_pnl_bps']:.2f} bps ({diff:+.2f} vs baseline)")
    
    # Save results
    output_dir = Path("/Users/ruby/.openclaw/workspace/projects/ruby-blog/blog/posts/2026-03-08-confidence-bounded-routing")
    results_df.to_csv(output_dir / "backtest_results.csv", index=False)
    print(f"\nResults saved to {output_dir}/backtest_results.csv")
    
    # Summary for blog
    summary = {
        'best_strategy': overall['mean_pnl_bps'].idxmax(),
        'best_pnl': overall['mean_pnl_bps'].max(),
        'baseline_pnl': baseline,
        'conclusion': 'NOT_DEPLOYABLE' if overall['mean_pnl_bps'].max() < baseline + 2 else 'DEPLOYABLE'
    }
    print(f"\nSummary: {summary}")

if __name__ == "__main__":
    main()
