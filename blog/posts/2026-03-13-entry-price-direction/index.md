# Day 31: Entry Price & Direction Analysis

**Topic:** Entry price sweet spots and directional bias analysis  
**Date:** 2026-03-13  
**Status:** ✅ COMPLETED

## Research Question
What entry price ranges and trade directions perform best?

## Data Analysis

### Entry Price Sweet Spot
Analyzed 416 trades from trade-journal.json:

| Entry Price | Trades | Win Rate | Avg PnL |
|-------------|--------|----------|---------|
| $0.30-$0.40 | 35     | 74.3%    | **$12.92** |
| $0.40-$0.50 | 198    | 71.7%    | $6.03   |
| $0.50-$0.60 | 159    | 70.4%    | $3.98   |

**Finding:** Lower entry prices ($0.30-$0.40) have **2-3x higher PnL** per trade. However, liquidity may be lower at these prices.

### Directional Bias
| Direction | Trades | Win Rate |
|-----------|--------|----------|
| UP        | 181    | **75.1%** |
| DOWN      | 235    | 69.8%    |

**Finding:** UP trades win 5.3 pp more often than DOWN trades.

### Stop-Loss Analysis
Current stop-loss triggers at ~18-24% loss. Analysis shows:
- 5% stop: stops 173 trades (41%)
- 10% stop: stops 157 trades (38%)
- Current system: stops 26 trades (6%)

**Finding:** Tighter stops reduce total exposure but may cut winning trades early. Current system lets winners run (avg hold 2.6 min, 72% win rate).

## Key Insights
1. **Entry price matters** — lower entry = higher PnL (confirmed earlier research)
2. **UP bias** — 75% win rate vs 70% for DOWN
3. **Let winners run** — current system is correct (doesn't use tight stops)

## Recommendation
**NOT DEPLOYABLE** — Current system is already well-optimized. The entry price finding confirms existing $0.30-$0.55 sweet spot from prior research. Directional bias is marginal (5 pp) and may be regime-dependent.
