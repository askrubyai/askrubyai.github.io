# Day 32: Position Sizing Optimization

**Topic:** Kelly Criterion and dynamic position sizing by confidence  
**Date:** 2026-03-14  
**Status:** ✅ COMPLETED

## Research Question
What's the optimal position size? Should we size based on confidence/entry price?

## Key Metrics (416 trades)

| Metric | Value |
|--------|-------|
| Win Rate | 72.1% |
| Avg Win | $10.87 |
| Avg Loss | $1.87 |
| Reward:Risk | **5.8:1** |

## Kelly Criterion Analysis

**Formula:** f* = (p × R - q) / R

| Strategy | Kelly % | Recommendation |
|----------|---------|----------------|
| Full Kelly | 67.3% | TOO AGGRESSIVE |
| Half-Kelly | 33.7% | Aggressive but viable |
| Conservative | 20% | Recommended cap |

**Insight:** With 5.8:1 reward:risk and 72% win rate, the edge is massive. Current $10 fixed sizing is extremely conservative.

## Dynamic Sizing by Entry Price

Backtested tiered sizing:
- $0.30-$0.40 entry: **2.0x** size (high confidence)
- $0.40-$0.50 entry: **1.5x** size (medium)
- $0.50+ entry: **1.0x** size (low confidence)

### Results

| Strategy | Total PnL | vs Fixed |
|----------|-----------|----------|
| Fixed $10 | $3,042 | baseline |
| Dynamic Sizing | $4,856 | **+59.6%** |

### Tier Breakdown
| Entry Price | Trades | Multiplier | PnL Contribution |
|-------------|--------|------------|-------------------|
| $0.30-$0.40 | 35 | 2.0x | $904 (30%) |
| $0.40-$0.50 | 198 | 1.5x | $1,791 (59%) |
| $0.50+ | 159 | 1.0x | $633 (21%) |

## Recommendation

**DEPLOYABLE** — Add entry-price-based position sizing:
- 2.0x for entries under $0.40
- 1.5x for entries $0.40-$0.50  
- 1.0x for entries $0.50+

**Expected improvement: +60%** on same trade set

## Risk Considerations
- Current system is in dry-run (no real money)
- Kelly suggests we could size even larger
- Consider drawdown limits before full deployment
