# Day 32: Position Sizing & Kelly Criterion Analysis

**Topic:** Optimal position sizing for Polymarket 15-min trading  
**Date:** 2026-03-14  
**Status:** ✅ COMPLETED

## Research Question
What position sizing maximizes risk-adjusted returns?

## Data Analysis

Analyzed 416 trades from trade-journal.json:

| Metric | Value |
|--------|-------|
| Win rate | 72.1% |
| Avg win | $10.87 |
| Avg loss | $1.87 |
| W/L ratio | 5.80 |

### Kelly Criterion
- **Full Kelly**: 67.3% of bankroll (impractical)
- **Half Kelly**: 33.7% of bankroll (aggressive but survivable)

### Simulated Performance by Position Size

| Position Size | Final Balance | Max Drawdown |
|--------------|--------------|--------------|
| 5% | $70.85 | - |
| 10% | $131.71 | 4.3% |
| 20% | $253.41 | 5.6% |
| 30% | $375.12 | 6.2% |

## Key Findings

1. **Aggressive sizing works** — Even 20-30% of bankroll yields <6% max drawdown
2. **Current system uses ~$5-10 per trade** — Conservative vs Kelly suggests 10-20% is optimal
3. **W/L ratio is exceptional** (5.8:1) — Allows aggressive sizing
4. **Kelly is unrealistic** — 67% per trade is too risky; 10-20% is prudent maximum

## Recommendation

**NOT DEPLOYABLE** — Current sizing is conservative and appropriate. The math suggests we *could* size larger, but:
- Black swan risk (market resoluton errors, smart contract bugs)
- Execution constraints (minimum order sizes, slippage)
- Psychological comfort

Current 1-2% of bankroll per trade is safe. For live trading with real money, recommend 5-10% max.

## Next Steps
- Consider live trading when comfortable with position sizing
- Current dry-run validates filters; ready to transition to paper/live
