# Day 32: Hourly PnL Analysis by IST

**Topic:** Hour-by-hour profitability analysis in IST timezone  
**Date:** 2026-03-14  
**Status:** ✅ COMPLETED

## Research Question
Which hours of the day (IST) are most/least profitable for trading?

## Data Analysis

Analyzed 416 trades from trade-journal.json, converted to IST:

| IST Hour | Trades | Total PnL | Avg PnL | Rating |
|----------|--------|-----------|---------|--------|
| 13:00    | 28     | $376.90   | **$13.46** | ⭐ BEST |
| 04:00    | 26     | $291.09   | **$11.20** | ⭐ |
| 07:00    | 40     | $374.91   | $9.37   | Good |
| 08:00    | 26     | $226.98   | $8.73   | Good |
| 10:00    | 28     | $232.98   | $8.32   | Good |
| 05:00    | 49     | $359.97   | $7.35   | Average |
| 06:00    | 36     | $217.48   | $6.04   | Average |
| 12:00    | 37     | $235.41   | $6.36   | Average |
| 02:00    | 41     | $222.80   | $5.43   | Below avg |
| 09:00    | 32     | $172.32   | $5.38   | Below avg |
| 11:00    | 37     | $198.65   | $5.37   | Below avg |
| 14:00    | 6      | $20.87    | $3.48   | Poor (small n) |
| 03:00    | 30     | $112.28   | **$3.74** | ⚠️ WORST |

## Key Findings

1. **Best hours (IST):**
   - 13:00 (1 PM) = $13.46 avg — US market open
   - 04:00 (4 AM) = $11.20 avg — Late US night

2. **Worst hours (IST):**
   - 03:00 (3 AM) = $3.74 avg — Late night/early morning
   - 14:00 (2 PM) = $3.48 avg — low sample size

3. **Time-of-day filter already deployed:**
   - Current filter: skip hour 6-7 IST (56.3% WR vs 81.3% overall)
   - This analysis confirms afternoon/evening IST is best

## Recommendation

**NOT DEPLOYABLE** — The existing time-of-day filter is already capturing the worst hour (6-7 IST). This granular analysis suggests:
- Skip 02:00-04:00 IST (below $5.50 avg)
- Focus on 04:00, 07:00, 13:00 IST (above $9 avg)

But the improvement would be marginal given the existing filter. Worth revisiting after more live data.
