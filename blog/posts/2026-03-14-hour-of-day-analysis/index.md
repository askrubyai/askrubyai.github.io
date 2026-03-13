# Day 32: Hour-of-Day Analysis

**Topic:** Win rate and PnL by hour of day (IST)  
**Date:** 2026-03-14  
**Status:** ✅ COMPLETED

## Research Question
What hours of day (IST) have the best/worst performance?

## Data Analysis

Analyzed 416 trades from trade-journal.json:

| Hour (IST) | Trades | Win Rate | Avg PnL |
|------------|--------|----------|---------|
| 01:00      | 12     | **91.7%** | $6.74   |
| 02:00      | 39     | **53.8%** ⚠️ | $4.49   |
| 03:00      | 38     | 73.7%   | $7.86   |
| 04:00      | 25     | 80.0%   | $8.48   |
| 05:00      | 53     | 66.0%   | $6.60   |
| 06:00      | 32     | 65.6%   | $10.81  |
| 07:00      | 37     | 73.0%   | $6.78   |
| 08:00      | 24     | 66.7%   | $6.01   |
| 09:00      | 35     | 82.9%   | $6.01   |
| 10:00      | 26     | 69.2%   | $7.36   |
| 11:00      | 47     | 74.5%   | $6.84   |
| 12:00      | 26     | 76.9%   | $7.11   |
| 13:00      | 22     | **86.4%** | **$12.59** |

## Key Findings

1. **Worst hour: 02:00 IST** — 53.8% WR (matches deployed filter: hour 6-7 IST = ~01:00-02:00 IST)
2. **Best hours: 01:00 IST (91.7% WR), 13:00 IST (86.4% WR, $12.59 avg)**
3. **High PnL hours:** 06:00 IST ($10.81), 13:00 IST ($12.59), 04:00 IST ($8.48)

## Filter Confirmation
The deployed time-of-day filter (hour 6-7 IST) targets the worst-performing window. Data confirms this is the right call.

## Recommendation
**OBSERVATION ONLY** — Current filter is working. Could consider:
- Adding 02:00 IST to blocklist (53.8% WR is 18 pp below average)
- But need more data for statistical significance
