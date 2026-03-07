---
title: "Day 22: Probabilistic Toxicity Modeling — Predicting P(Δmaker-taker > 0)"
date: 2026-03-07
description: "Upgrade toxicity routing from binary gating to probabilistic modeling with calibrated probability estimates"
tags: [polymarket, quant, backtest, maker-taker, toxicity]
---

# Day 22: Probabilistic Toxicity Modeling

## Objective

Upgrade the toxicity routing from binary maker/taker decisions to probabilistic modeling — predict **P(Δ_maker-taker > 0 | x)** where x = microstructure features, then use calibrated probabilities to make routing decisions.

## Key Insight

Previous approaches used hard thresholds:
- "If momentum > threshold → use taker"
- "If feature score > X → suppress maker"

This loses information. Instead, predict the probability that maker will underperform taker, then decide based on expected value:

```
E[edge] = P(taker) * E[Δ|taker] + P(maker) * E[Δ|maker]
```

Only deviate from baseline (always-maker) when E[edge] exceeds uncertainty buffer.

## Methodology

### Feature Set (same as Day 21 PM)
- `mom5`: 5-minute price momentum (bps)
- `mom15`: 15-minute momentum  
- `vol5`: 5-minute volatility (bps)
- `vol15`: 15-minute volatility
- `spread`: Current bid-ask spread (bps)
- `depth_imbalance`: Order book imbalance
- `time_of_day`: Hour indicator

### Model
- Logistic regression for P(toxicity | x)
- Walk-forward training: 6 months train → 1 month test
- Calibrate probabilities using isotonic regression

### Routing Policy
```
if P(toxicity) > 0.7 and E[edge_above_threshold]:
    use taker
else:
    use maker (baseline)
```

## Expected Outcome

Probabilistic routing should outperform binary gating because:
1. Uses full distribution information, not just threshold
2. Can weight decisions by confidence
3. Naturally handles feature uncertainty

## Run ID
`day-22-prob-toxicity`
