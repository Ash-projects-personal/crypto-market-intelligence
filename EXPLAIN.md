# How this works, in plain words

## The setup

Every row is one day. The target is simple: is the price 7 days later
higher than today, yes or no. Features come from prices and volume
before that day only. The last feature window ends before the target
window starts, so nothing leaks from the future.

The split is by time, not shuffled. Shuffling a time series puts
nearly identical neighbouring days on both sides of the split and
inflates the score. Training on 2019 to mid 2024 and testing on mid
2024 onwards is the honest version, and it is also why the score is
low: the test years are a different market regime from much of the
training data.

## Why the baselines matter

Always predict up scores 52.8% here only because BTC went up on 52.8%
of the test days. It knows nothing. Any model that cannot beat it has
not learned anything useful for this task, whatever its accuracy
looks like on its own. That comparison is the whole point of this
rebuild. The old version reported a big accuracy number with no
baseline and synthetic data, which made the number meaningless.

## The ensemble

Logistic regression gives a probability, gradient boosting gives a
probability, and the ensemble averages them. Simple, and both models
really do contribute. Averaging helped a bit over each model alone
(51.8% against 48.9% and 49.7%), which is a common ensemble effect,
but not enough to beat the naive baseline.

## What I would try next, honestly labelled as not done

- A longer feature history and features from other markets.
- Predicting return size bands instead of a yes or no direction.
- Testing on more assets and more test periods instead of one window
  of one asset. One test window can mislead in either direction.
