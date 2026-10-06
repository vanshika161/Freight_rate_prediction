# Loom script (about 2:30)

**0:00 - Intro (10s)** "I built a LightGBM model that predicts freight rates. I'll cover the data, quality issues, model choice, validation and the code."

**0:10 - Key findings (30s)** Rate is driven mostly by distance (about 97% rank correlation), with rate-per-mile falling from ~$2.9 on short hauls to ~$1.9 on long ones. Reefer and flatbed cost more than dry van. quote_signal has a U-shaped effect. market_index moves daily and drifts: 1.08 in train vs 0.93 in validation.

**0:40 - Data-quality issues (40s)** Negative weights are sign errors, so I use the absolute value. 1,204 weights sit at a 47,500 cap, so I flag them. Missing weight and market_index are handled with NaN or a flag, and a same-day median for market_index. About 1.3% of target labels are corrupted (about 3.6x or 0.3x the normal rate-per-mile), so I use a robust L1 loss. Validation has 8 unseen cities, so I model with lat/lon, distance and bearing instead of city IDs.

**1:20 - Model choice (25s)** Compared baseline, ridge and LightGBM with L2, Huber and L1 loss. Tree models capture the non-linear distance, quote_signal and equipment effects. L1 on log(rate) won on Sep-Oct, and the final model is a bag of 6.

**1:45 - Split and validation (30s)** Validation is the future, so no random K-fold. Fold A: train Jan-Jun, test Jul-Aug. Fold B: train Jan-Aug, test Sep-Oct. Fold C: unseen cities. Result on fold B: 3.6% MAPE on clean loads, 5.9% with corrupted labels included.

**2:15 - Code walkthrough (25s)** Show `src/features.py` (cleaning), `src/validate.py` (folds), `src/train_predict.py` (final fit and December inputs), then run `score.py` and show the chart.
