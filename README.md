# Freight Rate Prediction

Predicts `posted_rate` (USD) for truckload freight. Trained on `data/train_test.csv` (48,000 loads, Jan-Oct 2025),
applied to `data/validation.csv` (12,000 loads, Nov-Dec 2025).

## Run it

```bash
python -m pip install -r requirements.txt

# Option 1 - notebook (EDA + validation + model + submission + chart, ~3-4 min)
jupyter notebook Freight_Rate_Prediction.ipynb        # then Kernel > Restart & Run All
# or headless:
jupyter nbconvert --to notebook --execute --inplace Freight_Rate_Prediction.ipynb

# Option 2 - plain scripts
python src/validate.py        # optional: time-aware validation + model comparison -> outputs/validation_metrics.csv
python src/train_predict.py   # trains final model, writes validation_predictions.csv, fills data/december_chart_inputs.csv

# Scorer (also called inside the notebook)
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`score.py` validates both files and writes `scorer_results/candidate_december.png`. Tested on Python 3.12.
Run everything from the repository root (the folder containing this README).

## Layout

| Path | Purpose |
|---|---|
| `Freight_Rate_Prediction.ipynb` | Full pipeline with outputs: EDA, data-quality audit, features, validation, model, predictions, chart |
| `Freight_Rate_Prediction.html` | Static export of the executed notebook (open in any browser) |
| `Freight_Rate_Report.docx` | Report: split/validation approach + December chart |
| `LOOM_SCRIPT.md` | Talking points for the 2-3 minute Loom |
| `validation_predictions.csv` | **Final submission** (`load_id,predicted_rate`, 12,000 rows) |
| `data/` | Provided inputs; `december_chart_inputs.csv` has `predicted_rate` filled in |
| `src/features.py` | Cleaning + feature engineering |
| `src/validate.py` | Forward-chaining + unseen-city validation, compares 6 models |
| `src/train_predict.py` | Final fit on all labelled data, writes submission files |
| `score.py` | Provided scorer (unchanged) |
| `outputs/` | Metrics CSVs, `candidate_december.png`, `scorer_results/` |

## Approach in brief

* **Split:** time-based (never random). Train on the past, test on the next two months, mirroring Jan-Oct -> Nov-Dec.
  An extra unseen-city fold covers the 8 cities that appear only in the validation file.
* **Data quality:** negative weights (sign errors) -> absolute value; 47,500 weight cap flagged; missing weight /
  market_index handled (same-day median for market_index, native NaN + flag for weight); ~1.35% corrupted target
  labels handled with a robust (L1) loss.
* **Model:** bag of 6 LightGBM models, L1 loss on `log(posted_rate)`, averaged in log space.
* **December chart:** the file has no `market_index` / `quote_signal`, so each day uses that day's median value
  observed in `validation.csv` (features only, no labels).
