# Explainable Transformer-Based Renewable Energy Forecasting

Code for the dissertation proposal *"Explainable Transformer-Based Renewable
Energy Forecasting for Smart Grid Decision Support"* (Areef Ahammad Shaik,
MSc AI, UEL, CN7000). Implements Objectives 2–6: data pipeline, LSTM/GRU
baselines, Informer/Autoformer/PatchTST forecasters, MAE/RMSE/MAPE/R²
evaluation across forecast horizons, and SHAP-based feature importance.

## Setup

```bash
pip install -r requirements.txt
```

## Before running anything for real

Replace the two files under `data/raw/` with your finalised dataset export
(generation records + meteorological records). Until then,
`scripts/generate_synthetic_dataset.py` writes a small sinusoidal dummy
dataset there purely so the pipeline can be exercised end-to-end — **it is
not real solar/wind/weather data**, only useful for checking the code runs.

```bash
python scripts/generate_synthetic_dataset.py   # optional, dummy data only
python -m src.cli train    --model lstm --data-config configs/data.yaml --model-config configs/lstm.yaml
python -m src.cli evaluate --model lstm --data-config configs/data.yaml --model-config configs/lstm.yaml
python -m src.cli explain  --model lstm --data-config configs/data.yaml --model-config configs/lstm.yaml
python -m src.cli compare  --models lstm gru informer autoformer patchtst --save-plot
```

Model names: `lstm`, `gru`, `informer`, `autoformer`, `patchtst`. Each has
its own config in `configs/`.

## Outstanding: research ethics form

Not something code can fix — the Study Guide (Section 19) implies a Research
Ethics Form A/B self-certification is required even for desk-based research
with no human participants, and only the human-participants form (Form B)
was present in the project folder. Confirm with your supervisor whether
Form A still needs submitting before the dissertation is finalised.

For everything else — the plan-review fixes (TensorFlow inclusion,
normalization, data alignment, marking scheme, code-appendix packaging) and
the real bugs found and fixed during integration testing (MAPE, SHAP output
shape, Autoformer performance, Informer's missing positional encoding) —
see [`DEVELOPMENT_NOTES.md`](DEVELOPMENT_NOTES.md).

## Repository structure

```
configs/            data.yaml + one YAML per model
src/data/            ingest -> align -> clean -> scale -> features -> windowing
src/models/          base interface, baselines/, transformers/, factory.py
src/training/        trainer (shared PyTorch loop + LR warm-up), seed, Optuna tuner
src/evaluation/       MAE/RMSE/MAPE/R², per-horizon breakdown, comparison tables
src/explain/          SHAP feature importance (torch + Keras)
src/pipeline.py       wires the data steps together in the correct order
src/cli.py            train / evaluate / explain / tune / compare
tests/                unit tests (metrics, scaling leakage, windowing, model smoke tests)
scripts/generate_synthetic_dataset.py   dummy data for pipeline testing only
```
