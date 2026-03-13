# Time-series-Prediction-Model-Exploration-Project

This repository contains two small, aligned experiments exploring time‑series models for predicting **sector times** from motorsport telemetry. A similar workflow is used in both notebooks to allow easy comparison between data types and model architectures.

---

## Le Mans vs F1 Datasets

The main differences between the two datasets are the number of relevant or usable features present in the F1 data that are not present in the Le Mans data, and the long continuous data series that comes from a 24‑hour race, which is not present in a 2‑hour race.

[WEC Dataset (contains Le Mans 2021 and 2022)](https://www.kaggle.com/datasets/tristenterracciano/fia-wec-lap-data-20122022)

[Fastf1 Dataset acquired via API](https://docs.fastf1.dev/index.html)

---

## Models Tested

### RNN Baselines
- RNN layers (easily converted to LSTM or GRU layers) with Conv1D layers as an attempt to capture characteristics of the data's variance.
- Common problems observed: 
 
  - **Prediction collapse to mean**  
  - **Random walks**  
  - **Volatility in training**  
  - **Inconsistent time‑step offset when reacting to signalled phenomena such as safety cars (model lag)**

### Transformer Regressor and TSMixer

A generic Transformer regressor is introduced as a control model for comparison against simpler RNN architectures.  
TSMixer was also introduced for the same purpose but primarily out of curiosity.

---

## Key Findings
- **Transformers and TSMixers outperform RNNs** in minimising loss and stability, but not necessarily in predicting the exact shape of the series.
- Models based on RNN layers are highly sensitive to data preparation and training parameters.
- Sector predictions can recombine into accurate lap‑time estimates, but errors from parallel sector models compound.

---

## Workflow (Common to Both Notebooks)

1. Load and clean telemetry  
2. Chronological train / validation / test split  
3. Encode categorical variables  
4. Standardise numerical features  
5. Build fixed‑length time windows (done via a predefined function)  
6. Train models with MAE loss + AdamW + early stopping + LR plateau  
7. Predict and inverse‑transform outputs  
8. Plot actual vs predicted values for each sector and the full lap


