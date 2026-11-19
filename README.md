# Mortgage Risk Replication

This project replicates the core methodology of **“Deep Learning in Mortgage Risk”** using an available Freddie Mac sample. The goal is to model monthly transitions of mortgage loans across credit-performance states and evaluate how loan history and macroeconomic factors help predict a loan’s state in the following month.

## Scope

This is a **methodological and qualitative replication**, not an exact numerical reproduction of every result in the original paper.

The original paper uses a much larger CoreLogic dataset: more than 120 million mortgages, approximately 3.5 billion monthly observations, loans originated between 1995 and 2014, and extensive loan-level, geographic, and macroeconomic covariates.

This project uses a smaller Freddie Mac sample: 50,000 loans from the 2010 origination vintage and approximately 3.4 million monthly performance observations.

The project replicates the following components:

- A monthly multi-state mortgage transition model.
- Monthly transitions between loan-performance states.
- A temporal train/validation/test split.
- Multinomial logistic regression as the 0-layer baseline.
- Path-dependent lagged state-history features.
- National mortgage-rate and prepayment-incentive features.
- A small neural network.
- An empirical one-month transition matrix.
- State-conditional out-of-sample AUC.

## Data

The project uses two Freddie Mac input files:

- `data/raw/sample_2010/sample_orig_2010.txt` — origination data.
- `data/raw/sample_2010/sample_perf_2010.txt` — monthly performance data.

The origination file contains 50,000 loans and 31 columns. The performance file contains 3,455,416 monthly records and 35 columns.

Macroeconomic data are stored in:

- `data/raw/freddie_mac_pmms_30year_fixed.csv`
- `data/raw/freddiemac_mortgage30us.csv`
- `data/raw/PMMS_history.csv`

These files provide the national mortgage rate used to construct the prepayment-incentive feature.

## Loan States

For each loan and calendar month, the project assigns one of the following states:

- `CURRENT`
- `D30`
- `D60`
- `D90_PLUS`
- `PREPAID`
- `TERMINATED_OTHER`

The current modeling pipeline does not include a separate monthly foreclosure state. Although the raw Freddie Mac performance data contain foreclosure and REO-related events, these events are extremely rare in the available sample. Therefore, they are not modeled as separate states in the main transition model.

`RA` corresponds to REO acquisition, while zero-balance code `09` corresponds to REO disposition. These are distinct events and are not combined into a single state.

## Transitions

The model considers only transitions between adjacent calendar months. For every observation, the target variable is the loan’s state in the following month.

Examples of modeled transitions include:

- `CURRENT → CURRENT`
- `CURRENT → D30`
- `D30 → D60`
- `D60 → D90_PLUS`
- `D30 → CURRENT`
- `CURRENT → PREPAID`

## Project Structure

```text
mortgage-risk-replication/
├── config.yaml
├── configs/
│   └── replication.yaml
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── README.md
├── models/
├── reports/
│   └── tables/
├── src/
├── tests/
├── requirements.txt
└── README.md
```

### Source Code

The main source files are:

```text
src/
├── 03_build_panel.py
├── 04_merge_origination.py
├── 05_build_modeling_dataset.py
├── 06_baselinemultinomiallogit.py
├── 10_add_lagged_features.py
├── 11_baselinewithlags.py
├── 13_add_macro_features.py
├── 14_train_neural_network.py
├── 15_transition_matrix.py
├── 16_state_conditional_auc.py
├── config.py
├── data.py
├── evaluate.py
├── features.py
├── states.py
└── train.py
```

### Processed Data

The main processed datasets are:

```text
data/processed/
├── performance_with_states.parquet
├── panel.parquet
├── panel_with_origination.parquet
├── modeling_dataset.parquet
├── modeling_dataset_with_lags.parquet
└── modeling_dataset_with_lags_and_macro.parquet
```

### Models

Trained models are stored in:

```text
models/
├── baselinemultinomiallogit.joblib
├── baselinewithlags.joblib
├── baselinewithlagsandmacro.joblib
├── neuralnetwork.joblib
└── neuralnetworklagsonly.joblib
```

### Reports

Model metrics, confusion matrices, the empirical transition matrix, and state-conditional AUC results are stored in:

```text
reports/tables/
```

## Models

The project trains and compares the following models:

| Model | Features | Role |
|---|---|---|
| Multinomial logistic regression | Origination and current performance features | Baseline model |
| Multinomial logistic regression | Plus lagged state history | Best-performing model |
| Multinomial logistic regression | Plus lagged history, national mortgage rate, and prepayment incentive | Tests macroeconomic features |
| Neural network | Plus lagged state history | Tests nonlinear modeling |
| Neural network | Lag-only features | Tests nonlinear modeling without macro features |

## Main Results

### Empirical Transition Matrix

The empirical one-month transition matrix qualitatively reproduces the patterns in Table 10 of the original paper.

| Transition | Original paper | This project |
|---|---:|---:|
| `CURRENT → CURRENT` | 96.7% | 98.27% |
| `CURRENT → D30` | 1.6% | 0.37% |
| `CURRENT → PREPAID` | 1.61% | 1.35% |
| `D30 → CURRENT` | 34.2% | 40.17% |
| `D30 → D30` | 44.5% | 41.98% |
| `D30 → D60` | 19.3% | 16.25% |
| `D60 → CURRENT` | 12.0% | 13.60% |
| `D60 → D90_PLUS` | 33.8% | 37.49% |
| `D90_PLUS → D90_PLUS` | 80.4% | 88.51% |
| `D90_PLUS → CURRENT` | 4.1% | 6.37% |

The project reproduces the following key patterns:

- Most current loans remain current in the following month.
- Delinquency is persistent, particularly for loans in `D90_PLUS`.
- Loans in `D30` frequently transition to `D60`.
- Loans in `D60` frequently transition to `D90_PLUS`.
- A meaningful share of delinquent loans cure and return to `CURRENT`.
- Prepayment is the main terminal outcome from the `CURRENT` state.

### State-Conditional AUC

State-conditional AUC is calculated separately for each transition `current state → next state`. For a given current state, the model evaluates whether a loan will enter a particular next state in the following month.

| Transition | AUC in this project | AUC in original paper, 0-layer model |
|---|---:|---:|
| `CURRENT → PREPAID` | 0.659 | 0.65 |
| `D30 → PREPAID` | 0.655 | 0.77 |
| `D60 → PREPAID` | 0.629 | 0.68 |
| `D90_PLUS → PREPAID` | 0.558 | 0.59 |

The closest match is `CURRENT → PREPAID`: 0.659 in this project versus 0.65 in the original paper.

### Model Comparison

| Model | Test macro ROC-AUC | Test log loss |
|---|---:|---:|
| Baseline without lag features | 0.8660 | 0.1239 |
| Lag-only multinomial logit | 0.9033 | 0.1137 |
| Lag plus macro features | 0.8660 | 0.1372 |
| Neural network with lag features | 0.8500 | 0.1202 |

The best model on this sample is the **lag-only multinomial logistic regression**. Adding lagged state history substantially improves predictive performance: test macro ROC-AUC increases from 0.866 to 0.903, while test log loss decreases from 0.124 to 0.114.

Adding the national mortgage rate and prepayment-incentive features does not improve the linear model on this sample. The small neural network also does not outperform the lag-only logistic regression.

## What Is Not Replicated

An exact numerical replication of the original paper is not possible because of differences in data availability and scope:

- Freddie Mac data are used instead of CoreLogic data.
- The sample is much smaller.
- The project uses only the 2010 origination vintage rather than loans originated from 1995 through 2014.
- The project does not include the full set of geographic and macroeconomic covariates used in the paper.
- There is no separate monthly foreclosure state in the current modeling pipeline.
- Some rare transitions have very few observations.
- The neural network is much smaller than the five-layer network used in the original paper.

## How to Run

Run the pipeline from the project root:

```bash
python -m src.03_build_panel
python -m src.04_merge_origination
python -m src.05_build_modeling_dataset
python -m src.06_baselinemultinomiallogit
python -m src.10_add_lagged_features
python -m src.11_baselinewithlags
python -m src.13_add_macro_features
python -m src.14_train_neural_network
python -m src.15_transition_matrix
python -m src.16_state_conditional_auc
```

## Key Output Files

- `reports/tables/empirical_transition_matrix.csv` — empirical one-month transition matrix.
- `reports/tables/state_conditional_auc.csv` — state-conditional AUC results.
- `reports/tables/baselinemultinomiallogitconfusionmatrix.csv` — baseline-model confusion matrix.
- `reports/tables/baselinewithlagsconfusionmatrix.csv` — lag-only logistic-regression confusion matrix.
- `reports/tables/baselinewithlagsandmacroconfusionmatrix.csv` — lag-plus-macro logistic-regression confusion matrix.
- `reports/tables/neuralnetworklagsonlyconfusionmatrix.csv` — lag-only neural-network confusion matrix.

## Conclusion

This project replicates the central methodology of the original mortgage-risk paper on an available Freddie Mac sample. It reproduces the multi-state transition framework, empirical transition dynamics, state-conditional AUC evaluation, and the comparison between linear and nonlinear models.

The results qualitatively match the original paper: current loans mostly remain current, delinquency is persistent and progresses through D30, D60, and D90+ states, and prepayment is the dominant terminal outcome from the current state. The lag-augmented multinomial logistic regression is the strongest model in this project, while the small neural network does not improve upon it.

## Future Work

The following extensions could make the project closer to the original paper, but they are outside the current replication scope:

- Add separate foreclosure and REO states.
- Add state-level unemployment, housing-price, and zip-code-level economic variables.
- Add lagged local default and prepayment rates.
- Include the full set of 272 explanatory variables.
- Train a deeper neural network and an ensemble of networks.
- Perform nonlinear sensitivity and interaction analysis.
- Implement portfolio selection and pool-level prepayment prediction.

These extensions would require additional external data sources and substantial additional modeling work.