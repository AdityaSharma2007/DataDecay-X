# DataDecay-X

## Predictive Model Reliability Forecasting and Proactive Synthetic Remediation under Temporal Distribution Shift

> **DataDecay-X** is an applied machine-learning and MLOps-oriented research prototype for studying how machine-learning model reliability changes under temporal distribution shift, forecasting future reliability degradation, estimating **Time-to-Decay (TTD)**, and evaluating controlled synthetic-data-based remediation through chronological future validation.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Problem Statement](#2-problem-statement)
- [3. Motivation](#3-motivation)
- [4. Central Research Proposition](#4-central-research-proposition)
- [5. Research Gap](#5-research-gap)
- [6. Objectives](#6-objectives)
- [7. Research Questions](#7-research-questions)
- [8. Scope](#8-scope)
- [9. End-to-End Workflow](#9-end-to-end-workflow)
- [10. System Architecture](#10-system-architecture)
- [11. Major Components](#11-major-components)
- [12. Temporal Drift Simulator](#12-temporal-drift-simulator)
- [13. Drift Detection](#13-drift-detection)
- [14. Model Reliability Monitoring](#14-model-reliability-monitoring)
- [15. Reliability Forecasting](#15-reliability-forecasting)
- [16. Time-to-Decay](#16-time-to-decay)
- [17. Future-Regime Estimation](#17-future-regime-estimation)
- [18. Synthetic Remediation](#18-synthetic-remediation)
- [19. Synthetic Data Quality Gate](#19-synthetic-data-quality-gate)
- [20. Recovery and Hybrid Adaptation](#20-recovery-and-hybrid-adaptation)
- [21. Independent Future Validation](#21-independent-future-validation)
- [22. Experimental Design](#22-experimental-design)
- [23. Evaluation Metrics](#23-evaluation-metrics)
- [24. Dashboard](#24-dashboard)
- [25. Experiment Logging and Reproducibility](#25-experiment-logging-and-reproducibility)
- [26. Project Structure](#26-project-structure)
- [27. Technology Stack](#27-technology-stack)
- [28. Installation](#28-installation)
- [29. Running the Application](#29-running-the-application)
- [30. Demo Workflow](#30-demo-workflow)
- [31. Testing](#31-testing)
- [32. Scientific and Data-Leakage Rules](#32-scientific-and-data-leakage-rules)
- [33. Implementation Priorities](#33-implementation-priorities)
- [34. Core vs Optional Technologies](#34-core-vs-optional-technologies)
- [35. Limitations](#35-limitations)
- [36. Future Scope](#36-future-scope)
- [37. Expected Outcomes](#37-expected-outcomes)
- [38. Project Design Philosophy](#38-project-design-philosophy)
- [39. Academic Positioning](#39-academic-positioning)
- [40. Project Information](#40-project-information)
- [41. License](#41-license)

---

# 1. Project Overview

Machine-learning models are generally trained using historical data, while the environments in which they operate continue to change.

Customer behaviour, operating conditions, feature distributions, data quality, prediction behaviour, uncertainty and temporal patterns may change after deployment. As a result, a model may continue to execute normally while its predictive reliability gradually decreases.

DataDecay-X treats **model reliability as a temporal quantity**.

Instead of stopping at the question:

> **"Has the data drifted?"**

the project investigates a broader question:

> **"Given the signals observable now, can future model reliability be forecast early enough to prepare an intervention before an unacceptable reliability threshold is crossed?"**

The project therefore connects:

```text
Temporal Monitoring
        ↓
Drift Detection
        ↓
Reliability Measurement
        ↓
Reliability Forecasting
        ↓
Time-to-Decay Estimation
        ↓
Future-Regime Approximation
        ↓
Controlled Synthetic Remediation
        ↓
Synthetic Quality Control
        ↓
Hybrid Adaptation
        ↓
Independent Future Validation
```

The implementation is intentionally designed as a **compact, reproducible B.Tech project**. Classical statistical methods and lightweight machine-learning methods are used first. Computationally expensive research techniques are treated as optional extensions rather than mandatory requirements.

---

# 2. Problem Statement

A deployed machine-learning model may lose reliability when the distribution of its operating data changes over time.

Conventional monitoring can identify:

- Statistical drift
- Data-quality problems
- Changes in model predictions
- Observed performance degradation

However, a monitoring system that only reports current conditions may still be reactive.

There is a practical difference between:

```text
Detecting that something has already changed
```

and:

```text
Forecasting that model reliability is likely to become unacceptable
```

The second problem becomes particularly important when ground-truth labels are delayed.

For example, the true outcome associated with a prediction may only become available later. During that interval, directly calculating model accuracy may not be possible.

DataDecay-X therefore investigates a chronological monitoring framework in which currently observable signals are used to estimate the future reliability trajectory.

When the forecast indicates that reliability may cross a configured operational threshold, the system prepares a controlled remediation experiment rather than automatically assuming that retraining is necessary.

---

# 3. Motivation

Traditional drift monitoring often treats drift detection as the final output.

DataDecay-X treats drift as one of several signals that may explain future model degradation.

The motivation is based on the following sequence:

```text
Historical training data
        ↓
Model deployment
        ↓
Environment changes over time
        ↓
Input distributions change
        ↓
Prediction behaviour may change
        ↓
Reliability may decrease
        ↓
Ground truth may arrive later
        ↓
Reactive monitoring may detect failure late
```

The project attempts to introduce an earlier decision-support stage:

```text
Current observable signals
        ↓
Reliability trajectory
        ↓
Forecast
        ↓
Time-to-Decay
        ↓
Preparation of a possible recovery dataset
```

The objective is **not** to claim that every drift event causes model failure.

A central principle is:

> **Drift does not automatically imply harmful model degradation.**

The project therefore monitors statistical distribution changes and downstream model reliability as related but separate quantities.

---

# 4. Central Research Proposition

The central proposition of DataDecay-X is:

> **Forecast → Prepare → Adapt → Verify**

The complete practical loop is:

```text
Monitor
   ↓
Detect
   ↓
Measure Reliability
   ↓
Forecast
   ↓
Estimate Time-to-Decay
   ↓
Generate
   ↓
Quality-Control
   ↓
Adapt
   ↓
Validate
   ↓
Recover / Retain Baseline
```

Synthetic data is **not** assumed to be automatically beneficial.

Synthetic records are generated only when the intervention logic indicates that remediation should be considered. Generated records are then screened before they are allowed to influence a candidate model.

Finally, the candidate model must demonstrate its value on temporally later data that was protected from the adaptation process.

---

# 5. Research Gap

DataDecay-X does not claim that individual techniques such as PSI, KS testing, XGBoost, synthetic sampling or retraining are new.

The practical research gap addressed by the project is their integration into a small, reproducible control loop.

### Gap 1 — Drift is often treated as the endpoint

DataDecay-X connects statistical drift to downstream model reliability.

### Gap 2 — Monitoring is often descriptive

DataDecay-X adds a lightweight forecast of the reliability trajectory.

### Gap 3 — Warning and intervention are often separated

DataDecay-X uses **Time-to-Decay** as a planning signal for preparing remediation.

### Gap 4 — Synthetic data can introduce pseudo-label errors

DataDecay-X introduces a quality gate before synthetic records are used for adaptation.

### Gap 5 — Recovery benefit can be difficult to attribute

DataDecay-X uses chronological evaluation and an untouched future validation segment.

### Gap 6 — Research prototypes can be difficult to demonstrate

DataDecay-X exposes the complete workflow through an interactive Streamlit dashboard with:

- Status cards
- Drift timelines
- Reliability curves
- Forecast curves
- TTD indicators
- Synthetic quality statistics
- Recovery comparisons
- Experiment configuration
- Experiment history

---

# 6. Objectives

The project objectives are:

### Objective 1 — Temporal-shift simulation

Build a controlled temporal-shift simulator capable of producing:

- Stable scenarios
- Gradual drift
- Abrupt drift
- Recurring drift

with known ground truth.

### Objective 2 — Multi-metric drift detection

Build a drift detection engine using practical statistical measures such as:

- Population Stability Index (PSI)
- Kolmogorov–Smirnov statistic
- Wasserstein distance

where applicable.

### Objective 3 — Reliability measurement

Build a baseline reliability engine using supervised models and chronological evaluation metrics such as:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC where appropriate

### Objective 4 — Reliability forecasting

Forecast future reliability using lightweight methods such as:

- Persistence
- Recent linear trend
- XGBoost regression on lagged monitoring features

### Objective 5 — Time-to-Decay

Estimate the first future monitoring window at which predicted reliability crosses the configured minimum acceptable threshold.

### Objective 6 — Controlled synthetic generation

Generate synthetic samples representing an expected shifted regime using:

- Constrained resampling
- Numerical feature perturbation
- Controlled categorical probability adjustment

without requiring a complex deep generative model.

### Objective 7 — Synthetic quality control

Implement a quality gate using:

- Distribution similarity
- Validity checks
- Range checks
- Diversity
- Duplicate detection
- Confidence-based pseudo-label filtering

### Objective 8 — Hybrid adaptation

Combine accepted synthetic samples with recent real observations and evaluate the candidate model on independent future data.

### Objective 9 — Interactive dashboard

Build a Streamlit dashboard for:

- Monitoring
- Forecasting
- TTD
- Synthetic-data analysis
- Quality control
- Recovery comparison
- Experiment history

---

# 7. Research Questions

DataDecay-X investigates the following research questions:

### RQ1

Can the controlled monitoring pipeline correctly detect known temporal shifts introduced by the simulator?

### RQ2

Can a lightweight forecasting model predict future reliability better than persistence and simple trend baselines?

### RQ3

Can Time-to-Decay provide a useful early-warning measure before the configured reliability threshold is crossed?

### RQ4

Does forecast-conditioned synthetic adaptation improve future-window performance compared with a static or reactive baseline in controlled scenarios?

### RQ5

Does the synthetic-data quality gate reduce low-confidence or distributionally inconsistent samples before adaptation?

### RQ6

Can the complete pipeline be reproduced and inspected through an interactive dashboard without requiring specialised hardware?

---

# 8. Scope

## Included in the core implementation

- Chronological dataset handling
- Dataset validation
- Temporal windowing
- Controlled temporal drift simulation
- Multi-metric drift detection
- Baseline classification
- Reliability monitoring
- Lightweight forecasting
- Time-to-Decay estimation
- Rule-based intervention
- Controlled synthetic generation
- Synthetic quality gate
- Real-only recovery
- Synthetic-only recovery
- Hybrid recovery
- Independent future validation
- Streamlit dashboard
- SQLite experiment history
- Automated tests
- Reproducible experiment configuration

## Outside the mandatory core scope

The following may be investigated later but are not required for the core B.Tech implementation:

- Transformer-based forecasting
- Large-scale LSTM/TCN architectures
- GAN/VAE/diffusion-based generators
- Complex survival-analysis systems
- Fully autonomous production deployment
- Large-scale cloud MLOps
- Real-time distributed streaming infrastructure

---

# 9. End-to-End Workflow

```text
                         DATADECAY-X

Dataset
   │
   ▼
Validation & Preprocessing
   │
   ▼
Baseline Model
   │
   ▼
Chronological Windowing
   │
   ▼
Temporal Drift
   │
   ├───────────────┐
   ▼               │
Drift Detection    │
   │               │
   ├─ PSI          │
   ├─ KS           │
   └─ Wasserstein  │
   │               │
   ▼               │
Reliability Monitoring
   │
   ├─ Performance
   ├─ Confidence
   ├─ Prediction Behaviour
   └─ Data Quality
   │
   ▼
Reliability Forecast
   │
   ▼
Time-to-Decay
   │
   ▼
Intervention Decision
   │
   ├── MONITOR
   ├── WATCH
   ├── PREPARE
   └── RECOVER
   │
   ▼
Future-Regime Approximation
   │
   ▼
Synthetic Generation
   │
   ▼
Synthetic Quality Gate
   │
   ├── ACCEPT
   └── REJECT
   │
   ▼
Recovery Candidates
   │
   ├── No Recovery
   ├── Real Only
   ├── Synthetic Only
   └── Hybrid
   │
   ▼
Independent Future Evaluation
   │
   ▼
Recovery Decision
   │
   ▼
Experiment Logging
   │
   ▼
Dashboard / Report
```

---

# 10. System Architecture

The revised architecture is modular:

```text
M1  Data Ingestion
 ↓
M2  Baseline Model
 ↓
M3  Temporal Window Engine
 ↓
M4  Drift Detection
 ↓
M5  Reliability Engine
 ↓
M6  Reliability Forecasting
 ↓
M7  TTD Engine
 ↓
M8  Future-Regime Estimator
 ↓
M9  Synthetic Generator
 ↓
M10 Synthetic Quality Gate
 ↓
M11 Hybrid Adaptation
 ↓
M12 Independent Validation
 ↓
M13 Dashboard / Experiment Logger
```

The architecture deliberately separates:

- Observation
- Prediction
- Intervention
- Validation

This separation helps prevent future information from influencing earlier decisions.

---

# 11. Major Components

| Module | Purpose | Priority |
|---|---|---|
| Data Ingestion | Load and validate datasets | Core |
| Preprocessing | Leakage-safe preprocessing | Core |
| Temporal Simulator | Generate controlled temporal shift | Core |
| Drift Detection | Quantify distribution movement | Core |
| Baseline Models | Establish initial predictive performance | Core |
| Reliability Engine | Track model performance over time | Core |
| Forecasting | Predict future reliability | Core |
| TTD Engine | Estimate threshold-crossing time | Core |
| Intervention Engine | Determine recovery state | Core |
| Future-Regime Estimator | Approximate expected shifted distribution | Core |
| Synthetic Generator | Create controlled candidate data | Core |
| Quality Gate | Reject unsafe synthetic samples | Core |
| Recovery Engine | Compare adaptation strategies | Core |
| Evaluation Engine | Perform future validation | Core |
| Database | Store experiment history | Core |
| Reporting | Produce experiment summaries | Core |
| Dashboard | Expose the complete workflow | Core |

---

# 12. Temporal Drift Simulator

The simulator is the first major experimental foundation of DataDecay-X.

It converts a reference dataset into ordered temporal windows and introduces controlled changes to selected feature distributions.

Each experiment records:

- Drift type
- Affected features
- Start window
- Duration
- Magnitude
- Random seed
- Ground-truth schedule

This allows the system to determine whether its monitoring logic responds correctly to a known change.

## Supported scenarios

| Scenario | Description | Purpose |
|---|---|---|
| Stable | No intentional distribution shift | Control experiment |
| Gradual | Progressive shift across windows | Test slow degradation |
| Abrupt | Sudden distribution change | Test sudden failure |
| Recurring | Shift returns toward an earlier regime | Test recurring behaviour |

## Numerical features

Controlled numerical drift can be implemented through bounded:

- Mean shifts
- Scaling changes
- Noise
- Distribution perturbation

## Categorical features

Categorical drift can be implemented by changing sampling probabilities.

## Ground truth

The simulator maintains the drift configuration separately from the resulting predictor data.

This is important because the ground truth is used to evaluate whether the detector correctly identifies the known intervention.

---

# 13. Drift Detection

DataDecay-X uses multiple statistical signals rather than depending on a single drift score.

## 13.1 Population Stability Index

PSI compares reference and current proportions across bins.

Conceptually:

```text
PSI = Σ (Current_i - Reference_i)
           × ln(Current_i / Reference_i)
```

PSI is used as a practical distribution-shift indicator.

## 13.2 Kolmogorov–Smirnov Statistic

For suitable numerical features, the KS statistic compares empirical distributions.

The project can record:

- KS statistic
- KS p-value

The statistic measures distributional separation; it should not be interpreted as a universal operational threshold by itself.

## 13.3 Wasserstein Distance

Wasserstein distance measures the transport distance between numerical distributions.

It is useful for observing the magnitude and direction of distribution movement.

## 13.4 Drift Severity

Feature-level statistics can be aggregated into a dashboard-oriented **Drift Severity Index**.

The index is an engineering summary and is not intended to represent a universal statistical law.

---

# 14. Model Reliability Monitoring

Drift and model performance are tracked separately.

For every chronological window, the system can record:

### Input signals

- PSI
- KS
- Wasserstein distance

### Prediction behaviour

- Predicted-class proportions
- Probability summaries
- Prediction frequency

### Confidence signals

- Maximum predicted probability
- Entropy
- Decision margin where available

### Data-quality signals

- Missingness
- Duplicate rate
- Invalid ranges
- Schema changes

### Temporal signals

- Rolling mean
- Slope
- Volatility
- Lagged values

### Observed reliability

When labels become available:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC where appropriate

This creates a chronological reliability record.

---

# 15. Reliability Forecasting

The forecasting layer uses information available up to the current monitoring window.

The core implementation starts with transparent baselines.

## 15.1 Persistence Forecast

The simplest forecast is:

```text
Future Reliability ≈ Current Reliability
```

This establishes a naive baseline.

## 15.2 Linear Trend

A recent-window trend can be extrapolated forward.

The model estimates whether reliability is:

- Stable
- Increasing
- Decreasing

## 15.3 Optional Classical ML Forecast

A supervised forecaster such as XGBoost can use lagged monitoring features including:

- Previous reliability
- Drift scores
- Confidence
- Missingness
- Rolling statistics
- Temporal features

Advanced temporal deep-learning models are optional.

## Forecast evaluation

Forecasts are evaluated using:

- MAE
- RMSE

The project should compare forecasting methods against persistence rather than reporting only the performance of a preferred model.

---

# 16. Time-to-Decay

Time-to-Decay is one of the central operational concepts of DataDecay-X.

Let:

```text
A_min = minimum acceptable reliability threshold
```

Let:

```text
Â(t+k) = predicted reliability at future horizon k
```

Then the practical TTD definition is:

```text
TTD(t) = min { k > 0 : Â(t+k) ≤ A_min }
```

If the forecast does not cross the threshold within the configured horizon:

```text
TTD > H
```

where `H` is the forecast horizon.

## Interpretation

TTD should be treated as:

> **An estimate of the number of future monitoring windows before predicted reliability crosses the configured threshold.**

It is not a guaranteed countdown.

A sudden drift event can invalidate a previously calculated forecast. Therefore, TTD should be recalculated as new monitoring information becomes available.

---

# 17. Future-Regime Estimation

After the reliability forecast indicates a possible future decline, DataDecay-X approximates the emerging operating regime.

The core implementation uses a transparent rule-based approach.

The estimator examines:

- Recent drift statistics
- Direction of drift
- Affected features
- Forecast direction
- Recent temporal behaviour

The purpose is not to build a perfect generative model.

The purpose is to determine:

```text
Which features appear to require stronger perturbation?
Which features appear relatively stable?
Which categorical probabilities should be adjusted?
```

This estimated regime then guides synthetic-data generation.

---

# 18. Synthetic Remediation

Synthetic data has a specific role in DataDecay-X.

It is not generated merely to increase dataset size.

Its purpose is:

> **To provide additional training coverage for an expected emerging regime before sufficient future real labels become available.**

## Core generation mechanisms

### Numerical data

Possible operations include:

- Resampling
- Bounded feature perturbation
- Mean/scale adjustment
- Controlled noise

### Categorical data

Possible operations include:

- Probability reweighting
- Controlled category sampling

### Pseudo-labels

Synthetic labels may be obtained from the current model only when its confidence exceeds a configured threshold.

This is important because a degraded model can otherwise propagate its own errors into the synthetic training set.

---

# 19. Synthetic Data Quality Gate

Synthetic data is not automatically accepted.

The quality gate is a mandatory control point between generation and model adaptation.

## Quality checks

### 1. Schema validation

Verify:

- Expected columns
- Data types
- Required target field

### 2. Missing-value validation

Reject or flag samples with unacceptable missingness.

### 3. Range validation

Check whether numerical values remain inside valid or configured ranges.

### 4. Distribution similarity

Compare synthetic and reference/expected-regime distributions.

### 5. Diversity

Check whether synthetic records contain excessive concentration around a small number of patterns.

### 6. Duplicate detection

Reject excessively duplicated records.

### 7. Class balance

Check whether generated samples create an unrealistic class distribution.

### 8. Pseudo-label confidence

Reject synthetic records whose generated labels are below the configured confidence threshold.

### 9. Downstream utility

Where practical, evaluate whether synthetic augmentation improves or harms a validation task.

## Gate output

The quality gate should produce:

```text
ACCEPT
or
REJECT
```

along with:

- Accepted count
- Rejected count
- Quality score
- Failed checks
- Warnings
- Distribution statistics

Synthetic records that fail critical quality checks must not automatically enter the recovery pipeline.

---

# 20. Recovery and Hybrid Adaptation

DataDecay-X compares multiple recovery policies.

## Strategy A — No Recovery

The baseline model remains unchanged.

This is the control condition.

## Strategy B — Real Only

Recent real observations are used to retrain/adapt the model.

## Strategy C — Synthetic Only

Validated synthetic data is used for adaptation.

## Strategy D — Hybrid

Accepted synthetic samples are combined with a controlled amount of recent real data.

The hybrid strategy is particularly important because the project does not assume synthetic data should replace real observations.

Conceptually:

```text
Recent Real Data
       +
Accepted Synthetic Data
       ↓
Candidate Training Set
       ↓
Candidate Model
```

The protected future validation set remains untouched.

---

# 21. Independent Future Validation

This is one of the most important scientific controls in the project.

A recovery candidate must be evaluated on data that was not used for:

- Candidate training
- Synthetic generation
- Synthetic quality tuning
- Recovery tuning

The evaluation follows chronological order.

```text
Past
│
├── Baseline Training
│
├── Monitoring
│
├── Forecasting
│
├── Recovery Preparation
│
├── Candidate Training
│
└── Future Validation
        ↑
        Protected
```

The comparison should include:

- Baseline future F1
- Candidate future F1
- Recovery gain/loss
- Remaining degradation
- Drift level
- Training time
- Synthetic sample count
- Final decision

If recovery does not improve future performance, the baseline can remain the preferred model.

This is a deliberate design choice.

---

# 22. Experimental Design

DataDecay-X uses controlled experiments before or alongside real chronological evaluation.

## Experiment A — Stable Scenario

Purpose:

- Verify that drift metrics remain approximately stable.
- Verify that unnecessary intervention is avoided.
- Establish baseline behaviour.

## Experiment B — Gradual Drift

Purpose:

- Introduce a controlled feature shift across several windows.
- Test drift detection.
- Test reliability forecasting.
- Test TTD.

## Experiment C — Abrupt Drift

Purpose:

- Introduce a sudden distribution shift.
- Examine whether proactive forecasting has enough historical evidence.
- Measure the limitations of forecasting under sudden change.

## Experiment D — Recurring Drift

Purpose:

- Return a shifted distribution toward an earlier regime.
- Evaluate how the system handles repeated patterns.

## Experiment E — Model Comparison

Compare lightweight baseline models such as:

- Logistic Regression
- Linear SVM
- Naive Bayes

Optional models may be included when justified.

## Experiment F — Recovery Ablation

Compare:

```text
No Recovery
vs
Real Only
vs
Synthetic Only
vs
Hybrid
```

## Experiment G — Forecast Trigger

Evaluate whether forecast-driven intervention differs from:

- No intervention
- Reactive intervention

## Experiment H — Robustness

Vary:

- Drift magnitude
- Drift duration
- Drift start time
- Number of monitoring windows
- Reliability threshold
- Random seed
- Label delay where supported

---

# 23. Evaluation Metrics

## Drift metrics

| Metric | Purpose |
|---|---|
| PSI | Distribution stability |
| KS Statistic | Numerical distribution comparison |
| KS p-value | Statistical evidence of distribution difference |
| Wasserstein Distance | Distribution movement |
| Drift Severity Index | Dashboard-oriented aggregate |

## Reliability metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification performance |
| Precision | Positive prediction quality |
| Recall | Positive class coverage |
| F1 | Balanced classification reliability |
| ROC-AUC | Ranking performance where appropriate |

## Forecast metrics

| Metric | Purpose |
|---|---|
| MAE | Average forecast error |
| RMSE | Penalizes larger forecast errors |
| Trend | Direction of reliability |
| Warning Lead Time | Early-warning usefulness |
| TTD | Estimated threshold-crossing horizon |

## Remediation metrics

| Metric | Purpose |
|---|---|
| Recovery Gain | Candidate minus baseline future performance |
| Remaining Degradation | Residual performance loss |
| Threshold-Crossing Duration | Time until unacceptable reliability |
| False Intervention Count | Unnecessary recovery events |
| Synthetic Acceptance Rate | Quality-gate effectiveness |
| Synthetic Sample Count | Intervention size |
| Training Time | Computational cost |

---

# 24. Dashboard

The Streamlit application is designed as an ML reliability control center.

## Suggested navigation

```text
DATADECAY-X
────────────────────────
🏠 Overview

🧪 Experiments
   ├── New Experiment
   └── History

📊 Dataset

🌊 Drift Monitor

🤖 Model Reliability

🔮 Forecast

🧬 Synthetic Lab

🛡 Quality Gate

♻ Recovery

📈 Evaluation

📄 Reports

⚙ Settings
```

## Overview

Displays:

- Current reliability
- Drift severity
- Forecast status
- TTD
- Current system state
- Latest experiment
- Recovery status

## Dataset

Displays:

- Dataset shape
- Timestamp field
- Target field
- Feature types
- Missing values
- Class distribution
- Chronological range

## Drift Monitor

Displays:

- Feature-level drift
- PSI
- KS
- Wasserstein
- Drift timeline
- Drift severity

## Model Reliability

Displays:

- F1
- Accuracy
- Precision
- Recall
- Reliability trajectory
- Degradation percentage
- Confidence indicators

## Forecast

Displays:

- Historical reliability
- Forecast trajectory
- Forecast horizon
- Threshold line
- TTD
- Forecast error

## Synthetic Lab

Displays:

- Expected regime
- Generation settings
- Synthetic sample count
- Feature distributions
- Pseudo-label confidence

## Quality Gate

Displays:

- Individual checks
- Passed checks
- Failed checks
- Accepted samples
- Rejected samples
- Quality score

## Recovery

Displays:

- Recovery strategy
- Training configuration
- Candidate model
- Synthetic contribution
- Recent real-data contribution

## Evaluation

Displays:

- Baseline future performance
- Candidate future performance
- Recovery gain
- Remaining degradation
- Final recommendation

---

# 25. Experiment Logging and Reproducibility

Every experiment should preserve enough information to reproduce the result.

Recommended metadata includes:

- Experiment ID
- Experiment name
- Dataset name
- Dataset version
- Dataset hash where available
- Timestamp column
- Target column
- Feature configuration
- Preprocessing configuration
- Chronological split
- Model type
- Model parameters
- Drift type
- Affected features
- Drift magnitude
- Drift duration
- Drift start window
- Random seed
- Detection thresholds
- Reliability threshold
- Forecasting method
- Forecast horizon
- Synthetic generation method
- Synthetic sample count
- Quality-gate configuration
- Recovery strategy
- Evaluation window
- Final metrics
- Final decision

SQLite is used for lightweight experiment persistence.

MLflow may be introduced later for more extensive experiment tracking.

---

# 26. Project Structure

```text
DataDecay-X/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── synthetic/
│   └── samples/
│
├── experiments/
├── models/
├── reports/
├── logs/
│
├── tests/
│   ├── __init__.py
│   ├── test_simulator.py
│   ├── test_detection.py
│   ├── test_monitoring.py
│   ├── test_forecasting.py
│   ├── test_synthetic.py
│   ├── test_quality_gate.py
│   ├── test_recovery.py
│   └── test_evaluation.py
│
└── src/
    └── datadecay/
        ├── __init__.py
        ├── config.py
        ├── utils.py
        ├── data_loader.py
        ├── preprocessing.py
        ├── simulator.py
        ├── detection.py
        ├── models.py
        ├── monitoring.py
        ├── forecasting.py
        ├── intervention.py
        ├── synthetic.py
        ├── quality_gate.py
        ├── recovery.py
        ├── evaluation.py
        ├── experiment.py
        ├── database.py
        ├── reporting.py
        └── demo_data.py
```

---

# 27. Technology Stack

| Category | Technology | Role |
|---|---|---|
| Programming | Python | Core implementation |
| Data Processing | NumPy | Numerical operations |
| Data Processing | Pandas | Dataset manipulation and temporal processing |
| Scientific Computing | SciPy | Statistical tests and distances |
| Machine Learning | scikit-learn | Baseline models, preprocessing and metrics |
| Gradient Boosting | XGBoost / LightGBM | Optional forecasting/model extensions |
| Deep Learning | PyTorch | Optional advanced forecasting/generative extensions |
| Statistical Modelling | statsmodels | Optional statistical analysis |
| Visualization | Matplotlib | Static experiment figures |
| Visualization | Plotly | Interactive dashboard charts |
| Dashboard | Streamlit | Application interface |
| Experiment Tracking | MLflow | Optional experiment tracking |
| Containerization | Docker | Optional reproducible deployment |
| Persistence | SQLite | Experiment history |
| Optional Database | PostgreSQL | Larger deployment extension |
| Version Control | Git / GitHub | Source control |

---

# 28. Installation

## Requirements

Recommended environment:

- Windows 10/11, Linux or macOS
- Python 3.10+
- 8 GB RAM minimum
- 16 GB RAM recommended for larger datasets
- SSD storage
- Multi-core CPU
- GPU optional

The core implementation is designed to run on standard CPU hardware.

## Windows PowerShell

```powershell
git clone <repository-url>
cd DataDecay-X

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Linux / macOS

```bash
git clone <repository-url>
cd DataDecay-X

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 29. Running the Application

The main application entry point is:

```text
app.py
```

Run:

```powershell
python -m streamlit run app.py
```

or:

```bash
streamlit run app.py
```

The application normally opens at:

```text
http://localhost:8501
```

The core implementation uses Streamlit as the application interface and imports the DataDecay-X modules directly. A separate backend server is not required for the core B.Tech version.

---

# 30. Demo Workflow

The project should provide a reproducible demonstration mode so the complete pipeline can be evaluated without requiring an external dataset.

## Demo sequence

```text
Generate Demo Dataset
        ↓
Validate Dataset
        ↓
Chronological Split
        ↓
Train Baseline
        ↓
Create Temporal Windows
        ↓
Inject Controlled Drift
        ↓
Detect Drift
        ↓
Measure Reliability
        ↓
Forecast Reliability
        ↓
Estimate TTD
        ↓
Determine Intervention State
        ↓
Estimate Future Regime
        ↓
Generate Synthetic Samples
        ↓
Run Quality Gate
        ↓
Train Recovery Candidates
        ↓
Evaluate Future Window
        ↓
Compare Strategies
        ↓
Save Experiment
        ↓
Generate Report
```

The demo should never use hard-coded fake performance numbers. Dashboard values must be generated from the actual pipeline.

---

# 31. Testing

Run the full test suite:

```powershell
pytest -q
```

Verbose mode:

```powershell
pytest -v
```

Individual test modules:

```powershell
pytest tests/test_simulator.py -v
pytest tests/test_detection.py -v
pytest tests/test_monitoring.py -v
pytest tests/test_forecasting.py -v
pytest tests/test_synthetic.py -v
pytest tests/test_quality_gate.py -v
pytest tests/test_recovery.py -v
pytest tests/test_evaluation.py -v
```

## Testing objectives

Tests should verify:

- Temporal ordering
- Drift simulator correctness
- Ground-truth drift configuration
- Statistical metric calculations
- Reliability calculations
- Forecasting behaviour
- TTD threshold crossing
- Synthetic generation constraints
- Quality-gate rejection rules
- Recovery comparison
- Chronological future evaluation
- Experiment persistence

---

# 32. Scientific and Data-Leakage Rules

DataDecay-X follows strict chronological evaluation rules.

## Rule 1 — No future information in earlier forecasts

A forecast for time `t` may only use information available at or before `t`.

## Rule 2 — No random shuffling across forecasting boundaries

Random train-test shuffling can expose future distribution information.

Chronological separation is therefore required.

## Rule 3 — Future labels are protected

Future labels must not be used to construct an earlier forecast.

They are reserved for later evaluation.

## Rule 4 — Synthetic generation cannot use the protected future set

Synthetic data must be created from information available at the intervention point.

## Rule 5 — Quality-gate tuning must not contaminate future validation

The future validation set remains untouched.

## Rule 6 — Drift is not equal to failure

A high drift statistic does not automatically mean that the model has failed.

## Rule 7 — Recovery must be compared against a baseline

A candidate model is meaningful only when compared against:

```text
No Recovery
```

## Rule 8 — Negative results are valid results

If synthetic recovery makes the model worse, the result must be reported.

## Rule 9 — TTD is an estimate

TTD should never be presented as a guaranteed prediction of the exact failure time.

## Rule 10 — No fake dashboard metrics

Every displayed metric should originate from the experiment pipeline.

---

# 33. Implementation Priorities

The project follows a **complete-system-first** strategy.

## Phase 1 — Foundation

- Project structure
- Configuration
- Dataset loader
- Demo dataset
- Basic preprocessing
- SQLite setup

## Phase 2 — Temporal Benchmark

- Temporal windowing
- Stable scenario
- Gradual drift
- Abrupt drift
- Recurring drift
- Ground-truth logging

## Phase 3 — Monitoring

- PSI
- KS
- Wasserstein
- Drift Severity Index
- Reliability metrics
- Confidence signals
- Data-quality indicators

## Phase 4 — Forecasting

- Persistence baseline
- Linear trend
- Forecast evaluation
- TTD

## Phase 5 — Remediation

- Future-regime approximation
- Synthetic generator
- Pseudo-label confidence filtering
- Quality gate

## Phase 6 — Recovery

- No recovery
- Real only
- Synthetic only
- Hybrid
- Future evaluation

## Phase 7 — Dashboard

- Overview
- Drift
- Reliability
- Forecast
- Synthetic Lab
- Quality Gate
- Recovery
- Evaluation
- Reports

## Phase 8 — Validation

- Unit tests
- End-to-end demo
- Experiment reproducibility
- Controlled scenarios
- Documentation

---

# 34. Core vs Optional Technologies

The submitted technology direction is retained while implementation priorities are kept practical.

## Core

- Python
- NumPy
- Pandas
- SciPy
- scikit-learn
- Matplotlib / Plotly
- Streamlit
- Git / GitHub
- SQLite

## Conditional / Optional

- XGBoost
- LightGBM
- PyTorch
- statsmodels
- MLflow
- Docker
- PostgreSQL

The core B.Tech project does not depend on advanced deep-learning or cloud infrastructure.

---

# 35. Limitations

The current implementation has deliberate limitations.

### 1. Covariate drift focus

The controlled simulator primarily models feature-distribution drift.

It does not automatically reproduce every form of concept drift.

### 2. Lightweight forecasting

The default forecasting methods are intentionally simple.

They provide interpretable baselines before advanced temporal architectures are introduced.

### 3. Statistical synthetic generation

The core synthetic generator uses controlled statistical methods rather than deep generative architectures.

### 4. Classification focus

The initial workflow focuses primarily on classification tasks.

### 5. TTD uncertainty

TTD depends on the forecast model, threshold and available history.

### 6. Controlled experiments are not production reality

A simulator provides known ground truth, but successful behaviour under simulation does not prove generalisation to every real-world deployment.

### 7. Delayed labels

When labels are unavailable, the system must rely on leading signals instead of directly measuring actual performance.

### 8. CPU-oriented implementation

The project is designed for standard academic hardware rather than large-scale distributed production systems.

---

# 36. Future Scope

Potential future extensions include:

- Concept drift simulation
- Concept drift detection
- Advanced future-regime estimation
- LSTM forecasting
- TCN forecasting
- Transformer-based forecasting
- VAE-based synthetic generation
- GAN-based synthetic generation
- Diffusion-based synthetic generation
- Regression support
- Streaming monitoring
- Online learning
- Advanced uncertainty estimation
- Conformal risk control
- Calibration monitoring
- Causal drift analysis
- MLflow experiment management
- Docker deployment
- Cloud deployment
- Distributed monitoring
- Real-time event processing
- Portfolio-level model monitoring
- Automated MLOps integration

These are future extensions and are not mandatory for the core semester implementation.

---

# 37. Expected Outcomes

The expected project outputs are:

1. A reproducible DataDecay-X software prototype implementing chronological monitoring, drift detection, reliability forecasting, TTD and proactive synthetic remediation.

2. A controlled temporal-shift simulator with known schedules for stable, gradual, abrupt and recurring scenarios.

3. A multi-metric drift engine producing feature-level statistics and an aggregated Drift Severity Index.

4. A lightweight reliability forecasting module with persistence, trend and optional XGBoost baselines.

5. A transparent TTD engine linked to an operational reliability threshold.

6. A constrained synthetic-data generator and auditable quality gate.

7. A hybrid real-plus-synthetic adaptation pipeline.

8. Independent future-window validation.

9. An interactive Streamlit/Plotly dashboard suitable for project demonstration and experiment analysis.

10. Reproducible experiment records containing dataset information, preprocessing, chronological split, model settings, thresholds, random seed and final metrics.

11. A final report documenting positive results, negative results, limitations and failure cases.

The project does **not** assume that the proposed remediation will always improve performance.

---

# 38. Project Design Philosophy

DataDecay-X is intentionally a **decision-support system**, not an autonomous model replacement system.

The project does not follow:

```text
Drift detected
      ↓
Retrain immediately
```

Instead:

```text
Drift detected
      ↓
Measure actual model impact
      ↓
Forecast reliability
      ↓
Estimate possible decay
      ↓
Determine whether intervention is justified
      ↓
Generate controlled recovery candidates
      ↓
Validate synthetic data
      ↓
Evaluate candidate on future unseen data
      ↓
Retain baseline or select recovery candidate
```

This makes the workflow:

- Measurable
- Reproducible
- Auditable
- Chronologically valid
- Experimentally defensible

---

# 39. Academic Positioning

DataDecay-X is positioned within:

- Artificial Intelligence
- Machine Learning
- Machine Learning Operations (MLOps)
- Temporal Distribution Shift
- Concept Drift
- Model Reliability
- Temporal Forecasting
- Synthetic Data
- Model Adaptation

The project is an **applied experimental research and engineering framework**.

Its contribution is not a claim that PSI, KS testing, XGBoost, synthetic sampling or retraining were individually invented by the project.

Instead, the project studies how these established components can be connected into a reproducible reliability-monitoring and remediation workflow under temporal distribution shift.

The central implementation proposition is:

```text
Monitor → Detect → Forecast → Estimate TTD
→ Generate → Quality-Control → Adapt → Validate → Recover
```

---

# 40. Project Information

| Field | Details |
|---|---|
| **Project** | DataDecay-X |
| **Official Title** | Predictive Model Reliability Forecasting and Proactive Synthetic Remediation under Temporal Distribution Shift |
| **Author** | Aditya Sharma |
| **Program** | B.Tech – Computer Science Engineering |
| **Specialization** | Artificial Intelligence & Machine Learning |
| **University** | Maharishi Markandeshwar (Deemed to be University) |
| **Department** | Computer Science & Engineering |
| **Batch** | 2024–2028 |
| **Semester** | 5th Semester |
| **Project Type** | Major / Academic Project |
| **Research Type** | Applied experimental research |
| **Domain** | AI, ML, MLOps, Temporal Drift, Reliability Forecasting, Synthetic Data |
| **Project Version** | v1.0.0 |
| **Status** | Core Implementation |

---

# 41. License

This project is developed as an academic B.Tech project.

Unless a separate license is added to the repository, the source code should be treated as an academic project and not as a production-ready software distribution.

---

## DataDecay-X in One Sentence

> **DataDecay-X monitors how machine-learning reliability changes over time, forecasts potential future degradation, estimates when a reliability threshold may be crossed, prepares controlled synthetic remediation, validates that data before adaptation, and checks whether recovery actually works on unseen future data.**
