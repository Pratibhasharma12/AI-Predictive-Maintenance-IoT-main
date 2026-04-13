# Student Project Guide

## A. Project Explanation

### Simple explanation

AI-powered predictive maintenance means using data from machines to predict whether a machine is likely to fail soon. Instead of waiting for a breakdown, companies monitor machine conditions and act early.

This helps industries:

- avoid sudden machine stoppage
- reduce repair cost
- improve productivity
- plan maintenance in advance
- improve worker and plant safety

### Technical explanation

Predictive maintenance combines sensor telemetry, feature engineering, and machine learning to estimate failure risk before an asset reaches a critical state. Typical input variables include temperature, vibration, pressure, current, rotational speed, torque, or wear indicators. The model learns patterns associated with past failures and then predicts risk on unseen data.

### Why industries care

- Manufacturing plants: unplanned downtime can stop an entire production line.
- Factories: maintenance teams can schedule repairs during planned downtime.
- Power plants: failure prediction helps protect expensive turbines, pumps, and generators.
- Automotive industry: predictive insights improve line efficiency and machine utilization.
- Aviation industry: maintenance planning improves reliability and safety.

### Business value

- Predict machine failure before breakdown
- Reduce downtime and maintenance delays
- Save cost on emergency repair and replacement
- Improve operational efficiency
- Improve asset life and maintenance planning

### Full workflow

1. Sensor data collection
   A real system collects temperature, pressure, vibration, torque, speed, wear, or other telemetry from IoT sensors.
2. Preprocessing
   Missing values, duplicates, noise, inconsistent formats, and wrong data types are fixed.
3. Feature engineering
   Useful signals are created, such as temperature difference, wear interactions, or power proxies.
4. Model training
   The model learns the relationship between sensor conditions and machine failure labels.
5. Prediction
   The trained model predicts whether a machine is likely to fail.
6. Alert generation
   Probability thresholds are converted into maintenance alerts like normal, medium, and high risk.
7. Visualization
   Charts, confusion matrices, and alert summaries help explain the results to engineers and managers.

## B. Tech Stack Options

### Option A: Easiest

- Tools: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- Dataset type: tabular industrial sensor data
- Model types: Logistic Regression, Decision Tree, Random Forest
- Difficulty: beginner
- GPU: not required
- Output: failure prediction, confusion matrix, alert table, plots

### Option B: Intermediate

- Tools: Python, Pandas, Scikit-learn, XGBoost or LightGBM, Streamlit
- Dataset type: richer sensor history or time-ordered machine records
- Model types: Gradient Boosting, XGBoost
- Difficulty: intermediate
- GPU: optional
- Output: stronger model metrics and a simple dashboard

### Option C: Advanced

- Tools: Python, PyTorch or TensorFlow, LSTM, Docker, FastAPI, Streamlit
- Dataset type: time-series sequences such as NASA CMAPSS
- Model types: LSTM, GRU, sequence anomaly detection
- Difficulty: advanced
- GPU: helpful but not mandatory
- Output: sequence modeling, RUL forecasting, deployment-ready API

## C. Selected Approach

The best choice for a student portfolio project is Option A with one useful industry improvement: compare Logistic Regression and Random Forest, then keep the best model.

Why this is the best fit:

- easy to run on a normal laptop
- easy to explain in placements and internships
- realistic enough for industrial analytics storytelling
- produces strong GitHub proof through plots, model artifacts, and reports
- avoids deep learning complexity at the start

### Selected dataset

Use the UCI AI4I 2020 Predictive Maintenance Dataset.

Why:

- public and credible
- synthetic but industry-style
- includes machine condition variables similar to IoT telemetry
- beginner-friendly structure
- enough realism for portfolio storytelling

## D. Architecture

### Text-based block diagram

```text
Industrial Sensor Dataset
        |
        v
Data Loading Module
        |
        v
Cleaning + Standardization
        |
        v
Feature Engineering
        |
        v
Train/Test Split
        |
        v
ML Model Training
        |
        v
Failure Prediction
        |
        v
Alert Generation
        |
        v
Visualization + Reports + GitHub Proof
```

### Module explanation

- Input data module: loads virtual machine telemetry.
- Preprocessing module: cleans and standardizes raw data.
- Feature engineering module: creates extra machine-health indicators.
- Modeling module: trains classification models.
- Evaluation module: calculates accuracy, precision, recall, ROC-AUC, and confusion matrix.
- Alert module: converts model probability into maintenance priority.
- Visualization module: saves plots for reports and README screenshots.

### Data flow

Raw dataset -> cleaned dataset -> engineered features -> model-ready features -> trained model -> predictions -> alerts -> visual outputs

## D1. Full Project Implementation Plan

### Phase 1 - Setup

- What to do: create project folders, create virtual environment, install dependencies.
- Why: this gives you a clean and reproducible workspace.
- Expected output: working project structure and installed packages.
- Common mistakes: installing packages globally, forgetting virtual environment activation.
- How to verify: run `python --version` and `pip list`.

### Phase 2 - Dataset Loading

- What to do: download and load the UCI AI4I dataset.
- Why: this acts as the virtual IoT sensor stream.
- Expected output: `data/raw/ai4i2020.csv`.
- Common mistakes: wrong path handling or API download failure.
- How to verify: open the CSV and confirm sensor columns are present.

### Phase 3 - Data Cleaning

- What to do: standardize column names, trim category values, remove duplicates.
- Why: clean input improves code reliability and model quality.
- Expected output: consistent lowercase snake_case columns.
- Common mistakes: training on dirty column names or duplicated records.
- How to verify: inspect `outputs/reports/preprocessing_summary.txt`.

### Phase 4 - Feature Engineering

- What to do: create temperature difference, power proxy, wear interactions, and wear-speed ratio.
- Why: engineered features help the model capture machine stress patterns.
- Expected output: extra health-related columns in processed data.
- Common mistakes: using target-derived columns or dividing by zero.
- How to verify: open `data/processed/processed_ai4i2020.csv`.

### Phase 5 - Model Building

- What to do: train Logistic Regression and Random Forest pipelines.
- Why: comparing models gives stronger proof than using just one model.
- Expected output: selected best model and saved artifact in `models/`.
- Common mistakes: ignoring class imbalance or not separating train and test data.
- How to verify: inspect `outputs/reports/model_selection.json`.

### Phase 6 - Evaluation

- What to do: compute accuracy, precision, recall, ROC-AUC, and confusion matrix.
- Why: evaluation shows whether the model is practical for failure prediction.
- Expected output: metrics JSON and confusion matrix plot.
- Common mistakes: checking only accuracy when classes are imbalanced.
- How to verify: inspect `outputs/reports/metrics.json` and `outputs/figures/confusion_matrix.png`.

### Phase 7 - Failure Prediction

- What to do: predict failure probability on the test set.
- Why: probability-based output is more realistic for maintenance decisions.
- Expected output: `outputs/reports/test_predictions.csv`.
- Common mistakes: using only class labels and not storing probabilities.
- How to verify: confirm `failure_probability` exists in the prediction report.

### Phase 8 - Visualization

- What to do: generate failure distribution, probability histogram, and top-risk charts.
- Why: visuals help recruiters and interviewers understand your project quickly.
- Expected output: PNG files in `outputs/figures/`.
- Common mistakes: creating plots without saving them.
- How to verify: open the generated images locally.

### Phase 9 - GitHub Publishing

- What to do: initialize git, commit in logical steps, push to GitHub, and add topics.
- Why: clean commit history improves proof of work.
- Expected output: public repository with meaningful history.
- Common mistakes: one giant commit or no README screenshots.
- How to verify: view commits and README on GitHub.

### Phase 10 - Final Output

- What to do: polish README, add screenshots, explain learning outcomes, and pin the repo.
- Why: final presentation matters for internships and placements.
- Expected output: professional portfolio repository.
- Common mistakes: missing explanation of business value or industry relevance.
- How to verify: ask a friend to read the README and explain the project back to you.

## E. Folder Structure

```text
AI-Predictive-Maintenance-IoT/
│
├── data/
│   ├── raw/                 # downloaded original dataset
│   └── processed/           # cleaned and engineered dataset
├── notebooks/               # optional EDA notebooks
├── src/                     # all source modules
├── models/                  # saved trained models
├── outputs/
│   ├── figures/             # charts and confusion matrix
│   └── reports/             # metrics, predictions, alerts, summaries
├── images/                  # screenshots used inside README
├── docs/                    # project guide and proof plan
├── README.md                # professional project overview
├── requirements.txt         # dependencies
├── .gitignore               # ignore local env and generated files
└── main.py                  # pipeline entry point
```

## F. Installation and Environment Setup

### Recommended Python version

- Python 3.11 or 3.12 recommended for beginners
- Python 3.13 can also work if all packages install correctly

### Libraries

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- ucimlrepo

### Windows

```powershell
cd D:\AI-Predictive-Maintenance-IoT
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Mac/Linux

```bash
cd /path/to/AI-Predictive-Maintenance-IoT
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## G. Code Overview

### Important files

- `main.py`: runs the complete workflow.
- `src/data_loader.py`: downloads and loads the UCI dataset.
- `src/preprocess.py`: cleans and normalizes data.
- `src/features.py`: creates engineered features and prepares model inputs.
- `src/model.py`: builds candidate models and selects the best one.
- `src/evaluate.py`: saves metrics and prediction outputs.
- `src/alerts.py`: creates maintenance alerts from prediction probabilities.
- `src/visualize.py`: saves project charts.
- `src/pipeline.py`: orchestrates the complete process.

### Practical modeling note

The dataset includes failure-mode columns like `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`. Since `Machine failure` is derived from these flags, using them as input would leak the answer to the model. This project removes them so the learning pipeline stays realistic.

## H. Virtual Simulation

### How the dataset simulates IoT sensor data

The dataset acts like a virtual stream of machine condition records. Each row represents a machine state snapshot with values such as:

- air temperature
- process temperature
- rotational speed
- torque
- tool wear
- product type

### How machine failure is represented

The `machine_failure` column is the target label:

- `0` means no failure
- `1` means failure

### How the model predicts breakdown

The model learns patterns in sensor conditions. If the combination of temperature, speed, torque, and wear looks risky, the predicted failure probability becomes higher.

### Simulation steps

1. Load dataset as virtual telemetry.
2. Clean and standardize columns.
3. Engineer new condition features.
4. Split data into training and testing sets.
5. Train classification models.
6. Predict failure probability on unseen records.
7. Label predictions into normal, medium, and high alert.
8. Save plots and reports as proof assets.

### Outputs to generate

- processed dataset
- model selection report
- metrics report
- test prediction table
- top alerts table
- summary text report
- failure distribution chart
- confusion matrix
- failure probability histogram
- top risky machines chart

### Student proof to capture

- terminal run output
- saved figures in `outputs/figures/`
- metrics JSON or summary text
- GitHub commit history
- README screenshots

## I. How to Run the Project

### Train and evaluate

```bash
python main.py --mode full --threshold 0.45
```

### What happens when you run it

- dataset is downloaded if not already cached
- cleaned data is saved
- features are engineered
- two baseline models are trained
- best model is selected
- metrics and predictions are saved
- alert file and plots are generated

### Example console output format

```text
Pipeline completed successfully.
Selected model: random_forest
Accuracy=0.97 | Precision=0.78 | Recall=0.63 | ROC-AUC=0.95
Reports saved to: .../outputs/reports
```

The exact numbers may vary slightly depending on package versions.

## J. GitHub Upload Strategy

### Good repository names

- `AI-Predictive-Maintenance-IoT`
- `predictive-maintenance-iot-ml`
- `industrial-predictive-maintenance-ml`

### Suggested repository description

`Industry-oriented predictive maintenance project using virtual IoT sensor data, machine learning, alert generation, and data visualization.`

### Suggested topics

- `python`
- `machine-learning`
- `predictive-maintenance`
- `iot`
- `industrial-analytics`
- `scikit-learn`
- `data-science`
- `manufacturing`

### Git commands

```bash
git init
git add .
git commit -m "Initial project scaffold for predictive maintenance system"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

### Commit strategy

- `feat: scaffold predictive maintenance project structure`
- `feat: add dataset loading and preprocessing pipeline`
- `feat: add feature engineering and model training`
- `feat: add evaluation reports and alert generation`
- `docs: add README and project architecture`
- `docs: add screenshots and proof assets`

## K. README Guidance

Your README should include:

- project overview
- business problem
- industry relevance
- architecture
- tech stack
- dataset details
- installation steps
- usage commands
- results and screenshots
- learning outcomes

The repository already includes a polished README you can extend with screenshots from the `outputs/figures/` folder.

## L. Day-by-Day GitHub Proof Plan

### Day 1

- create repo
- add folder structure
- add `requirements.txt`
- commit proof: setup screenshots and first commit

### Day 2

- add dataset loading
- explain dataset columns
- commit proof: dataset preview screenshot

### Day 3

- add cleaning and preprocessing
- explain leakage removal
- commit proof: before and after preprocessing screenshot

### Day 4

- add feature engineering and baseline models
- commit proof: model training terminal screenshot

### Day 5

- add evaluation metrics
- save confusion matrix and prediction outputs
- commit proof: metrics screenshot

### Day 6

- add alert logic and final visuals
- commit proof: top risky machine chart screenshot

### Day 7

- polish README
- upload screenshots
- add project explanation and architecture
- commit proof: final GitHub repository screenshot

## M. Proof Checklist

- [ ] professional README
- [ ] clean folder structure
- [ ] working Python pipeline
- [ ] public dataset source mentioned
- [ ] metrics saved
- [ ] plots saved
- [ ] model artifact saved
- [ ] screenshots added to README
- [ ] meaningful git commits
- [ ] project summary written in simple language
- [ ] interview talking points prepared
