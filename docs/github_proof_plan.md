# GitHub Proof Plan

## 7-Day Commit Plan

### Day 1 - Setup

- Create repository and base folders
- Add `.gitignore`, `requirements.txt`, and `main.py`
- Commit message:
  `feat: initialize predictive maintenance project scaffold`

### Day 2 - Dataset

- Add dataset download and loading logic
- Document chosen dataset and why it fits IoT simulation
- Commit message:
  `feat: add industrial dataset loading pipeline`

### Day 3 - Preprocessing

- Normalize column names
- Remove duplicates
- Drop leakage columns from modeling inputs
- Commit message:
  `feat: add preprocessing and leakage-safe data preparation`

### Day 4 - Modeling

- Add Logistic Regression and Random Forest pipelines
- Select best model using training F1
- Commit message:
  `feat: add baseline model training and selection workflow`

### Day 5 - Evaluation

- Save metrics JSON
- Save test predictions
- Add confusion matrix generation
- Commit message:
  `feat: add model evaluation reports and confusion matrix`

### Day 6 - Visualization

- Add distribution and risk charts
- Add maintenance alert logic
- Commit message:
  `feat: add alert generation and failure visualization assets`

### Day 7 - Portfolio Polish

- Improve README
- Add screenshots in `images/`
- Add architecture and business explanation
- Commit message:
  `docs: polish README and portfolio-ready project documentation`

## Proof Assets to Upload

- `images/dataset_preview.png`
- `images/preprocessing_output.png`
- `images/model_training_log.png`
- `images/confusion_matrix.png`
- `images/failure_probability_distribution.png`
- `images/top_risky_machines.png`
- `images/github_repo_preview.png`

## README Screenshot Usage

Use short captions like:

- `Dataset preview from the virtual machine telemetry source`
- `Confusion matrix for machine failure prediction`
- `Top risky machines identified by the alert engine`

## Professional Tips

- Keep commit messages meaningful
- Use repository topics
- Pin the repository on your GitHub profile
- Add 3 to 5 screenshots only, not too many
- Mention what you learned and what you would improve next
