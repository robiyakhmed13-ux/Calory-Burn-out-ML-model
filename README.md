# 🔥 Calorie Burn Prediction

A machine learning project that predicts **calories burned during a workout session** by merging exercise biometric data with calorie records, then training and comparing two state-of-the-art gradient boosting regressors — **XGBoost** and **LightGBM**.

---

## 📌 Project Overview

Accurate calorie estimation during exercise is valuable for fitness tracking, sports science, and health management. This project goes beyond a single model — it benchmarks XGBoost against LightGBM across multiple metrics to determine which gradient boosting framework delivers the best predictions on workout biometric data.

| Item | Detail |
|------|--------|
| **Algorithms** | XGBoost Regressor, LightGBM Regressor |
| **Task** | Regression |
| **Dataset** | [Calories & Exercise – Kaggle](https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos) |
| **Data Source** | Two CSVs merged: `exercise.csv` + `calories.csv` |
| **Target** | `Calories` — kilocalories burned per workout session |

---

## 📂 Project Structure

```
calorie_burn_prediction/
│
├── calorie_burn_prediction.ipynb    # Jupyter Notebook (full walkthrough)
├── calorie_burn_prediction.py       # Clean Python script
├── requirements.txt                 # Dependencies
├── exercise.csv                     # Exercise dataset (download from Kaggle)
├── calories.csv                     # Calories dataset (download from Kaggle)
├── eda_plots.png                    # 6-panel EDA visualization
├── correlation_heatmap.png          # Feature correlation heatmap
├── actual_vs_predicted.png          # XGBoost vs LightGBM predictions
├── feature_importances.png          # Side-by-side feature importance comparison
├── residuals.png                    # Residuals analysis for both models
└── README.md
```

---

## 📊 Dataset Features

Two separate CSV files are merged side-by-side to form the final dataset:

### `exercise.csv` — Workout Biometrics
| Feature | Description |
|---------|-------------|
| `User_ID` | Unique user identifier (dropped — not a feature) |
| `Gender` | Gender — male=0, female=1 |
| `Age` | Age of the user (years) |
| `Height` | Height in centimetres |
| `Weight` | Weight in kilograms |
| `Duration` | Duration of the workout (minutes) |
| `Heart_Rate` | Average heart rate during workout (bpm) |
| `Body_Temp` | Body temperature during workout (°C) |

### `calories.csv` — Target
| Feature | Description |
|---------|-------------|
| `User_ID` | Matching user identifier |
| `Calories` | ✅ **Target** — kilocalories burned during session |

---

## 🔀 Why Two Datasets?

Real-world ML pipelines often require **merging data from multiple sources**. This project demonstrates that skill: the exercise biometrics and calorie outcomes are stored separately and must be joined before modelling — a common pattern in health tech and sports analytics data engineering.

```
exercise.csv  ──┐
                ├── pd.concat(axis=1) ──▶ merged_df ──▶ model
calories.csv  ──┘
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/calorie-burn-prediction.git
cd calorie-burn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download both `exercise.csv` and `calories.csv` from [Kaggle](https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos) and place them in the project root.

### 4. Run
```bash
python calorie_burn_prediction.py
```

---

## 🔄 Pipeline

```
exercise.csv + calories.csv
         │
         ▼
   Merge (pd.concat axis=1)
         │
         ▼
  EDA — 6 panels: calorie distribution, gender breakdown,
        age distribution, calories by gender, duration vs
        calories, heart rate vs calories + correlation heatmap
         │
         ▼
  Encode Gender (male=0, female=1)
         │
         ▼
  Drop User_ID (identifier, not a feature)
         │
         ▼
  Train / Test Split (80% / 20%)
         │
         ├──▶  XGBoost Regressor  ──▶ R², MAE, RMSE
         │
         └──▶  LightGBM Regressor ──▶ R², MAE, RMSE
                     │
                     ▼
         Side-by-side comparison table
                     │
                     ▼
         Actual vs Predicted | Feature Importances | Residuals
                     │
                     ▼
         Single-session calorie prediction
```

---

## 📈 Results

| Model | Train R² | Test R² | Test MAE | Test RMSE |
|-------|----------|---------|----------|-----------|
| **XGBoost** | ~0.9996 | ~0.9987 | ~1.8 kcal | ~2.9 kcal |
| **LightGBM** | ~0.9993 | ~0.9982 | ~2.1 kcal | ~3.4 kcal |

> Both models achieve exceptional accuracy. XGBoost edges ahead on this dataset, predicting calories burned to within ~2 kcal on average.

---

## 🔑 Key Findings

- **Duration** and **Heart Rate** are the two most important predictors of calories burned — the harder and longer you work out, the more you burn
- **Body Temperature** is a surprisingly strong signal — it reflects physiological exertion intensity
- **Weight** contributes meaningfully — heavier individuals burn more calories for the same effort
- **Gender** and **Age** have lower but still relevant importance scores

---

## 🔮 Sample Prediction

```python
# Male (0), age 68, height 190 cm, weight 94 kg,
# 29 min session, heart rate 105 bpm, body temp 40.8°C
sample = (0, 68, 190.0, 94.0, 29.0, 105.0, 40.8)

predict_calories(xgboost_model, sample, model_name="XGBoost")
# 🔥 [XGBoost]  Predicted Calories Burned: ~245.3 kcal

predict_calories(lightgbm_model, sample, model_name="LightGBM")
# 🔥 [LightGBM] Predicted Calories Burned: ~243.8 kcal
```

---

## 🤖 Model Highlights

### XGBoost
- Gradient boosted decision trees with regularisation (L1 & L2)
- Strong default performance, highly tunable
- Slightly higher accuracy on this dataset

### LightGBM
- Leaf-wise tree growth (vs level-wise in XGBoost) — faster training
- Lower memory usage, excellent on large datasets
- Near-identical accuracy to XGBoost with faster fit time

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas / numpy** — data processing & merging
- **xgboost** — XGBoost Regressor
- **lightgbm** — LightGBM Regressor
- **scikit-learn** — train/test split, evaluation metrics
- **seaborn / matplotlib** — visualization

---

## 🚀 Future Improvements

- [ ] Add CatBoost to the comparison for a full gradient boosting trio
- [ ] Hyperparameter tuning with `Optuna` or `GridSearchCV`
- [ ] Cross-validation (k-fold) for robust metric estimates
- [ ] Feature engineering: BMI (`Weight / Height²`), intensity score (`Heart_Rate × Duration`)
- [ ] Deploy as a Streamlit fitness calculator web app

---

## 📄 License

MIT License

---

## Result:
<img width="575" height="457" alt="image" src="https://github.com/user-attachments/assets/4ed1fbaa-afba-4d85-b440-845d774ed510" />
