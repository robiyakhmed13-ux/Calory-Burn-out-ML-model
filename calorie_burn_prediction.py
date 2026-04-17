# =============================================================================
# Calorie Burn Prediction — XGBoost vs LightGBM Comparison
# Author: [Your Name]
# Dataset: https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos
# =============================================================================
#
# This project merges two datasets (exercise.csv + calories.csv) and trains
# two state-of-the-art gradient boosting regressors to predict calories burned
# during a workout session, then compares their performance head-to-head.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# =============================================================================
# 1. Data Loading & Merging
# =============================================================================

def load_and_merge(exercise_path: str, calories_path: str) -> pd.DataFrame:
    """
    Load both CSVs and merge them side-by-side on shared index.
    exercise.csv  — workout biometrics per session
    calories.csv  — calories burned per session (target)
    """
    exercise_data = pd.read_csv(exercise_path)
    calory_data   = pd.read_csv(calories_path)

    print(f"Exercise data : {exercise_data.shape[0]} rows, {exercise_data.shape[1]} cols")
    print(f"Calories data : {calory_data.shape[0]} rows, {calory_data.shape[1]} cols")

    df = pd.concat([exercise_data, calory_data], axis=1)
    print(f"\nMerged dataset: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"Missing values:\n{df.isnull().sum()}")
    return df


# =============================================================================
# 2. Exploratory Data Analysis
# =============================================================================

def plot_eda(df: pd.DataFrame) -> None:
    """
    Comprehensive EDA:
      - Calorie distribution
      - Gender distribution
      - Age distribution
      - Calories burned by gender
      - Key biometric feature distributions
      - Correlation heatmap
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Calorie Burn Prediction — Exploratory Data Analysis", fontsize=17)

    # 1. Target distribution
    sns.histplot(df['Calories'], bins=35, kde=True,
                 ax=axes[0, 0], color='tomato')
    axes[0, 0].set_title("Distribution of Calories Burned")
    axes[0, 0].set_xlabel("Calories Burned")

    # 2. Gender distribution
    gender_counts = df['Gender'].value_counts()
    axes[0, 1].pie(gender_counts, labels=['Male', 'Female'],
                   autopct='%1.1f%%', colors=['steelblue', 'lightcoral'],
                   startangle=90)
    axes[0, 1].set_title("Gender Distribution")

    # 3. Age distribution
    sns.histplot(df['Age'], bins=25, kde=True,
                 ax=axes[0, 2], color='steelblue')
    axes[0, 2].set_title("Age Distribution")
    axes[0, 2].set_xlabel("Age (years)")

    # 4. Calories burned by gender (box plot)
    sns.boxplot(x='Gender', y='Calories', data=df,
                ax=axes[1, 0], palette=['steelblue', 'lightcoral'])
    axes[1, 0].set_title("Calories Burned by Gender")
    axes[1, 0].set_xticklabels(['Male', 'Female'])

    # 5. Duration vs Calories (scatter)
    axes[1, 1].scatter(df['Duration'], df['Calories'],
                       alpha=0.4, color='seagreen', s=15)
    axes[1, 1].set_title("Workout Duration vs Calories Burned")
    axes[1, 1].set_xlabel("Duration (minutes)")
    axes[1, 1].set_ylabel("Calories Burned")

    # 6. Heart rate vs Calories
    axes[1, 2].scatter(df['Heart_Rate'], df['Calories'],
                       alpha=0.4, color='tomato', s=15)
    axes[1, 2].set_title("Heart Rate vs Calories Burned")
    axes[1, 2].set_xlabel("Heart Rate (bpm)")
    axes[1, 2].set_ylabel("Calories Burned")

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=150)
    plt.show()
    print("EDA plots saved as 'eda_plots.png'")

    # Correlation heatmap
    plt.figure(figsize=(11, 8))
    numeric_df = df.select_dtypes(include=np.number).drop(columns=['User_ID'], errors='ignore')
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f',
                cmap='coolwarm', square=True, linewidths=0.5)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.show()
    print("Correlation heatmap saved as 'correlation_heatmap.png'")


# =============================================================================
# 3. Preprocessing
# =============================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Gender: male=0, female=1."""
    df = df.copy()
    df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
    print("Gender encoded — male: 0 | female: 1")
    return df


# =============================================================================
# 4. Feature / Target Split
# =============================================================================

def split_features_target(df: pd.DataFrame):
    """Drop User_ID (identifier) and Calories (target) from features."""
    X = df.drop(columns=['Calories', 'User_ID'], axis=1)
    Y = df['Calories']
    print(f"\nFeatures : {X.shape}  |  Target : {Y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    return X, Y


# =============================================================================
# 5. Train / Test Split
# =============================================================================

def split_data(X, Y, test_size=0.2, random_state=2):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    print(f"Train : {X_train.shape[0]} samples  |  Test : {X_test.shape[0]} samples")
    return X_train, X_test, Y_train, Y_test


# =============================================================================
# 6. Model Training
# =============================================================================

def train_models(X_train, Y_train) -> dict:
    """
    Train XGBoost and LightGBM regressors.
    Returns a dict of {model_name: fitted_model}.
    """
    models = {
        "XGBoost":  xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                      max_depth=6, random_state=2,
                                      verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                       max_depth=6, random_state=2,
                                       verbose=-1),
    }
    for name, model in models.items():
        model.fit(X_train, Y_train)
        print(f"✔ {name} training complete.")

    return models


# =============================================================================
# 7. Model Evaluation & Comparison
# =============================================================================

def evaluate_models(models: dict, X_train, Y_train,
                    X_test, Y_test) -> pd.DataFrame:
    """
    Print metrics for each model and return a comparison DataFrame.
    Metrics: R², MAE, RMSE on both train and test sets.
    """
    results = []

    for name, model in models.items():
        train_preds = model.predict(X_train)
        test_preds  = model.predict(X_test)

        train_r2   = r2_score(Y_train, train_preds)
        test_r2    = r2_score(Y_test,  test_preds)
        test_mae   = mean_absolute_error(Y_test, test_preds)
        test_rmse  = np.sqrt(mean_squared_error(Y_test, test_preds))

        results.append({
            "Model":      name,
            "Train R²":   round(train_r2,  4),
            "Test R²":    round(test_r2,   4),
            "Test MAE":   round(test_mae,  2),
            "Test RMSE":  round(test_rmse, 2),
        })

        print(f"\n── {name} ──")
        print(f"  Train R²  : {train_r2:.4f}")
        print(f"  Test  R²  : {test_r2:.4f}")
        print(f"  Test  MAE : {test_mae:.2f} calories")
        print(f"  Test  RMSE: {test_rmse:.2f} calories")

    comparison_df = pd.DataFrame(results).set_index("Model")
    print(f"\n{'='*45}")
    print("Model Comparison Summary")
    print('='*45)
    print(comparison_df.to_string())
    return comparison_df


# =============================================================================
# 8. Visualisations — Actual vs Predicted & Feature Importances
# =============================================================================

def plot_results(models: dict, X_train, Y_train, X_test, Y_test) -> None:
    """
    For each model:
      - Actual vs Predicted scatter plot (test set)
    Plus a side-by-side feature importance comparison.
    """
    sns.set_style("whitegrid")

    # Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Actual vs Predicted Calories Burned", fontsize=15)

    colors = ['steelblue', 'seagreen']
    for ax, (name, model), color in zip(axes, models.items(), colors):
        test_preds = model.predict(X_test)
        ax.scatter(Y_test, test_preds, alpha=0.4, color=color, s=20)
        mn = min(float(Y_test.min()), float(test_preds.min()))
        mx = max(float(Y_test.max()), float(test_preds.max()))
        ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect fit')
        ax.set_xlabel("Actual Calories")
        ax.set_ylabel("Predicted Calories")
        ax.set_title(f"{name}  (R²={r2_score(Y_test, test_preds):.4f})")
        ax.legend()

    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    plt.show()
    print("Actual vs Predicted plot saved as 'actual_vs_predicted.png'")

    # Feature Importances (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Feature Importances Comparison", fontsize=15)

    feat_colors = ['steelblue', 'seagreen']
    for ax, (name, model), color in zip(axes, models.items(), feat_colors):
        importances = pd.Series(model.feature_importances_,
                                index=X_train.columns).sort_values()
        importances.plot(kind='barh', ax=ax, color=color)
        ax.set_title(f"{name} — Feature Importances")
        ax.set_xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig("feature_importances.png", dpi=150)
    plt.show()
    print("Feature importances saved as 'feature_importances.png'")

    # Residuals plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Residuals (Actual − Predicted)", fontsize=15)

    for ax, (name, model), color in zip(axes, models.items(), colors):
        test_preds = model.predict(X_test)
        residuals  = np.array(Y_test) - test_preds
        ax.scatter(test_preds, residuals, alpha=0.4, color=color, s=20)
        ax.axhline(0, color='red', linestyle='--', lw=1.5)
        ax.set_xlabel("Predicted Calories")
        ax.set_ylabel("Residual")
        ax.set_title(f"{name} — Residuals")

    plt.tight_layout()
    plt.savefig("residuals.png", dpi=150)
    plt.show()
    print("Residuals plot saved as 'residuals.png'")


# =============================================================================
# 9. Predictive System
# =============================================================================

def predict_calories(model, input_data: tuple,
                     model_name: str = "Model") -> None:
    """
    Predict calories burned for a single workout session.

    Parameters
    ----------
    input_data : tuple of 7 values:
        (Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp)

        Gender     : male=0, female=1
        Age        : age in years
        Height     : height in cm
        Weight     : weight in kg
        Duration   : workout duration in minutes
        Heart_Rate : average heart rate during workout (bpm)
        Body_Temp  : body temperature during workout (°C)
    """
    arr        = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(arr)[0]
    print(f"\n🔥 [{model_name}] Predicted Calories Burned: {prediction:.1f} kcal")


# =============================================================================
# Main Pipeline
# =============================================================================

if __name__ == "__main__":
    EXERCISE_PATH = "exercise.csv"   # update path if needed
    CALORIES_PATH = "calories.csv"   # update path if needed

    # 1. Load & Merge
    df = load_and_merge(EXERCISE_PATH, CALORIES_PATH)
    print("\nFirst 5 rows:\n", df.head())

    # 2. EDA
    plot_eda(df)

    # 3. Preprocess
    df = preprocess(df)

    # 4. Features & Target
    X, Y = split_features_target(df)

    # 5. Split
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    # 6. Train both models
    models = train_models(X_train, Y_train)

    # 7. Evaluate & compare
    comparison = evaluate_models(models, X_train, Y_train, X_test, Y_test)

    # 8. Plots
    plot_results(models, X_train, Y_train, X_test, Y_test)

    # 9. Sample prediction
    # Male, age 68, height 190 cm, weight 94 kg,
    # 29 min workout, heart rate 105 bpm, body temp 40.8°C
    sample = (0, 68, 190.0, 94.0, 29.0, 105.0, 40.8)

    for name, model in models.items():
        predict_calories(model, sample, model_name=name)
