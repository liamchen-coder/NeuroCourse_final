import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

print("==================================================")
print("   NBA Proportion Prediction - LightGBM Official")
print("==================================================")

# ==============================================================================
# File Path Setup
# ==============================================================================
FULL_DATA_PATH = r"C:\Users\user\Desktop\nn_nba_plus_confrontation.csv"
OUTPUT_LGB_PLOT = r"C:\Users\user\Desktop\LightGBM_Prop_Actual_vs_Predicted.png"
# ==============================================================================

# 1. Load the dataset
try:
    df = pd.read_csv(FULL_DATA_PATH)
    print(" SUCCESS: Dataset loaded successfully.")
except FileNotFoundError:
    print(f" ERROR: Cannot find the file at: {FULL_DATA_PATH}")
    exit()

# 2. Split Features (X) and Target Variable (y)
# Automatically drop metadata and away proportion, keeping all 18 tactical PLUS features
X = df.drop(columns=['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PROP', 'AWAY_PROP'])
y = df['HOME_PROP'] # Target is the proportion

print(f"• Locked [{X.shape[1]}] tactical confrontation features for training.")

# 3. Data Split: 80% Training Data & 20% Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"• [80% Training Set] Sample size   : {X_train.shape[0]} games")
print(f"• [20% Testing Set]  Sample size   : {X_test.shape[0]} games")

# 4. Feature Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and Train LightGBM Regressor
print("\n[TRAINING] Fitting LightGBM Regressor via Residual Learning...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=150,        # Number of boosting rounds
    max_depth=4,             # Limit depth to control overfitting on noise
    num_leaves=15,           # Maximum tree leaves
    learning_rate=0.05,      # Step size shrinkage
    subsample=0.8,           # Row subsampling
    colsample_bytree=0.8,    # Feature subsampling
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train_scaled, y_train)
y_pred = lgb_model.predict(X_test_scaled)
print(" [TRAINING] Model fitting completed!")

# 6. Evaluation Metrics Report
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50 + "\n LightGBM Proportion Model Evaluation \n" + "="*50)
print(f"• Mean Absolute Error (MAE)   : {mae:.4f}")
print(f"• Coefficient of Determination (R2 Score): {r2:.4f}")

# 7. Preview of 5 Blind Test Samples (Actual vs. Predicted Proportion)
print("\n [Real-world Test Sample Comparisons] :")
print("-" * 60)
for i in range(5):
    act_val = y_test.iloc[i]
    pred_val = y_pred[i]
    print(f"Game {i+1} -> Actual Proportion: {act_val:.3f} | LightGBM Predicted: {pred_val:.3f}")
print("-" * 60)

# 8. Image Generation (Ultra-Compact Size Setup: 4 x 3.2)
print("\n[PLOTTING] Generating ultra-compact LightGBM visualization...")
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(4, 3.2), dpi=300)

sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color="#2c3e50", edgecolor="w", s=15, label="Predicted")

min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", linewidth=1.2, label="y=x")

plt.xlabel("Actual Home Proportion", fontsize=8, fontweight='bold')
plt.ylabel("LightGBM Predicted", fontsize=8, fontweight='bold')
plt.title("LightGBM: Actual vs. Predicted", fontsize=9, fontweight='bold', pad=8)

metric_text = f"MAE: {mae:.4f}\nMSE: {mse:.5f}\nR2: {r2:.4f}"
plt.gca().text(0.05, 0.95, metric_text, transform=plt.gca().transAxes, fontsize=7,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#bdc3c7'))

plt.legend(loc="lower right", fontsize=7)
plt.tight_layout()

# Save and Show
plt.savefig(OUTPUT_LGB_PLOT, bbox_inches='tight')
plt.show()
plt.close(fig)

print("\n" + "="*50)
print(" SUCCESS: LightGBM proportion plot saved to Desktop! ")
print("==================================================")