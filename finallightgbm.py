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
print("   NBA Confrontation Features - LightGBM Model")
print("==================================================")

# ==============================================================================
# 🔧 File Path Setup
# ==============================================================================
FULL_DATA_PATH = r"C:\Users\user\Desktop\nn_nba_plus_confrontation.csv"
OUTPUT_LGB_PLOT = r"C:\Users\user\Desktop\LightGBM_Actual_vs_Predicted.png"
# ==============================================================================

# 1. Load the dataset
try:
    df = pd.read_csv(FULL_DATA_PATH)
    print(" SUCCESS: Dataset loaded successfully.")
except FileNotFoundError:
    print(f" ERROR: Cannot find the file at: {FULL_DATA_PATH}")
    exit()

# 2. Split Features (X) and Target Variable (y)
X = df.drop(columns=['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PROP', 'AWAY_PROP'])
y = df['HOME_PROP']

# 3. Perform 80% Train and 20% Test Split (Fixed seed for consistency)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# 🚀 5. Initialize and Train LightGBM Regressor
# ==============================================================================
print("\n[TRAINING] Fitting LightGBM Regressor via Residual Learning...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=150,        # 建立 150 棵樹來進行殘差修正
    max_depth=4,             # 限制樹深防止運動數據中的隨機雜訊造成過擬合
    num_leaves=15,           # 限制每棵樹的最大葉子節點數
    learning_rate=0.05,      # 穩定的學習率（步長）
    subsample=0.8,           # 随機抽取 80% 的樣本建立每棵樹，增加魯棒性
    colsample_bytree=0.8,    # 隨機抽取 80% 的特徵變數，避免單一戰術主導
    random_state=42,         # 鎖定隨機種子
    n_jobs=-1,               # 啟動所有 CPU 核心平行加速
    verbose=-1               # 隱藏冗餘的警告訊息
)
lgb_model.fit(X_train_scaled, y_train)
y_pred = lgb_model.fit(X_train_scaled, y_train).predict(X_test_scaled)
print(" [TRAINING] Model fitting completed!")

# 6. Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print(" 📊 LightGBM Performance Evaluation Report 📊")
print("="*50)
print(f"• Mean Absolute Error (MAE)   : {mae:.4f}")
print(f"• Coefficient of Determination (R2 Score): {r2:.4f}")
print("--------------------------------------------------")

# ==============================================================================
# 📈 7. Image Generation (Ultra-Compact Size Setup: 4 x 3.2)
# ==============================================================================
print("\n[PLOTTING] Displaying ultra-compact LightGBM visualization...")
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(4, 3.2), dpi=300)

# Create scatter plot
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color="#2c3e50", edgecolor="w", s=15, label="Predicted")

# Add perfect diagonal identity line (y = x)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", linewidth=1.2, label="y=x")

# Formatting titles and axis labels (全英文學術規範)
plt.xlabel("Actual Home Proportion", fontsize=8, fontweight='bold')
plt.ylabel("LightGBM Predicted", fontsize=8, fontweight='bold')
plt.title("LightGBM Model: Actual vs. Predicted", fontsize=9, fontweight='bold', pad=8)

# Information box inside the chart
metric_text = f"MAE: {mae:.4f}\nMSE: {mse:.5f}\nR2: {r2:.4f}"
plt.gca().text(0.05, 0.95, metric_text, transform=plt.gca().transAxes, fontsize=7,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#bdc3c7'))

plt.legend(loc="lower right", fontsize=7)
plt.tight_layout()

# Save first, then display to screen
plt.savefig(OUTPUT_LGB_PLOT, bbox_inches='tight')
plt.show()
plt.close(fig)

print("\n" + "="*50)
print(" 🚀 SUCCESS: LightGBM plot shown and saved to Desktop! 🚀")
print("==================================================")