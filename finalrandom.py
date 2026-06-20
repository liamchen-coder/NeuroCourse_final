import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("==================================================")
print("   NBA Confrontation Features - RF (Ultra Compact)")
print("==================================================")

# ==============================================================================
#  File Path Setup
# ==============================================================================
FULL_DATA_PATH = r"C:\Users\user\Desktop\nn_nba_plus_confrontation.csv"
OUTPUT_IMAGE_PATH = r"C:\Users\user\Desktop\RF_Actual_vs_Predicted.png"
# ==============================================================================

# 1. Load the PLUS confrontation features dataset
try:
    df = pd.read_csv(FULL_DATA_PATH)
    print(f" SUCCESS: Dataset loaded successfully from:\n {FULL_DATA_PATH}")
except FileNotFoundError:
    print(f" ERROR: Cannot find the file at:\n 【{FULL_DATA_PATH}】")
    print(" Please check if the file path or file name is correct.")
    exit()

# 2. Split Features (X) and Target Variable (y)
X = df.drop(columns=['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PROP', 'AWAY_PROP'])
y = df['HOME_PROP']

# 3. Perform 80% Train and 20% Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Standardization (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=200,      
    max_depth=6,           
    min_samples_split=10,  
    min_samples_leaf=4,    
    random_state=42,       
    n_jobs=-1              
)

# 6. Train the Model
print("\n[TRAINING] Growing the Random Forest ensemble...")
rf_model.fit(X_train_scaled, y_train)
print(" [TRAINING] Model fitting completed!")

# 7. Model Prediction
y_pred = rf_model.predict(X_test_scaled)

# 8. Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print(" Random Forest Model Performance Evaluation Report ")
print("="*50)
print(f"• Mean Absolute Error (MAE)   : {mae:.4f}")
print(f"• Coefficient of Determination (R2 Score): {r2:.4f}")
print("--------------------------------------------------")

# ==============================================================================
# 9. Image Generation (Ultra-Compact Size Setup for Random Forest)
# ==============================================================================
print("\n[PLOTTING] Generating ultra-compact performance visualization...")

# Set plotting style
sns.set_theme(style="whitegrid")

# 🔧 縮小尺寸至 (4, 3.2)，與 SVR、BPN 完美對齊
plt.figure(figsize=(4, 3.2), dpi=300) 

# Create scatter plot (點半徑 s=15 保持精緻度)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color="#2c3e50", edgecolor="w", s=15, label="Predicted")

# Add the perfect diagonal identity line (y = x)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", linewidth=1.2, label="y=x")

# Add labels and titles (全英文學術標籤，字體 8-9pt)
plt.xlabel("Actual Home Proportion", fontsize=8, fontweight='bold')
plt.ylabel("RF Predicted Proportion", fontsize=8, fontweight='bold')
plt.title("RF Model: Actual vs. Predicted", fontsize=9, fontweight='bold', pad=8)

# Information box inside the chart (字體 7pt，精簡內邊距)
metric_text = f"MAE: {mae:.4f}\nMSE: {mse:.5f}\nR2: {r2:.4f}"
plt.gca().text(0.05, 0.95, metric_text, transform=plt.gca().transAxes, fontsize=7,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#bdc3c7'))

plt.legend(loc="lower right", fontsize=7)
plt.tight_layout()

# Save image to desktop
plt.savefig(OUTPUT_IMAGE_PATH, bbox_inches='tight')
print(f" SUCCESS: Ultra-compact Random Forest plot saved to your Desktop at:\n {OUTPUT_IMAGE_PATH}")

# Display the plot window
plt.show()
print("==================================================")