import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("==================================================")
print("   NBA Confrontation Features - Lasso with Full Plots")
print("==================================================")

# ==============================================================================
# 🔧 File Path Setup
# ==============================================================================
FULL_DATA_PATH = r"C:\Users\user\Desktop\nn_nba_plus_confrontation.csv"
PLOT_ACTUAL_PRED = r"C:\Users\user\Desktop\Lasso_Actual_vs_Predicted.png"
PLOT_COEF_PATH = r"C:\Users\user\Desktop\Lasso_Coefficient_Path.png"
PLOT_IMPORTANCE = r"C:\Users\user\Desktop\Lasso_Feature_Importance.png"
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
feature_names = X.columns.tolist()

# 3. Perform 80% Train and 20% Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and Fit LassoCV to find the best alpha
print("\n[TRAINING] Fitting LassoCV...")
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=5000)
lasso_cv.fit(X_train_scaled, y_train)
best_alpha = lasso_cv.alpha_
y_pred = lasso_cv.predict(X_test_scaled)

# ==============================================================================
# 📋 5.1 Print Feature Selection Report in Terminal
# ==============================================================================
print("\n" + "="*50)
print(" 🔍 LASSO FEATURE SELECTION REPORT 🔍")
print("="*50)
retained_count = 0
for name, coef in zip(feature_names, lasso_cv.coef_):
    if coef != 0:
        print(f" ✅ [RETAINED] {name:<30} : Coef = {coef:.5f}")
        retained_count += 1
    else:
        print(f" ❌ [DROPPED ] {name:<30} : Coef = 0.00000")
print(f"--------------------------------------------------")
print(f" Total Retained Features: {retained_count} / {len(feature_names)}")
print("==================================================")

# ==============================================================================
# 📈 Plot 1: Actual vs. Predicted Plot (Ultra-Compact: 4 x 3.2)
# ==============================================================================
print("[PLOTTING] Displaying Plot 1: Actual vs. Predicted...")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

sns.set_theme(style="whitegrid")
fig1 = plt.figure(figsize=(4, 3.2), dpi=300)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color="#2c3e50", edgecolor="w", s=15, label="Predicted")

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", linewidth=1.2, label="y=x")

plt.xlabel("Actual Home Proportion", fontsize=8, fontweight='bold')
plt.ylabel("Lasso Predicted Proportion", fontsize=8, fontweight='bold')
plt.title("Lasso: Actual vs. Predicted", fontsize=9, fontweight='bold', pad=8)

metric_text = f"MAE: {mae:.4f}\nMSE: {mse:.5f}\nR2: {r2:.4f}"
plt.gca().text(0.05, 0.95, metric_text, transform=plt.gca().transAxes, fontsize=7,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#bdc3c7'))
plt.legend(loc="lower right", fontsize=7)
plt.tight_layout()
plt.savefig(PLOT_ACTUAL_PRED, bbox_inches='tight')
plt.show() # 🔧 修正：先讓螢幕顯示出來
plt.close(fig1)

# ==============================================================================
# 📈 Plot 2: Lasso Coefficient Path Plot (Ultra-Compact: 4 x 3.2)
# ==============================================================================
print("[PLOTTING] Displaying Plot 2: Lasso Coefficient Path...")
alphas = np.logspace(-4, -1, 100)
coefs = []
for a in alphas:
    lasso_path = Lasso(alpha=a, max_iter=5000, random_state=42)
    lasso_path.fit(X_train_scaled, y_train)
    coefs.append(lasso_path.coef_)

fig2 = plt.figure(figsize=(4, 3.2), dpi=300)
plt.plot(alphas, coefs)
plt.axvline(x=best_alpha, color='#e74c3c', linestyle='--', linewidth=1.2, label=f'Best Alpha ({best_alpha:.4f})')
plt.xscale('log')
plt.xlabel('Alpha (Penalty Intensity)', fontsize=8, fontweight='bold')
plt.ylabel('Coefficients Weight', fontsize=8, fontweight='bold')
plt.title('Lasso Coefficient Path', fontsize=9, fontweight='bold', pad=8)
plt.legend(loc="lower left", fontsize=6)
plt.tight_layout()
plt.savefig(PLOT_COEF_PATH, bbox_inches='tight')
plt.show() # 🔧 修正：先讓螢幕顯示出來
plt.close(fig2)

# ==============================================================================
# 📈 Plot 3: Lasso Feature Importance Bar Chart (Ultra-Compact: 4 x 3.2)
# ==============================================================================
print("[PLOTTING] Displaying Plot 3: Lasso Feature Importance...")
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lasso_cv.coef_
})
importance_df['Abs_Coef'] = importance_df['Coefficient'].abs()
importance_df = importance_df[importance_df['Coefficient'] != 0].sort_values(by='Abs_Coef', ascending=False)

fig3 = plt.figure(figsize=(4, 3.2), dpi=300)
if not importance_df.empty:
    colors = ['#2980b9' if c > 0 else '#c0392b' for c in importance_df['Coefficient']]
    sns.barplot(x='Coefficient', y='Feature', data=importance_df, palette=colors, hue='Feature', legend=False)
else:
    plt.text(0.5, 0.5, 'All coefficients are zeroed', dict(size=10))

plt.xlabel('Coefficient Weight (Impact)', fontsize=8, fontweight='bold')
plt.ylabel('', fontsize=8)
plt.title('Lasso Selected Feature Importance', fontsize=9, fontweight='bold', pad=8)
plt.tick_params(axis='y', labelsize=6)
plt.tight_layout()
plt.savefig(PLOT_IMPORTANCE, bbox_inches='tight')
plt.show() # 🔧 修正：先讓螢幕顯示出來
plt.close(fig3)

print("\n" + "="*50)
print(" 🚀 SUCCESS: All 3 plots shown and saved to Desktop! 🚀")
print("==================================================")