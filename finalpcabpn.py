import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("==================================================")
print("   NBA Confrontation Features - PCA + BPN Pipeline")
print("==================================================")

# ==============================================================================
#  File Path Setup
# ==============================================================================
FULL_DATA_PATH = r"C:\Users\user\Desktop\nn_nba_plus_confrontation.csv"
OUTPUT_PCA_BPN_PLOT = r"C:\Users\user\Desktop\PCABPN_Actual_vs_Predicted.png"
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

# 3. Perform 80% Train and 20% Test Split (Fixed seed for fair comparison)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Standardization (Crucial for PCA and Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# 5. Apply PCA (Dimensionality Reduction)
# Reduce 18 correlated features down to 5 orthogonal principal components
# ==============================================================================
print("\n[REDUCTION] Applying Principal Component Analysis (PCA)...")
n_components = 5
pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Calculate how much variance (information) is retained
explained_variance = np.sum(pca.explained_variance_ratio_) * 100
print(f" -> Compressed 18 features into {n_components} Principal Components.")
print(f" -> Total Cumulative Explained Variance: {explained_variance:.2f}%")

# ==============================================================================
# 6. Train the BPN (Multi-Layer Perceptron) Model
# ==============================================================================
print("[TRAINING] Fitting BPN weights via Backpropagation gradient descent...")
# Architecture: 2 hidden layers (32 -> 16 neurons) tailored for 5 inputs
pca_bpn_model = MLPRegressor(
    hidden_layer_sizes=(32, 16), 
    activation='relu', 
    solver='adam', 
    alpha=0.001,             # L2 penalty to suppress noise
    learning_rate_init=0.01, # Stable initial step size
    max_iter=600,            # Max epochs
    random_state=42
)
pca_bpn_model.fit(X_train_pca, y_train)
y_pred = pca_bpn_model.predict(X_test_pca)
print(" [TRAINING] Model fitting completed!")

# 7. Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print(" PCA + BPN Performance Evaluation Report ")
print("="*50)
print(f"• Mean Absolute Error (MAE)   : {mae:.4f}")
print(f"• Coefficient of Determination (R2 Score): {r2:.4f}")
print("--------------------------------------------------")

# ==============================================================================
#  8. Image Generation (Ultra-Compact Size Setup: 4 x 3.2)
# ==============================================================================
print("\n[PLOTTING] Displaying ultra-compact PCA+BPN visualization...")
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(4, 3.2), dpi=300)

# Scatter plot of results
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color="#2c3e50", edgecolor="w", s=15, label="Predicted")

# Identity line y=x
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", linewidth=1.2, label="y=x")

# Formatting labels and fonts
plt.xlabel("Actual Home Proportion", fontsize=8, fontweight='bold')
plt.ylabel("PCA+BPN Predicted", fontsize=8, fontweight='bold')
plt.title("PCA+BPN Model: Actual vs. Predicted", fontsize=9, fontweight='bold', pad=8)

# Metric text overlay
metric_text = f"MAE: {mae:.4f}\nMSE: {mse:.5f}\nR2: {r2:.4f}"
plt.gca().text(0.05, 0.95, metric_text, transform=plt.gca().transAxes, fontsize=7,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#bdc3c7'))

plt.legend(loc="lower right", fontsize=7)
plt.tight_layout()

# Save and Show plot window
plt.savefig(OUTPUT_PCA_BPN_PLOT, bbox_inches='tight')
plt.show()
plt.close(fig)

print("\n" + "="*50)
print("  SUCCESS: PCA+BPN plot shown and saved to Desktop! ")
print("==================================================")