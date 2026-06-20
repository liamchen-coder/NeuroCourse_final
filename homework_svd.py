import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# Step 1: Load Real Data (Typhoon Sinlaku)
# ============================================================================
print("--- Loading Real World Data ---")

main_path = r"C:\Users\user\Downloads\114-2 ANN_Week 06_Data _ Codes\\"
filename = "Data (Regression).xlsx"
file_full_path = os.path.join(main_path, filename)

df = pd.read_excel(file_full_path, sheet_name='2008 Sinlaku', header=0)

# 提取特徵 (X) 與 目標 (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"Dataset successfully loaded! {X.shape[0]} samples, {X.shape[1]} features.")

# ============================================================================
# Step 2: DATA SPLITTING (60% Train, 20% Val, 20% Test)
# ============================================================================
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"Data Split -> Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# 標準化
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# ============================================================================
# Step 3: SVR Model Training
# ============================================================================
model = SVR(kernel='rbf', C=100, epsilon=0.1)
model.fit(X_train_s, y_train)

# ============================================================================
# Step 4: Model Evaluation (計算指標)
# ============================================================================
def calculate_ce(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)

def get_results(X_s, y_true):
    y_pred = model.predict(X_s)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    ce = calculate_ce(y_true, y_pred)
    return y_pred, rmse, r2, ce

y_train_pred, rmse_tr, r2_tr, ce_tr = get_results(X_train_s, y_train)
y_val_pred, rmse_va, r2_va, ce_va = get_results(X_val_s, y_val)
y_test_pred, rmse_te, r2_te, ce_te = get_results(X_test_s, y_test)

# ---------------------------------------------------------
# 輸出 1: Test Output Table (像 BPN 的預測對照表)
# ---------------------------------------------------------
print("\nTest Output after training:")
print(f"{' ':4s}{'Predicted Output':20s}{'True Output':20s}")
for i in range(len(y_test)):
    print(f"{i:<4d}{y_test_pred[i]:<20.6f}{y_test[i]:<20.2f}")

# ---------------------------------------------------------
# 輸出 2: Final Evaluation Summary (像 BPN 的框框表格)
# ---------------------------------------------------------
print("\n============================================================")
print("--- Stage 3: Final Model Evaluation Results (SVR) ---")
print("============================================================")
print(f"{'[Training   Set]':15s} RMSE: {rmse_tr:.4f} | R2: {r2_tr:.4f} | CE: {ce_tr:.4f}")
print(f"{'[Validation Set]':15s} RMSE: {rmse_va:.4f} | R2: {r2_va:.4f} | CE: {ce_va:.4f}")
print(f"{'[Testing    Set]':15s} RMSE: {rmse_te:.4f} | R2: {r2_te:.4f} | CE: {ce_te:.4f}")
print("============================================================")

# ============================================================================
# Step 5: Visualization (像 RBFN 的彩色散佈圖)
# ============================================================================
plt.figure(figsize=(9, 6))

plt.scatter(y_train, y_train_pred, alpha=0.4, color='royalblue', label=f'Training (CE={ce_tr:.2f})')
plt.scatter(y_val, y_val_pred, alpha=0.4, color='green', label=f'Validation (CE={ce_va:.2f})')
plt.scatter(y_test, y_test_pred, alpha=0.8, color='crimson', marker='^', label=f'Testing (CE={ce_te:.2f})')

# 繪製 y=x 線
all_y = np.concatenate([y_train, y_train_pred])
min_val, max_val = all_y.min(), all_y.max()
plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction (y=x)')

plt.title("SVR Performance Evaluation (Scatter Plot)", fontsize=14)
plt.xlabel("True Depth (cm)", fontsize=12)
plt.ylabel("Predicted Depth (cm)", fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()