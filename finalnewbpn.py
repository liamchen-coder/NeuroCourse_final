import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("==========================================================")
print("   NBA Score & Derived Proportion - BPN Neural Network")
print("==========================================================")

# ==============================================================================
# 🔧 檔案與圖表儲存路徑（完美對齊：nn_nba_plus_confrontation_with_scores.csv）
# ==============================================================================
FULL_DATA_PATH  = r"C:\Users\user\Desktop\nn_nba_plus_confrontation_with_scores.csv"
PLOT_HOME_PATH  = r"C:\Users\user\Desktop\BPN_Final_Home_Score.png"
PLOT_AWAY_PATH  = r"C:\Users\user\Desktop\BPN_Final_Away_Score.png"
PLOT_PROP_PATH  = r"C:\Users\user\Desktop\BPN_Final_Derived_Proportion.png"
# ==============================================================================

# 1. 讀取整合檔案
try:
    df = pd.read_csv(FULL_DATA_PATH)
    print(" SUCCESS: Correct dataset loaded into BPN pipeline.")
except FileNotFoundError:
    print(f" ERROR: Cannot find the file at:\n 【{FULL_DATA_PATH}】")
    exit()

# 2. 動態特徵工程：在內建程式碼中動態算出單場總得分 (TOTAL_PTS) 作為 Pace 放大器
df['TOTAL_PTS'] = df['HOME_PTS'] + df['AWAY_PTS']

# 3. 特徵去耦掛勾：主隊網路用 HOME 進攻指標；客隊網路用 AWAY 進攻指標，各自加上 TOTAL_PTS
home_features = [c for c in df.columns if c.startswith('HOME_vs_AWAY_')] + ['TOTAL_PTS']
away_features = [c for c in df.columns if c.startswith('AWAY_vs_HOME_')] + ['TOTAL_PTS']

X_home = df[home_features]
X_away = df[away_features]
y_home = df['HOME_PTS']
y_away = df['AWAY_PTS']

# 4. 嚴謹執行 80% 訓練集、20% 測試集盲測分流 (固定隨機種子以利學術公平對比)
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_home, y_home, test_size=0.2, random_state=42)
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_away, y_away, test_size=0.2, random_state=42)

print(f"• [BPN Network] Input Dimension: {X_home.shape[1]} channels.")
print(f"• Train size: {X_train_h.shape[0]} games | Blind Test size: {X_test_h.shape[0]} games")

# 5. 特徵標準化 (類神經網路對特徵尺度極度敏感，此步驟非常關鍵)
scaler_h = StandardScaler()
X_train_h_scaled = scaler_h.fit_transform(X_train_h)
X_test_h_scaled = scaler_h.transform(X_test_h)

scaler_a = StandardScaler()
X_train_a_scaled = scaler_a.fit_transform(X_train_a)
X_test_a_scaled = scaler_a.transform(X_test_a)

# ==============================================================================
# 6. 訓練強烈對抗均值迴歸的 BPN (MLPRegressor) 模型
# 架構：包含 2 個隱藏層 [64, 32]，使用 ReLU 激活函數與 Adam 反向傳播優化器
# ==============================================================================
print("\n[TRAINING] Training BPN Neural Network for Home Team Score...")
bpn_home = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate_init=0.005,
    max_iter=350,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
bpn_home.fit(X_train_h_scaled, y_train_h)
pred_home = bpn_home.predict(X_test_h_scaled)

print("[TRAINING] Training BPN Neural Network for Away Team Score...")
bpn_away = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate_init=0.005,
    max_iter=350,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
bpn_away.fit(X_train_a_scaled, y_train_a)
pred_away = bpn_away.predict(X_test_a_scaled)

# ==============================================================================
# 🌟 7. 計算衍生分數比值 (Proportion)
# ==============================================================================
y_test_prop = df.loc[y_test_h.index, 'HOME_PROP']
pred_prop = pred_home / (pred_home + pred_away)

print("\n" + "="*50 + "\n BPN Model Evaluation Summary \n" + "="*50)
print(f"• BPN Home Score -> MAE: {mean_absolute_error(y_test_h, pred_home):.2f} | R2: {r2_score(y_test_h, pred_home):.4f}")
print(f"• BPN Away Score -> MAE: {mean_absolute_error(y_test_a, pred_away):.2f} | R2: {r2_score(y_test_a, pred_away):.4f}")
print(f"• BPN Derived Prop -> MAE: {mean_absolute_error(y_test_prop, pred_prop):.4f} | R2: {r2_score(y_test_prop, pred_prop):.4f}")

# ==============================================================================
# 📊 8. 盲測實戰 3 個預測數值前 5 場隨機對照預覽
# ==============================================================================
print("\n🔍 BPN REAL-WORLD BLIND TEST COMPARISONS (3 CRITICAL VALUES) :")
print("-" * 85)
for i in range(5):
    act_h = y_test_h.iloc[i]
    act_a = y_test_a.iloc[i]
    act_p = y_test_prop.iloc[i]
    print(f" Game {i+1:<1} | Actual: {act_h:3.0f}/{act_a:3.0f}/{act_p:.3f} | BPN Pred: {pred_home[i]:5.1f}/{pred_away[i]:5.1f}/{pred_prop[i]:.3f} 🌟")
print("-" * 85)

# ==============================================================================
# 📈 9. 學術標準出圖函數 (超精簡尺寸: 4 x 3.2, DPI 300)
# ==============================================================================
def generate_and_save_plot(y_true, y_pred, title_name, x_label, y_label, save_path):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(4, 3.2), dpi=300)
    
    # 繪製交點（同時代表實際與預測）
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, color="#2e4053", edgecolor="w", s=15, label="BPN Pred")
    
    # 繪製完美預測恆等線 y = x
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", linewidth=1.2, label="y=x")
    
    plt.xlabel(x_label, fontsize=8, fontweight='bold')
    plt.ylabel(y_label, fontsize=8, fontweight='bold')
    plt.title(title_name, fontsize=9, fontweight='bold', pad=8)
    
    # 內嵌統計框
    if "Prop" in title_name:
        metric_text = f"MAE: {mae:.4f}\nR2: {r2:.4f}"
    else:
        metric_text = f"MAE: {mae:.2f}\nR2: {r2:.4f}"
        
    plt.gca().text(0.05, 0.95, metric_text, transform=plt.gca().transAxes, fontsize=7,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#bdc3c7'))
    
    plt.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# ==============================================================================
# 10. 執行 BPN 三圖連發
# ==============================================================================
print("\n[PLOTTING] Generating ultra-compact BPN visualizations...")
generate_and_save_plot(y_test_h, pred_home, "BPN: Home Score Model", "Actual Home Score", "Predicted Home Score", PLOT_HOME_PATH)
generate_and_save_plot(y_test_a, pred_away, "BPN: Away Score Model", "Actual Away Score", "Predicted Away Score", PLOT_AWAY_PATH)
generate_and_save_plot(y_test_prop, pred_prop, "BPN: Derived Proportion Model", "Actual Home Proportion", "Predicted Home Proportion", PLOT_PROP_PATH)

print("\n" + "="*50)
print(" 🚀 🎉 SUCCESS: All THREE final BPN plots saved to Desktop! 🎉 🚀")
print("==========================================================")