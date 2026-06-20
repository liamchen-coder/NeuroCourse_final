import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("==========================================================")
print("   NBA Score & Derived Proportion - Random Forest Pipeline")
print("==========================================================")

# ==============================================================================
# 🔧 檔案與圖表儲存路徑（完美對齊：nn_nba_plus_confrontation_with_scores.csv）
# ==============================================================================
FULL_DATA_PATH  = r"C:\Users\user\Desktop\nn_nba_plus_confrontation_with_scores.csv"
PLOT_HOME_PATH  = r"C:\Users\user\Desktop\RF_Final_Home_Score.png"
PLOT_AWAY_PATH  = r"C:\Users\user\Desktop\RF_Final_Away_Score.png"
PLOT_PROP_PATH  = r"C:\Users\user\Desktop\RF_Final_Derived_Proportion.png"
# ==============================================================================

# 1. 讀取整合檔案
try:
    df = pd.read_csv(FULL_DATA_PATH)
    print(" SUCCESS: Correct dataset loaded into Random Forest pipeline.")
except FileNotFoundError:
    print(f" ERROR: Cannot find the file at:\n 【{FULL_DATA_PATH}】")
    exit()

# 2. 動態特徵工程：在內建程式中自動加總產生 TOTAL_PTS (Pace量體控制指標)
df['TOTAL_PTS'] = df['HOME_PTS'] + df['AWAY_PTS']

# 3. 特徵去耦掛勾：主隊森林用 HOME 進攻指標；客隊森林用 AWAY 進攻指標，各自橫向加入 TOTAL_PTS
home_features = [c for c in df.columns if c.startswith('HOME_vs_AWAY_')] + ['TOTAL_PTS']
away_features = [c for c in df.columns if c.startswith('AWAY_vs_HOME_')] + ['TOTAL_PTS']

X_home = df[home_features]
X_away = df[away_features]
y_home = df['HOME_PTS']
y_away = df['AWAY_PTS']

# 4. 嚴謹執行 80% 訓練集、20% 測試集盲測分流 (固定種子42以進行跨模型公平對照)
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_home, y_home, test_size=0.2, random_state=42)
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_away, y_away, test_size=0.2, random_state=42)

print(f"• [Random Forest] Individual Feature Dimension: {X_home.shape[1]} variables.")
print(f"• Train size: {X_train_h.shape[0]} games | Blind Test size: {X_test_h.shape[0]} games")

# 5. 特徵標準化
scaler_h = StandardScaler()
X_train_h_scaled = scaler_h.fit_transform(X_train_h)
X_test_h_scaled = scaler_h.transform(X_test_h)

scaler_a = StandardScaler()
X_train_a_scaled = scaler_a.fit_transform(X_train_a)
X_test_a_scaled = scaler_a.transform(X_test_a)

# ==============================================================================
# 6. 訓練雙通道隨機森林模型 (設定多線程 n_jobs=-1 自動加速)
# ==============================================================================
print("\n[TRAINING] Training Random Forest for Home Team Score...")
rf_home = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf_home.fit(X_train_h_scaled, y_train_h)
pred_home = rf_home.predict(X_test_h_scaled)

print("[TRAINING] Training Random Forest for Away Team Score...")
rf_away = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf_away.fit(X_train_a_scaled, y_train_a)
pred_away = rf_away.predict(X_test_a_scaled)

# ==============================================================================
# 🌟 7. 衍生分數比值 (Proportion) 計算
# ==============================================================================
y_test_prop = df.loc[y_test_h.index, 'HOME_PROP']
pred_prop = pred_home / (pred_home + pred_away)

print("\n" + "="*50 + "\n Random Forest Model Evaluation \n" + "="*50)
print(f"• RF Home Score -> MAE: {mean_absolute_error(y_test_h, pred_home):.2f} | R2: {r2_score(y_test_h, pred_home):.4f}")
print(f"• RF Away Score -> MAE: {mean_absolute_error(y_test_a, pred_away):.2f} | R2: {r2_score(y_test_a, pred_away):.4f}")
print(f"• RF Derived Prop -> MAE: {mean_absolute_error(y_test_prop, pred_prop):.4f} | R2: {r2_score(y_test_prop, pred_prop):.4f}")

# ==============================================================================
# 📊 8. 盲測實戰 3 個預測數值前 5 場隨機對照預覽
# ==============================================================================
print("\n🔍 RF REAL-WORLD BLIND TEST COMPARISONS (3 CRITICAL VALUES) :")
print("-" * 85)
for i in range(5):
    act_h = y_test_h.iloc[i]
    act_a = y_test_a.iloc[i]
    act_p = y_test_prop.iloc[i]
    print(f" Game {i+1:<1} | Actual: {act_h:3.0f}/{act_a:3.0f}/{act_p:.3f} | RF Pred: {pred_home[i]:5.1f}/{pred_away[i]:5.1f}/{pred_prop[i]:.3f} 🌟")
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
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, color="#1b4f72", edgecolor="w", s=15, label="RF Pred")
    
    # 繪製完美預測理想線 y = x
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", linewidth=1.2, label="y=x")
    
    plt.xlabel(x_label, fontsize=8, fontweight='bold')
    plt.ylabel(y_label, fontsize=8, fontweight='bold')
    plt.title(title_name, fontsize=9, fontweight='bold', pad=8)
    
    # 內嵌評估統計框
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
# 10. 執行隨機森林三圖連發
# ==============================================================================
print("\n[PLOTTING] Generating ultra-compact Random Forest visualizations...")
generate_and_save_plot(y_test_h, pred_home, "RF: Home Score Model", "Actual Home Score", "Predicted Home Score", PLOT_HOME_PATH)
generate_and_save_plot(y_test_a, pred_away, "RF: Away Score Model", "Actual Away Score", "Predicted Away Score", PLOT_AWAY_PATH)
generate_and_save_plot(y_test_prop, pred_prop, "RF: Derived Proportion Model", "Actual Home Proportion", "Predicted Home Proportion", PLOT_PROP_PATH)

print("\n" + "="*50)
print(" 🚀 🎉 SUCCESS: All THREE final Random Forest plots saved to Desktop! 🎉 🚀")
print("==========================================================")