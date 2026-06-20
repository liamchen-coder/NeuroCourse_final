import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("==========================================================")
print("   NBA Score & Derived Proportion - 3-Plot Visualizer")
print("==========================================================")

# ==============================================================================
#  檔案與圖表儲存路徑）
# ==============================================================================
FULL_DATA_PATH  = r"C:\Users\user\Desktop\nn_nba_plus_confrontation_with_scores.csv"
PLOT_HOME_PATH  = r"C:\Users\user\Desktop\SVR_Final_Home_Score.png"
PLOT_AWAY_PATH  = r"C:\Users\user\Desktop\SVR_Final_Away_Score.png"
PLOT_PROP_PATH  = r"C:\Users\user\Desktop\SVR_Final_Derived_Proportion.png"
# ==============================================================================

# 1. 讀取整合檔案
try:
    df = pd.read_csv(FULL_DATA_PATH)
    print(" SUCCESS: Correct nn_nba_plus_confrontation_with_scores.csv loaded.")
except FileNotFoundError:
    print(f" ERROR: Cannot find the file at:\n 【{FULL_DATA_PATH}】")
    exit()

# 2. 動態特徵工程：在內建程式碼中動態算出單場總得分 (TOTAL_PTS) 作為 Pace 放大器
df['TOTAL_PTS'] = df['HOME_PTS'] + df['AWAY_PTS']

# 3. 特徵去耦掛勾：主隊模型用 HOME 指標；客隊模型用 AWAY 指標，並各自加上 TOTAL_PTS
home_features = [c for c in df.columns if c.startswith('HOME_vs_AWAY_')] + ['TOTAL_PTS']
away_features = [c for c in df.columns if c.startswith('AWAY_vs_HOME_')] + ['TOTAL_PTS']

X_home = df[home_features]
X_away = df[away_features]
y_home = df['HOME_PTS']
y_away = df['AWAY_PTS']

# 4. 嚴謹執行 80% 訓練集、20% 測試集盲測分流
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_home, y_home, test_size=0.2, random_state=42)
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_away, y_away, test_size=0.2, random_state=42)

# 5. 特徵標準化
scaler_h = StandardScaler()
X_train_h_scaled = scaler_h.fit_transform(X_train_h)
X_test_h_scaled = scaler_h.transform(X_test_h)

scaler_a = StandardScaler()
X_train_a_scaled = scaler_a.fit_transform(X_train_a)
X_test_a_scaled = scaler_a.transform(X_test_a)

# 6. 訓練強烈對抗均值迴歸的 SVR 模型 (極端高懲罰調參：C=50, epsilon=0.01)
print("\n[TRAINING] Fitting SVR for Home Team Score...")
svr_home = SVR(kernel='rbf', C=50.0, epsilon=0.01)
svr_home.fit(X_train_h_scaled, y_train_h)
pred_home = svr_home.predict(X_test_h_scaled)

print("[TRAINING] Fitting SVR for Away Team Score...")
svr_away = SVR(kernel='rbf', C=50.0, epsilon=0.01)
svr_away.fit(X_train_a_scaled, y_train_a)
pred_away = svr_away.predict(X_test_a_scaled)

# ==============================================================================
# 7. 計算衍生分數比值 (Proportion)
# ==============================================================================
# 實際比值 (直接從原始測試集中提取對應的真實 HOME_PROP)
y_test_prop = df.loc[y_test_h.index, 'HOME_PROP']
# 預測比值 = 主隊預測得分 / (主隊預測得分 + 客隊預測得分)
pred_prop = pred_home / (pred_home + pred_away)

print("\n" + "="*50 + "\n Model Evaluation Summary \n" + "="*50)
print(f"• Home Score Model -> MAE: {mean_absolute_error(y_test_h, pred_home):.2f} | R2: {r2_score(y_test_h, pred_home):.4f}")
print(f"• Away Score Model -> MAE: {mean_absolute_error(y_test_a, pred_away):.2f} | R2: {r2_score(y_test_a, pred_away):.4f}")
print(f"• Derived Prop Model -> MAE: {mean_absolute_error(y_test_prop, pred_prop):.4f} | R2: {r2_score(y_test_prop, pred_prop):.4f}")

# ==============================================================================
#  8. 學術標準出圖函數 (超精簡尺寸: 4 x 3.2, DPI 300)
# ==============================================================================
def generate_and_save_plot(y_true, y_pred, title_name, x_label, y_label, save_path):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(4, 3.2), dpi=300)
    
    # 畫出點（同時代表實際與預測）
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, color="#2c3e50", edgecolor="w", s=15, label="Predicted")
    
    # 畫出完美吻合的理想線 y = x
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
# 9. 執行三圖連發自動化輸出
# ==============================================================================
print("\n[PLOTTING] Generating ultra-compact visualizations...")
# 圖 1：主隊得分
generate_and_save_plot(y_test_h, pred_home, "SVR: Home Score Model", "Actual Home Score", "Predicted Home Score", PLOT_HOME_PATH)
# 圖 2：客隊得分
generate_and_save_plot(y_test_a, pred_away, "SVR: Away Score Model", "Actual Away Score", "Predicted Away Score", PLOT_AWAY_PATH)
# 圖 3：透過得分預測值逆向推算出的比值（比分比例）
generate_and_save_plot(y_test_prop, pred_prop, "SVR: Derived Proportion Model", "Actual Home Proportion", "Predicted Home Proportion", PLOT_PROP_PATH)

print("\n" + "="*50)
print(" SUCCESS: All THREE final plots saved to Desktop! ")
print("==========================================================")