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

FULL_DATA_PATH  = r"C:\Users\user\Desktop\nn_nba_plus_confrontation_with_scores.csv"
PLOT_HOME_PATH  = r"C:\Users\user\Desktop\SVR_Final_Home_Score.png"
PLOT_AWAY_PATH  = r"C:\Users\user\Desktop\SVR_Final_Away_Score.png"
PLOT_PROP_PATH  = r"C:\Users\user\Desktop\SVR_Final_Derived_Proportion.png"
# ==============================================================================

try:
    df = pd.read_csv(FULL_DATA_PATH)
    print(" SUCCESS: Correct nn_nba_plus_confrontation_with_scores.csv loaded.")
except FileNotFoundError:
    print(f" ERROR: Cannot find the file at:\n 【{FULL_DATA_PATH}】")
    exit()

#(TOTAL_PTS) 作為 Pace 放大器
df['TOTAL_PTS'] = df['HOME_PTS'] + df['AWAY_PTS']

home_features = [c for c in df.columns if c.startswith('HOME_vs_AWAY_')] + ['TOTAL_PTS']
away_features = [c for c in df.columns if c.startswith('AWAY_vs_HOME_')] + ['TOTAL_PTS']

X_home = df[home_features]
X_away = df[away_features]
y_home = df['HOME_PTS']
y_away = df['AWAY_PTS']

# 4. 執行 80% 訓練集、20% 測試集盲測分流
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_home, y_home, test_size=0.2, random_state=42)
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_away, y_away, test_size=0.2, random_state=42)

# 5. 特徵標準化
scaler_h = StandardScaler()
X_train_h_scaled = scaler_h.fit_transform(X_train_h)
X_test_h_scaled = scaler_h.transform(X_test_h)

scaler_a = StandardScaler()
X_train_a_scaled = scaler_a.fit_transform(X_train_a)
X_test_a_scaled = scaler_a.transform(X_test_a)

#  SVR 模型 (極端高懲罰調參：C=50, epsilon=0.01)
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
print("  SUCCESS: All THREE final plots saved to Desktop! 🎉 🚀")
print("==========================================================")


# ==============================================================================
# 10. 實戰部署：嚴謹提取歷史真實特徵預測 雷霆 (OKC) vs 馬刺 (SAS)
# ==============================================================================
print("\n" + "="*60)
print("  SVR Live Prediction System: Oklahoma City Thunder vs. San Antonio Spurs")
print("==========================================================")

# 1. 自動偵測你的球隊識別欄位（支援常見的 'HOME_TEAM'/'AWAY_TEAM' 或 'HOME'/'AWAY'）
if 'HOME_TEAM' in df.columns and 'AWAY_TEAM' in df.columns:
    matchup_data = df[(df['HOME_TEAM'] == 'OKC') & (df['AWAY_TEAM'] == 'SAS')]
elif 'HOME' in df.columns and 'AWAY' in df.columns:
    matchup_data = df[(df['HOME'] == 'OKC') & (df['AWAY'] == 'SAS')]
else:
    print(" 錯誤：找不到球隊識別欄位（例如 HOME 或 HOME_TEAM）。")
    print(" 請檢查 CSV 中代表主隊與客隊名稱的欄位叫什麼，並修正程式碼第 12-15 行。")
    matchup_data = pd.DataFrame()

# 2. 嚴謹執行真實數據提取與預測
if not matchup_data.empty:
    # 提取該對戰組合在資料庫中最新一場（最後一筆）的真實歷史特徵向量
    live_sample = matchup_data.iloc[[-1]]
    
    # 分離出與訓練時完全一致的主隊與客隊特徵
    live_X_home = live_sample[home_features]
    live_X_away = live_sample[away_features]
    
    # 透過同一個 Scaler 進行嚴謹的標準化轉換
    live_X_home_scaled = scaler_h.transform(live_X_home)
    live_X_away_scaled = scaler_a.transform(live_X_away)
    
    # 呼叫你訓練好的終極 SVR 模型（C=50, epsilon=0.01）跑出精準預測
    live_pred_home = svr_home.predict(live_X_home_scaled)[0]
    live_pred_away = svr_away.predict(live_X_away_scaled)[0]
    
    # 計算衍生比值
    live_derived_prop = live_pred_home / (live_pred_home + live_pred_away)
    
    # 完美輸出真實預測結果
    print(f" • 數據來源：成功自 CSV 提取最新一場真實對戰特徵。")
    print(f" • Matchup: OKC (Home) vs. SAS (Away)")
    print(f" • Predicted Home Score (OKC): {live_pred_home:.2f} PTS")
    print(f" • Predicted Away Score (SAS): {live_pred_away:.2f} PTS")
    print(f" • Derived Home Proportion   : {live_derived_prop:.4f}")
else:
    if 'HOME_TEAM' in df.columns or 'HOME' in df.columns:
        print(" 警告：在你的資料庫中找不到『主場雷霆 (OKC) vs 客場馬刺 (SAS)』的真實交手數據！")
        print(" 請確認你的 CSV 檔案中，雷霆與馬刺的球隊縮寫是否為 'OKC' 與 'SAS'。")

print("==========================================================")



# ==============================================================================
# 10. Practical Deployment: Real-Data SVR Inference Pipe (OKC vs. SAS Matchups)
# ==============================================================================
print("\n" + "="*68)
print("  SVR LIVE DEPLOYMENT SYSTEM: REAL-DATA TARGET INFERENCE PIPELINE")
print("="*68)

# Determine the exact column format of the dataset
team_col_h = 'HOME_TEAM' if 'HOME_TEAM' in df.columns else ('HOME' if 'HOME' in df.columns else None)
team_col_a = 'AWAY_TEAM' if 'AWAY_TEAM' in df.columns else ('AWAY' if 'AWAY' in df.columns else None)

if team_col_h and team_col_a:
    # --------------------------------------------------------------------------
    # Scenario A: OKC (Home) vs. SAS (Away)
    # --------------------------------------------------------------------------
    matchup_A = df[(df[team_col_h] == 'OKC') & (df[team_col_a] == 'SAS')]
    print("\n[INFERENCE] Running Scenario A: OKC (Home) vs. SAS (Away)...")
    if not matchup_A.empty:
        sample_A = matchup_A.iloc[[-1]]
        
        X_h_A = scaler_h.transform(sample_A[home_features])
        X_a_A = scaler_a.transform(sample_A[away_features])
        
        pred_h_A = svr_home.predict(X_h_A)[0]
        pred_a_A = svr_away.predict(X_a_A)[0]
        prop_A = pred_h_A / (pred_h_A + pred_a_A)
        
        print(f"  • True Ground Truth : OKC {sample_A['HOME_PTS'].values[0]} - {sample_A['AWAY_PTS'].values[0]} SAS | Prop: {sample_A['HOME_PROP'].values[0]:.4f}")
        print(f"  • SVR Predicted PTS : OKC {pred_h_A:.2f} - {pred_a_A:.2f} SAS")
        print(f"  • SVR Derived Prop  : {prop_A:.4f} (OKC Dominance Score)")
    else:
        print("  Data Error: No records matching 'OKC' at Home and 'SAS' Away in the dataset.")

    # --------------------------------------------------------------------------
    # Scenario B: SAS (Home) vs. OKC (Away) [Reversal Validation]
    # --------------------------------------------------------------------------
    matchup_B = df[(df[team_col_h] == 'SAS') & (df[team_col_a] == 'OKC')]
    print("\n[INFERENCE] Running Scenario B: SAS (Home) vs. OKC (Away)...")
    if not matchup_B.empty:
        sample_B = matchup_B.iloc[[-1]]
        
        X_h_B = scaler_h.transform(sample_B[home_features])
        X_a_B = scaler_a.transform(sample_B[away_features])
        
        pred_h_B = svr_home.predict(X_h_B)[0]
        pred_a_B = svr_away.predict(X_a_B)[0]
        prop_B = pred_h_B / (pred_h_B + pred_a_B)
        
        print(f"  • True Ground Truth : SAS {sample_B['HOME_PTS'].values[0]} - {sample_B['AWAY_PTS'].values[0]} OKC | Prop: {sample_B['HOME_PROP'].values[0]:.4f}")
        print(f"  • SVR Predicted PTS : SAS {pred_h_B:.2f} - {pred_a_B:.2f} OKC")
        print(f"  • SVR Derived Prop  : {prop_B:.4f} (SAS Dominance Score)")
    else:
        print("  Data Error: No records matching 'SAS' at Home and 'OKC' Away in the dataset.")
else:
    print("\n  Error: Missing team identity columns. Please verify that team codes are stored under 'HOME'/'AWAY' or 'HOME_TEAM'/'AWAY_TEAM'.")

print("\n" + "="*68)