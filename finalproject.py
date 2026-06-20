# import pandas as pd
# import time
# from nba_api.stats.endpoints import teamgamelogs

# print("正在從 NBA 官方 API 獲取 2025-2026 例行賽數據...")

# try:
#     # 抓取 2025-26 賽季所有球隊的比賽
#     log_finder = teamgamelogs.TeamGameLogs(
#         season_nullable='2025-26', 
#         season_type_nullable='Regular Season'
#     )
#     df_raw = log_finder.team_game_logs.get_data_frame()
    
#     home_games = df_raw[df_raw['MATCHUP'].str.contains('vs.')].copy()
#     away_games = df_raw[df_raw['MATCHUP'].str.contains('@')].copy()

#     home_games = home_games[['GAME_ID', 'GAME_DATE', 'TEAM_NAME', 'PTS', 'WL']].rename(
#         columns={'TEAM_NAME': 'HOME_TEAM', 'PTS': 'HOME_PTS', 'WL': 'HOME_WL'}
#     )
#     away_games = away_games[['GAME_ID', 'TEAM_NAME', 'PTS']].rename(
#         columns={'TEAM_NAME': 'AWAY_TEAM', 'PTS': 'AWAY_PTS'}
#     )

#     # 合併主客隊資料
#     nba_season_df = pd.merge(home_games, away_games, on='GAME_ID')

#     # 3. 核心邏輯：計算兩隊總分以及你指定的比例 (A/(A+B) 與 B/(A+B))
#     nba_season_df['TOTAL_PTS'] = nba_season_df['HOME_PTS'] + nba_season_df['AWAY_PTS']
    
#     # 主客隊得分比例換算 
#     nba_season_df['HOME_PROP'] = (nba_season_df['HOME_PTS'] / nba_season_df['TOTAL_PTS']).round(3)
#     nba_season_df['AWAY_PROP'] = (nba_season_df['AWAY_PTS'] / nba_season_df['TOTAL_PTS']).round(3)


#     nba_season_df = nba_season_df.sort_values(by='GAME_DATE').reset_index(drop=True)

#     output_filename = 'nba_2025_2026_proportions.csv'
#     nba_season_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
#     print(f"\n 成功！已成功處理完成，檔案已儲存至：{output_filename}")
#     print(f"總共成功匯入 {len(nba_season_df)} 場例行賽數據。")
#     print("\n資料前幾行預覽：")
#     print(nba_season_df[['GAME_DATE', 'HOME_TEAM', 'HOME_PTS', 'AWAY_TEAM', 'AWAY_PTS', 'HOME_PROP', 'AWAY_PROP']].head())

# except Exception as e:
#     print(f"抓取過程中發生錯誤: {e}")
#     print("提示：如果遇到 Timeout 錯誤，可能是 NBA 伺服器阻擋，建議可以稍後再試，或在 nba_api 中加入 Proxy。")

import pandas as pd
from nba_api.stats.endpoints import teamgamelogs

print("正在從 NBA 官方 API 獲取 2024-2025 例行賽數據...")

try:
    # 抓取 2024-25 賽季所有球隊的比賽日誌
    log_finder = teamgamelogs.TeamGameLogs(
        season_nullable='2024-25', 
        season_type_nullable='Regular Season'
    )
    df_raw = log_finder.team_game_logs.get_data_frame()

    home_games = df_raw[df_raw['MATCHUP'].str.contains('vs.')].copy()
    away_games = df_raw[df_raw['MATCHUP'].str.contains('@')].copy()

    home_games = home_games[['GAME_ID', 'GAME_DATE', 'TEAM_NAME', 'PTS', 'WL']].rename(
        columns={'TEAM_NAME': 'HOME_TEAM', 'PTS': 'HOME_PTS', 'WL': 'HOME_WL'}
    )
    away_games = away_games[['GAME_ID', 'TEAM_NAME', 'PTS']].rename(
        columns={'TEAM_NAME': 'AWAY_TEAM', 'PTS': 'AWAY_PTS'}
    )

    # 合併主客隊資料
    nba_season_df = pd.merge(home_games, away_games, on='GAME_ID')

    # 3. 計算總分與 A/(A+B)、B/(A+B) 比例
    nba_season_df['TOTAL_PTS'] = nba_season_df['HOME_PTS'] + nba_season_df['AWAY_PTS']
    
    nba_season_df['HOME_PROP'] = (nba_season_df['HOME_PTS'] / nba_season_df['TOTAL_PTS']).round(3)
    nba_season_df['AWAY_PROP'] = (nba_season_df['AWAY_PTS'] / nba_season_df['TOTAL_PTS']).round(3)

    nba_season_df = nba_season_df.sort_values(by='GAME_DATE').reset_index(drop=True)

    output_filename = 'nba_2024_2025_proportions.csv'
    nba_season_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"\n 成功！2024-2025 賽季數據處理完成！")
    print(f"檔案已儲存至：{output_filename}")
    print(f"總共成功匯入 {len(nba_season_df)} 場例行賽數據。")
    print("\n2024-2025 資料前幾行預覽：")
    print(nba_season_df[['GAME_DATE', 'HOME_TEAM', 'HOME_PTS', 'AWAY_TEAM', 'AWAY_PTS', 'HOME_PROP', 'AWAY_PROP']].head())

except Exception as e:
    print(f"抓取過程中發生錯誤: {e}")