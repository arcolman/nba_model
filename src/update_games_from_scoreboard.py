import os
import argparse
from datetime import datetime, timedelta

import pandas as pd
import requests_cache
from nba_api.stats.endpoints import scoreboardv2

requests_cache.install_cache("nba_cache", expire_after=60 * 60)  # 1 hour cache


def fetch_final_games_for_date(mmddyyyy: str) -> pd.DataFrame:
    frames = scoreboardv2.ScoreboardV2(game_date=mmddyyyy, timeout=60).get_data_frames()
    game_header = frames[0]
    line_score = frames[1]

    if game_header.empty:
        return pd.DataFrame()

    if "GAME_STATUS_TEXT" in game_header.columns:
        finals = game_header[game_header["GAME_STATUS_TEXT"].astype(str).str.startswith("Final")].copy()
    else:
        finals = game_header.copy()

    if finals.empty:
        return pd.DataFrame()

    ls = line_score[line_score["GAME_ID"].isin(finals["GAME_ID"])].copy()
    if ls.empty:
        return pd.DataFrame()

    # store GAME_DATE as datetime (not string)
    game_date_dt = pd.to_datetime(datetime.strptime(mmddyyyy, "%m/%d/%Y").date())

    rows = []
    for gid in finals["GAME_ID"].unique():
        g_ls = ls[ls["GAME_ID"] == gid].copy()
        if len(g_ls) != 2:
            continue

        # Prefer MATCHUP parsing; fallback to IDs from header
        if "MATCHUP" in g_ls.columns:
            home_candidates = g_ls[g_ls["MATCHUP"].astype(str).str.contains("vs", case=False, na=False)]
            away_candidates = g_ls[g_ls["MATCHUP"].astype(str).str.contains("@", case=False, na=False)]
        else:
            home_candidates = pd.DataFrame()
            away_candidates = pd.DataFrame()

        if len(home_candidates) == 1 and len(away_candidates) == 1:
            home = home_candidates.iloc[0]
            away = away_candidates.iloc[0]
        else:
            gh = finals[finals["GAME_ID"] == gid].iloc[0]
            home_id = int(gh["HOME_TEAM_ID"])
            away_id = int(gh["VISITOR_TEAM_ID"])
            home = g_ls[g_ls["TEAM_ID"] == home_id].iloc[0]
            away = g_ls[g_ls["TEAM_ID"] == away_id].iloc[0]

        pts_home = float(home["PTS"])
        pts_away = float(away["PTS"])
        home_win = 1 if pts_home > pts_away else 0

        rows.append({
            "GAME_ID": str(gid),
            "GAME_DATE": game_date_dt,
            "TEAM_ID_HOME": int(home["TEAM_ID"]),
            "TEAM_ABBREVIATION_HOME": str(home["TEAM_ABBREVIATION"]),
            "PTS_HOME": pts_home,
            "TEAM_ID_AWAY": int(away["TEAM_ID"]),
            "TEAM_ABBREVIATION_AWAY": str(away["TEAM_ABBREVIATION"]),
            "PTS_AWAY": pts_away,
            "HOME_WIN": home_win,
        })

    df_new = pd.DataFrame(rows)
    if not df_new.empty:
        df_new["GAME_DATE"] = pd.to_datetime(df_new["GAME_DATE"], errors="coerce")
    return df_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (local). If omitted, uses yesterday.")
    args = ap.parse_args()

    if args.date is None:
        d = (datetime.today().date() - timedelta(days=1))
    else:
        d = datetime.strptime(args.date, "%Y-%m-%d").date()

    mmddyyyy = d.strftime("%m/%d/%Y")
    out_path = "data/processed/games.parquet"

    print(f"Fetching finals for {mmddyyyy}...")

    df_new = fetch_final_games_for_date(mmddyyyy)
    if df_new.empty:
        print("No FINAL games found yet (maybe games not finished, or no games that day).")
        return

    if os.path.exists(out_path):
        df_old = pd.read_parquet(out_path)
        # normalize old schema
        df_old["GAME_ID"] = df_old["GAME_ID"].astype(str)
        df_old["GAME_DATE"] = pd.to_datetime(df_old["GAME_DATE"], errors="coerce")

        df = pd.concat([df_old, df_new], ignore_index=True)
        df["GAME_ID"] = df["GAME_ID"].astype(str)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.drop_duplicates(subset=["GAME_ID"], keep="last")
    else:
        df = df_new.copy()

    # If any dates failed to parse, don't write garbage
    bad = df["GAME_DATE"].isna().sum()
    if bad:
        raise ValueError(f"Found {bad} rows with invalid GAME_DATE after parsing. Fix GAME_DATE format first.")

    df.to_parquet(out_path, index=False)
    print(f"Added {len(df_new)} games. Total games now: {df['GAME_ID'].nunique()}")
    print(df_new[["GAME_ID", "TEAM_ABBREVIATION_AWAY", "PTS_AWAY", "TEAM_ABBREVIATION_HOME", "PTS_HOME", "HOME_WIN"]])


if __name__ == "__main__":
    main()
