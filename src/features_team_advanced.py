import pandas as pd

def make_team_rollups(team_game_adv: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    # Attach game dates so we can roll in time order
    gdates = games[["GAME_ID","GAME_DATE"]].copy()
    gdates["GAME_DATE"] = pd.to_datetime(gdates["GAME_DATE"])

    t = team_game_adv.merge(gdates, on="GAME_ID", how="left")
    t = t.dropna(subset=["GAME_DATE"]).copy()
    t = t.sort_values(["TEAM_ID","GAME_DATE"])

    # rolling means using shift(1) to avoid leakage
    metrics = ["OFF_RATING","DEF_RATING","NET_RATING","PACE","EFG_PCT","TM_TOV_PCT","OREB_PCT","FT_RATE"]
    for w in [5, 10]:
        for m in metrics:
            t[f"{m}_L{w}"] = t.groupby("TEAM_ID")[m].shift(1).rolling(w).mean().reset_index(level=0, drop=True)

    keep = ["GAME_ID","TEAM_ID"] + [f"{m}_L5" for m in metrics] + [f"{m}_L10" for m in metrics]
    return t[keep].drop_duplicates(subset=["GAME_ID","TEAM_ID"], keep="last")

def merge_into_games(games_elo: pd.DataFrame, team_roll: pd.DataFrame) -> pd.DataFrame:
    games_elo = games_elo.drop_duplicates(subset=["GAME_ID"], keep="last").copy()

    home = team_roll.rename(columns={"TEAM_ID":"TEAM_ID_HOME"})
    away = team_roll.rename(columns={"TEAM_ID":"TEAM_ID_AWAY"})

    df = games_elo.merge(home, on=["GAME_ID","TEAM_ID_HOME"], how="left", suffixes=("","_HOME"))
    df = df.merge(away, on=["GAME_ID","TEAM_ID_AWAY"], how="left", suffixes=("_HOME","_AWAY"))

    # Build diffs (home - away) for the L10 set (best signal)
    diffs = []
    base_metrics = ["OFF_RATING","DEF_RATING","NET_RATING","PACE","EFG_PCT","TM_TOV_PCT","OREB_PCT","FT_RATE"]
    for m in base_metrics:
        df[f"{m}10_DIFF"] = df[f"{m}_L10_HOME"] - df[f"{m}_L10_AWAY"]
        df[f"{m}5_DIFF"]  = df[f"{m}_L5_HOME"]  - df[f"{m}_L5_AWAY"]
        diffs += [f"{m}10_DIFF", f"{m}5_DIFF"]

    return df

if __name__ == "__main__":
    games = pd.read_parquet("data/processed/games.parquet")
    games_elo = pd.read_parquet("data/processed/games_with_elo.parquet")
    games_elo["GAME_DATE"] = pd.to_datetime(games_elo["GAME_DATE"])

    team_adv = pd.read_parquet("data/raw/team_game_advanced.parquet")

    team_roll = make_team_rollups(team_adv, games)
    out = merge_into_games(games_elo, team_roll)

    # Drop rows without enough history
    out = out.dropna(subset=["NET_RATING10_DIFF","OFF_RATING10_DIFF","DEF_RATING10_DIFF","PACE10_DIFF"]).copy()

    out.to_parquet("data/processed/games_features_v3.parquet", index=False)
    print("Saved data/processed/games_features_v3.parquet")
    print(out[["GAME_ID","GAME_DATE","ELO_DIFF_PRE","NET_RATING10_DIFF","EFG_PCT10_DIFF","HOME_WIN"]].head())
