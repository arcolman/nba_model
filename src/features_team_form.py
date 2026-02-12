import pandas as pd

def build_team_game_features(team_games: pd.DataFrame) -> pd.DataFrame:
    tg = team_games.copy()
    tg["GAME_DATE"] = pd.to_datetime(tg["GAME_DATE"])
    tg = tg.sort_values(["TEAM_ID", "GAME_DATE"])

    # Deduplicate team-game rows just in case
    tg = tg.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="last").copy()

    tg["OPP_PTS"] = tg["PTS"] - tg["PLUS_MINUS"]

    tg["REST_DAYS"] = tg.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days
    tg["REST_DAYS"] = tg["REST_DAYS"].fillna(7).clip(lower=0)
    tg["B2B"] = (tg["REST_DAYS"] == 0).astype(int)

    for w in [5, 10]:
        tg[f"PTS_FOR_L{w}"] = (
            tg.groupby("TEAM_ID")["PTS"].shift(1).rolling(w).mean().reset_index(level=0, drop=True)
        )
        tg[f"PTS_AGAINST_L{w}"] = (
            tg.groupby("TEAM_ID")["OPP_PTS"].shift(1).rolling(w).mean().reset_index(level=0, drop=True)
        )
        tg[f"NET_PTS_L{w}"] = tg[f"PTS_FOR_L{w}"] - tg[f"PTS_AGAINST_L{w}"]

    keep = [
        "GAME_ID", "TEAM_ID", "GAME_DATE",
        "REST_DAYS", "B2B",
        "NET_PTS_L5", "NET_PTS_L10",
    ]
    return tg[keep].copy()

def make_game_level_features(games_elo: pd.DataFrame, team_feats: pd.DataFrame) -> pd.DataFrame:
    # Ensure one row per GAME_ID at the game table level
    games_elo = games_elo.copy()
    games_elo["GAME_DATE"] = pd.to_datetime(games_elo["GAME_DATE"])
    games_elo = games_elo.sort_values("GAME_DATE")

    # Hard dedupe: keep the latest row if duplicates exist
    games_elo = games_elo.drop_duplicates(subset=["GAME_ID"], keep="last").copy()

    home = team_feats.rename(columns={
        "TEAM_ID": "TEAM_ID_HOME",
        "GAME_DATE": "GAME_DATE_HOMEFEAT",
        "REST_DAYS": "REST_DAYS_HOME",
        "B2B": "B2B_HOME",
        "NET_PTS_L5": "NET_PTS_L5_HOME",
        "NET_PTS_L10": "NET_PTS_L10_HOME",
    })

    away = team_feats.rename(columns={
        "TEAM_ID": "TEAM_ID_AWAY",
        "GAME_DATE": "GAME_DATE_AWAYFEAT",
        "REST_DAYS": "REST_DAYS_AWAY",
        "B2B": "B2B_AWAY",
        "NET_PTS_L5": "NET_PTS_L5_AWAY",
        "NET_PTS_L10": "NET_PTS_L10_AWAY",
    })

    df = games_elo.merge(home, on=["GAME_ID", "TEAM_ID_HOME"], how="left")
    df = df.merge(away, on=["GAME_ID", "TEAM_ID_AWAY"], how="left")

    df["REST_DIFF"] = df["REST_DAYS_HOME"] - df["REST_DAYS_AWAY"]
    df["B2B_DIFF"] = df["B2B_HOME"] - df["B2B_AWAY"]
    df["NET5_DIFF"] = df["NET_PTS_L5_HOME"] - df["NET_PTS_L5_AWAY"]
    df["NET10_DIFF"] = df["NET_PTS_L10_HOME"] - df["NET_PTS_L10_AWAY"]

    return df

if __name__ == "__main__":
    team_games = pd.read_parquet("data/raw/team_games.parquet")
    games_elo = pd.read_parquet("data/processed/games_with_elo.parquet")

    team_feats = build_team_game_features(team_games)
    merged = make_game_level_features(games_elo, team_feats)

    merged = merged.dropna(subset=["NET10_DIFF", "NET5_DIFF", "REST_DIFF"]).copy()

    n_games = merged["GAME_ID"].nunique()
    n_rows = len(merged)
    print(f"Rows: {n_rows} | Unique games: {n_games}")

    # This should now always match
    if n_rows != n_games:
        dup = merged["GAME_ID"].value_counts().head(10)
        raise ValueError("Still duplicated GAME_IDs. Top duplicates:\n" + str(dup))

    merged.to_parquet("data/processed/games_features.parquet", index=False)
    print("Saved data/processed/games_features.parquet")
    print(merged[["GAME_ID","GAME_DATE","ELO_DIFF_PRE","REST_DIFF","NET10_DIFF","HOME_WIN"]].head())
