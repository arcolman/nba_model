import pandas as pd
import numpy as np
from joblib import load
from nba_api.stats.endpoints import scoreboardv2
import requests_cache

requests_cache.install_cache("nba_cache", expire_after=60*60)

def get_latest_elos():
    df = pd.read_parquet("data/processed/games_with_elo.parquet")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE")

    elos = {}
    for _, g in df.iterrows():
        elos[int(g["TEAM_ID_HOME"])] = float(g["ELO_HOME_PRE"])
        elos[int(g["TEAM_ID_AWAY"])] = float(g["ELO_AWAY_PRE"])
    return elos

if __name__ == "__main__":
    model = load("models/win_model.joblib")
    elos = get_latest_elos()

    # Change this to any date like "02/11/2026"
    game_date = "02/11/2026"

    frames = scoreboardv2.ScoreboardV2(game_date=game_date).get_data_frames()
    game_header = frames[0]   # GameHeader
    line_score  = frames[1]   # LineScore (has abbreviations)

    if game_header.empty:
        print("No games found for", game_date)
        raise SystemExit

    # Build a lookup for abbreviations from LineScore
    abbr = (
        line_score[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"]]
        .drop_duplicates()
        .set_index(["GAME_ID", "TEAM_ID"])["TEAM_ABBREVIATION"]
        .to_dict()
    )

    print("Predictions for", game_date)
    print("-" * 50)

    for _, g in game_header.iterrows():
        game_id = g["GAME_ID"]
        home_id = int(g["HOME_TEAM_ID"])
        away_id = int(g["VISITOR_TEAM_ID"])

        home_abbr = abbr.get((game_id, home_id), str(home_id))
        away_abbr = abbr.get((game_id, away_id), str(away_id))

        r_home = elos.get(home_id, 1500.0)
        r_away = elos.get(away_id, 1500.0)

        elo_diff = np.array([[ (r_home + 65.0) - r_away ]])
        p_home = model.predict_proba(elo_diff)[0][1]

        print(f"{away_abbr} @ {home_abbr}  |  P(home win) = {p_home:.3f}")
