import pandas as pd
import numpy as np
from joblib import load
from nba_api.stats.endpoints import scoreboardv2
import requests_cache

requests_cache.install_cache("nba_cache", expire_after=60*60)

# ---- Elo helpers (same as training) ----
def elo_expected(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

def compute_current_elos(games_parquet="data/processed/games.parquet", k=20.0, home_adv=65.0, base=1500.0):
    games = pd.read_parquet(games_parquet)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games = games.sort_values("GAME_DATE")

    ratings = {}
    for _, g in games.iterrows():
        h = int(g["TEAM_ID_HOME"])
        a = int(g["TEAM_ID_AWAY"])
        y = int(g["HOME_WIN"])

        r_h = ratings.get(h, base)
        r_a = ratings.get(a, base)

        exp_h = elo_expected(r_h + home_adv, r_a)
        r_h_new = r_h + k * (y - exp_h)
        r_a_new = r_a + k * ((1 - y) - (1 - exp_h))

        ratings[h] = r_h_new
        ratings[a] = r_a_new

    return ratings

# ---- Team rolling + rest features as of a given date ----
def latest_team_form_asof(team_games_parquet="data/raw/team_games.parquet", asof_date="2025-01-15"):
    tg = pd.read_parquet(team_games_parquet)
    tg["GAME_DATE"] = pd.to_datetime(tg["GAME_DATE"])
    asof = pd.to_datetime(asof_date)

    # only games strictly before the target date (no leakage)
    tg = tg[tg["GAME_DATE"] < asof].copy()
    tg = tg.sort_values(["TEAM_ID", "GAME_DATE"])

    # opponent points
    tg["OPP_PTS"] = tg["PTS"] - tg["PLUS_MINUS"]

    # rest days for each game (days since previous game)
    tg["REST_DAYS"] = tg.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days
    tg["REST_DAYS"] = tg["REST_DAYS"].fillna(7).clip(lower=0)
    tg["B2B"] = (tg["REST_DAYS"] == 0).astype(int)

    # rolling windows (shift(1) not needed here because we’re taking the latest completed game)
    for w in [5, 10]:
        tg[f"PTS_FOR_L{w}"] = tg.groupby("TEAM_ID")["PTS"].rolling(w).mean().reset_index(level=0, drop=True)
        tg[f"PTS_AGAINST_L{w}"] = tg.groupby("TEAM_ID")["OPP_PTS"].rolling(w).mean().reset_index(level=0, drop=True)
        tg[f"NET_PTS_L{w}"] = tg[f"PTS_FOR_L{w}"] - tg[f"PTS_AGAINST_L{w}"]

    # take latest row per team (most recent completed game)
    last = tg.groupby("TEAM_ID").tail(1).copy()

    # rest as of date = days since last game date
    last["REST_ASOF"] = (asof - last["GAME_DATE"]).dt.days.clip(lower=0)
    last["B2B_ASOF"] = (last["REST_ASOF"] == 0).astype(int)

    # return lookup per team
    out = last.set_index("TEAM_ID")[["REST_ASOF", "B2B_ASOF", "NET_PTS_L10", "NET_PTS_L5"]].to_dict(orient="index")
    return out

if __name__ == "__main__":
    bundle = load("models/win_model_v2.joblib")
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # change this date
    game_date = "02/11/2026"  # MM/DD/YYYY

    # get schedule/games
    frames = scoreboardv2.ScoreboardV2(game_date=game_date).get_data_frames()
    game_header = frames[0]
    line_score = frames[1]

    if game_header.empty:
        print("No games found for", game_date)
        raise SystemExit

    # abbreviations lookup
    abbr = (
        line_score[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"]]
        .drop_duplicates()
        .set_index(["GAME_ID", "TEAM_ID"])["TEAM_ABBREVIATION"]
        .to_dict()
    )

    # features as of that date (no leakage)
    elos = compute_current_elos()
    form = latest_team_form_asof(asof_date=pd.to_datetime(game_date).strftime("%Y-%m-%d"))

    print("Predictions (v2) for", game_date)
    print("-" * 60)

    for _, g in game_header.iterrows():
        game_id = g["GAME_ID"]
        home_id = int(g["HOME_TEAM_ID"])
        away_id = int(g["VISITOR_TEAM_ID"])

        home_abbr = abbr.get((game_id, home_id), str(home_id))
        away_abbr = abbr.get((game_id, away_id), str(away_id))

        r_home = elos.get(home_id, 1500.0)
        r_away = elos.get(away_id, 1500.0)

        home_form = form.get(home_id, {"REST_ASOF": 3, "B2B_ASOF": 0, "NET_PTS_L10": 0.0, "NET_PTS_L5": 0.0})
        away_form = form.get(away_id, {"REST_ASOF": 3, "B2B_ASOF": 0, "NET_PTS_L10": 0.0, "NET_PTS_L5": 0.0})

        row = {
            "ELO_DIFF_PRE": (r_home + 65.0) - r_away,
            "REST_DIFF": float(home_form["REST_ASOF"] - away_form["REST_ASOF"]),
            "B2B_DIFF": float(home_form["B2B_ASOF"] - away_form["B2B_ASOF"]),
            "NET10_DIFF": float(home_form["NET_PTS_L10"] - away_form["NET_PTS_L10"]),
            "NET5_DIFF": float(home_form["NET_PTS_L5"] - away_form["NET_PTS_L5"]),
        }

        X = np.array([[row[c] for c in feature_cols]], dtype=float)
        p_home = model.predict_proba(X)[0][1]

        print(f"{away_abbr} @ {home_abbr}  |  P(home win) = {p_home:.3f}")
