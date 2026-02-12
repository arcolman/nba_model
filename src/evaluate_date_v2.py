import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from nba_api.stats.endpoints import scoreboardv2
import requests_cache

requests_cache.install_cache("nba_cache", expire_after=60*60)

# ---- Elo helpers ----
def elo_expected(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

def compute_current_elos():
    games = pd.read_parquet("data/processed/games.parquet")
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games = games.sort_values("GAME_DATE")

    ratings = {}
    for _, g in games.iterrows():
        h = int(g["TEAM_ID_HOME"])
        a = int(g["TEAM_ID_AWAY"])
        y = int(g["HOME_WIN"])

        r_h = ratings.get(h, 1500.0)
        r_a = ratings.get(a, 1500.0)

        exp_h = elo_expected(r_h + 65.0, r_a)
        r_h_new = r_h + 20.0 * (y - exp_h)
        r_a_new = r_a + 20.0 * ((1 - y) - (1 - exp_h))

        ratings[h] = r_h_new
        ratings[a] = r_a_new

    return ratings

def latest_team_form_asof(asof_date):
    tg = pd.read_parquet("data/raw/team_games.parquet")
    tg["GAME_DATE"] = pd.to_datetime(tg["GAME_DATE"])
    asof = pd.to_datetime(asof_date)

    tg = tg[tg["GAME_DATE"] < asof].copy()
    tg = tg.sort_values(["TEAM_ID", "GAME_DATE"])

    tg["OPP_PTS"] = tg["PTS"] - tg["PLUS_MINUS"]

    for w in [5, 10]:
        tg[f"PTS_FOR_L{w}"] = tg.groupby("TEAM_ID")["PTS"].rolling(w).mean().reset_index(level=0, drop=True)
        tg[f"PTS_AGAINST_L{w}"] = tg.groupby("TEAM_ID")["OPP_PTS"].rolling(w).mean().reset_index(level=0, drop=True)
        tg[f"NET_PTS_L{w}"] = tg[f"PTS_FOR_L{w}"] - tg[f"PTS_AGAINST_L{w}"]

    last = tg.groupby("TEAM_ID").tail(1).copy()
    last["REST_ASOF"] = (asof - last["GAME_DATE"]).dt.days.clip(lower=0)
    last["B2B_ASOF"] = (last["REST_ASOF"] == 0).astype(int)

    return last.set_index("TEAM_ID")[["REST_ASOF","B2B_ASOF","NET_PTS_L10","NET_PTS_L5"]].to_dict(orient="index")

if __name__ == "__main__":
    bundle = load("models/win_model_v2.joblib")
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # IMPORTANT: use a date where games have already finished
    game_date = "02/11/2026"  # MM/DD/YYYY

    frames = scoreboardv2.ScoreboardV2(game_date=game_date).get_data_frames()
    game_header = frames[0]   # GameHeader
    line_score  = frames[1]   # LineScore (has team abbreviations + points)

    if game_header.empty:
        print("No games found for", game_date)
        raise SystemExit

    # Build lookup: (GAME_ID, TEAM_ID) -> (ABBR, PTS)
    ls = line_score[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"]].drop_duplicates()
    abbr = ls.set_index(["GAME_ID", "TEAM_ID"])["TEAM_ABBREVIATION"].to_dict()
    pts  = ls.set_index(["GAME_ID", "TEAM_ID"])["PTS"].to_dict()

    elos = compute_current_elos()
    form = latest_team_form_asof(game_date)

    preds = []
    actuals = []

    print(f"Evaluation for {game_date}")
    print("-" * 70)

    for _, g in game_header.iterrows():
        game_id = g["GAME_ID"]
        home_id = int(g["HOME_TEAM_ID"])
        away_id = int(g["VISITOR_TEAM_ID"])

        home_abbr = abbr.get((game_id, home_id), str(home_id))
        away_abbr = abbr.get((game_id, away_id), str(away_id))

        # Pred features
        r_home = elos.get(home_id, 1500.0)
        r_away = elos.get(away_id, 1500.0)

        home_form = form.get(home_id, {"REST_ASOF":3,"B2B_ASOF":0,"NET_PTS_L10":0.0,"NET_PTS_L5":0.0})
        away_form = form.get(away_id, {"REST_ASOF":3,"B2B_ASOF":0,"NET_PTS_L10":0.0,"NET_PTS_L5":0.0})

        row = {
            "ELO_DIFF_PRE": (r_home + 65.0) - r_away,
            "REST_DIFF": float(home_form["REST_ASOF"] - away_form["REST_ASOF"]),
            "B2B_DIFF": float(home_form["B2B_ASOF"] - away_form["B2B_ASOF"]),
            "NET10_DIFF": float(home_form["NET_PTS_L10"] - away_form["NET_PTS_L10"]),
            "NET5_DIFF": float(home_form["NET_PTS_L5"] - away_form["NET_PTS_L5"]),
        }

        X = np.array([[row[c] for c in feature_cols]], dtype=float)
        p_home = model.predict_proba(X)[0][1]

        # Actual result from LineScore points (only works if game is final and PTS is present)
        home_pts = pts.get((game_id, home_id), None)
        away_pts = pts.get((game_id, away_id), None)

        if home_pts is None or away_pts is None:
            # Game might not be final / missing data
            actual = None
        else:
            actual = 1 if float(home_pts) > float(away_pts) else 0

        preds.append(p_home)
        actuals.append(actual)

        actual_str = "NA" if actual is None else str(actual)
        score_str = "" if (home_pts is None or away_pts is None) else f" | Score {away_pts}-{home_pts}"
        print(f"{away_abbr} @ {home_abbr}{score_str}  | Pred P(home)={p_home:.3f} | Actual={actual_str}")

    # Keep only games with known actuals
    mask = [a is not None for a in actuals]
    preds = np.array([p for p, m in zip(preds, mask) if m], dtype=float)
    actuals = np.array([a for a in actuals if a is not None], dtype=int)

    if len(actuals) == 0:
        print("\nNo final scores available for that date yet. Try a past date where games are finished.")
        raise SystemExit

    print("\nDaily Metrics (final games only):")
    print("Games:", len(actuals))
    print("Accuracy:", accuracy_score(actuals, preds > 0.5))
    print("LogLoss:", log_loss(actuals, preds))
    print("Brier:", brier_score_loss(actuals, preds))
