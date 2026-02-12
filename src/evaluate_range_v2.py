import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from nba_api.stats.endpoints import scoreboardv2
import requests_cache
from datetime import datetime, timedelta

requests_cache.install_cache("nba_cache", expire_after=60*60)

def elo_expected(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

def compute_current_elos():
    games = pd.read_parquet("data/processed/games.parquet")
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games = games.sort_values("GAME_DATE")

    ratings = {}
    for _, g in games.iterrows():
        h = int(g["TEAM_ID_HOME"]); a = int(g["TEAM_ID_AWAY"]); y = int(g["HOME_WIN"])
        r_h = ratings.get(h, 1500.0); r_a = ratings.get(a, 1500.0)
        exp_h = elo_expected(r_h + 65.0, r_a)
        ratings[h] = r_h + 20.0 * (y - exp_h)
        ratings[a] = r_a + 20.0 * ((1 - y) - (1 - exp_h))
    return ratings

def latest_team_form_asof(asof_date):
    tg = pd.read_parquet("data/raw/team_games.parquet")
    tg["GAME_DATE"] = pd.to_datetime(tg["GAME_DATE"])
    asof = pd.to_datetime(asof_date)

    tg = tg[tg["GAME_DATE"] < asof].copy()
    tg = tg.sort_values(["TEAM_ID", "GAME_DATE"])

    tg = tg.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="last").copy()
    tg["OPP_PTS"] = tg["PTS"] - tg["PLUS_MINUS"]

    for w in [5, 10]:
        tg[f"PTS_FOR_L{w}"] = tg.groupby("TEAM_ID")["PTS"].rolling(w).mean().reset_index(level=0, drop=True)
        tg[f"PTS_AGAINST_L{w}"] = tg.groupby("TEAM_ID")["OPP_PTS"].rolling(w).mean().reset_index(level=0, drop=True)
        tg[f"NET_PTS_L{w}"] = tg[f"PTS_FOR_L{w}"] - tg[f"PTS_AGAINST_L{w}"]

    last = tg.groupby("TEAM_ID").tail(1).copy()
    last["REST_ASOF"] = (asof - last["GAME_DATE"]).dt.days.clip(lower=0)
    last["B2B_ASOF"] = (last["REST_ASOF"] == 0).astype(int)

    return last.set_index("TEAM_ID")[["REST_ASOF","B2B_ASOF","NET_PTS_L10","NET_PTS_L5"]].to_dict(orient="index")

def daterange(start_date, end_date):
    d0 = datetime.strptime(start_date, "%m/%d/%Y")
    d1 = datetime.strptime(end_date, "%m/%d/%Y")
    d = d0
    while d <= d1:
        yield d.strftime("%m/%d/%Y")
        d += timedelta(days=1)

if __name__ == "__main__":
    bundle = load("models/win_model_v2.joblib")
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # Pick a finished window (past dates)
    START = "01/01/2025"
    END   = "01/31/2025"

    elos = compute_current_elos()

    rows = []
    all_preds = []
    all_actuals = []

    for game_date in daterange(START, END):
        frames = scoreboardv2.ScoreboardV2(game_date=game_date).get_data_frames()
        game_header = frames[0]
        line_score  = frames[1]

        if game_header.empty:
            continue

        # lookups from line_score
        ls = line_score[["GAME_ID","TEAM_ID","TEAM_ABBREVIATION","PTS"]].drop_duplicates()
        abbr = ls.set_index(["GAME_ID","TEAM_ID"])["TEAM_ABBREVIATION"].to_dict()
        pts  = ls.set_index(["GAME_ID","TEAM_ID"])["PTS"].to_dict()

        form = latest_team_form_asof(game_date)

        for _, g in game_header.iterrows():
            game_id = g["GAME_ID"]
            home_id = int(g["HOME_TEAM_ID"])
            away_id = int(g["VISITOR_TEAM_ID"])

            home_ab = abbr.get((game_id, home_id), str(home_id))
            away_ab = abbr.get((game_id, away_id), str(away_id))

            # If not final / missing points, skip (no actual)
            home_pts = pts.get((game_id, home_id), None)
            away_pts = pts.get((game_id, away_id), None)
            if home_pts is None or away_pts is None:
                continue

            actual = 1 if float(home_pts) > float(away_pts) else 0

            r_home = elos.get(home_id, 1500.0)
            r_away = elos.get(away_id, 1500.0)

            hf = form.get(home_id, {"REST_ASOF":3,"B2B_ASOF":0,"NET_PTS_L10":0.0,"NET_PTS_L5":0.0})
            af = form.get(away_id, {"REST_ASOF":3,"B2B_ASOF":0,"NET_PTS_L10":0.0,"NET_PTS_L5":0.0})

            feats = {
                "ELO_DIFF_PRE": (r_home + 65.0) - r_away,
                "REST_DIFF": float(hf["REST_ASOF"] - af["REST_ASOF"]),
                "B2B_DIFF": float(hf["B2B_ASOF"] - af["B2B_ASOF"]),
                "NET10_DIFF": float(hf["NET_PTS_L10"] - af["NET_PTS_L10"]),
                "NET5_DIFF": float(hf["NET_PTS_L5"] - af["NET_PTS_L5"]),
            }

            X = np.array([[feats[c] for c in feature_cols]], dtype=float)
            p_home = model.predict_proba(X)[0][1]

            rows.append({
                "date": game_date,
                "game_id": game_id,
                "away": away_ab,
                "home": home_ab,
                "away_pts": float(away_pts),
                "home_pts": float(home_pts),
                "p_home": float(p_home),
                "actual_home_win": int(actual),
                "pred_home_win": int(p_home > 0.5),
            })

            all_preds.append(float(p_home))
            all_actuals.append(int(actual))

    out = pd.DataFrame(rows)
    out.to_csv("data/processed/eval_range_v2.csv", index=False)

    print(f"Saved data/processed/eval_range_v2.csv with {len(out)} games")

    if len(all_actuals) > 0:
        all_preds = np.array(all_preds)
        all_actuals = np.array(all_actuals)

        print("\nOverall metrics:")
        print("Games:", len(all_actuals))
        print("Accuracy:", accuracy_score(all_actuals, all_preds > 0.5))
        print("LogLoss:", log_loss(all_actuals, all_preds))
        print("Brier:", brier_score_loss(all_actuals, all_preds))
    else:
        print("No final games found in that date range.")
