import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from time import sleep
from joblib import load
from nba_api.stats.endpoints import scoreboardv2
import requests_cache

requests_cache.install_cache("nba_cache", expire_after=60 * 60)  # cache scoreboard calls

# ----------------- Elo helpers (walk-forward) -----------------
def elo_expected(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

def build_elo_stream(games_df: pd.DataFrame, k=20.0, home_adv=65.0, base=1500.0):
    games = games_df.copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games = games.sort_values("GAME_DATE").reset_index(drop=True)

    ratings = {}
    idx = 0

    def advance_until(cutoff_date: pd.Timestamp):
        nonlocal idx, ratings
        while idx < len(games) and games.loc[idx, "GAME_DATE"] < cutoff_date:
            g = games.loc[idx]
            h = int(g["TEAM_ID_HOME"])
            a = int(g["TEAM_ID_AWAY"])
            y = int(g["HOME_WIN"])

            r_h = ratings.get(h, base)
            r_a = ratings.get(a, base)

            exp_h = elo_expected(r_h + home_adv, r_a)
            ratings[h] = r_h + k * (y - exp_h)
            ratings[a] = r_a + k * ((1 - y) - (1 - exp_h))
            idx += 1
        return ratings

    return advance_until

# ----------------- Team form as-of (no leakage) -----------------
def team_form_asof(team_games: pd.DataFrame, asof_date: pd.Timestamp):
    tg = team_games.copy()
    tg["GAME_DATE"] = pd.to_datetime(tg["GAME_DATE"])
    tg = tg[tg["GAME_DATE"] < asof_date].copy()
    tg = tg.sort_values(["TEAM_ID", "GAME_DATE"])

    tg = tg.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="last").copy()

    tg["OPP_PTS"] = tg["PTS"] - tg["PLUS_MINUS"]

    for w in [5, 10]:
        pts_for = tg.groupby("TEAM_ID")["PTS"].shift(1).rolling(w).mean().reset_index(level=0, drop=True)
        pts_against = tg.groupby("TEAM_ID")["OPP_PTS"].shift(1).rolling(w).mean().reset_index(level=0, drop=True)
        tg[f"NET_L{w}"] = pts_for - pts_against

    last = tg.groupby("TEAM_ID").tail(1).copy()
    last["REST_ASOF"] = (asof_date - last["GAME_DATE"]).dt.days.clip(lower=0)
    last["B2B_ASOF"] = (last["REST_ASOF"] == 0).astype(int)

    return last.set_index("TEAM_ID")[["REST_ASOF", "B2B_ASOF", "NET_L5", "NET_L10"]].to_dict(orient="index")

# ----------------- Odds helpers -----------------
def load_latest_odds(path="data/raw/odds_snapshots.parquet") -> pd.DataFrame:
    odds = pd.read_parquet(path)
    odds["snapshot_utc"] = odds["snapshot_utc"].astype(str)
    latest = odds["snapshot_utc"].max()
    odds = odds[odds["snapshot_utc"] == latest].copy()
    return odds

def get_market_prob(odds_df: pd.DataFrame, home_abbr: str, away_abbr: str, date_candidates):
    """
    date_candidates: list like ["2026-02-12","2026-02-13"]
    Returns (p_home_mkt, matched_game_date) or (None, None)
    """
    row = odds_df[
        (odds_df["home_abbr"] == home_abbr) &
        (odds_df["away_abbr"] == away_abbr) &
        (odds_df["game_date"].isin(date_candidates))
    ]
    if len(row) == 0:
        return None, None

    row = row.sort_values("commence_time_utc").iloc[0]
    return float(row["p_home_mkt"]), str(row["game_date"])

# ----------------- Main -----------------
if __name__ == "__main__":
    MODEL_PATH = "models/win_model_v2.joblib"  # change to v3 if you want

    bundle = load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    games = pd.read_parquet("data/processed/games.parquet")
    team_games = pd.read_parquet("data/raw/team_games.parquet")

    advance_elo = build_elo_stream(games, k=20.0, home_adv=65.0, base=1500.0)
    odds = load_latest_odds("data/raw/odds_snapshots.parquet")

    today_mmddyyyy = datetime.today().strftime("%m/%d/%Y")
    today_date = datetime.strptime(today_mmddyyyy, "%m/%d/%Y").date()
    today_yyyy_mm_dd = today_date.strftime("%Y-%m-%d")
    tomorrow_yyyy_mm_dd = (today_date + timedelta(days=1)).strftime("%Y-%m-%d")
    date_candidates = [today_yyyy_mm_dd, tomorrow_yyyy_mm_dd]

    asof = pd.to_datetime(today_date)

    frames = scoreboardv2.ScoreboardV2(game_date=today_mmddyyyy, timeout=60).get_data_frames()
    game_header = frames[0]
    line_score = frames[1]

    if game_header.empty:
        print(f"No games found on {today_mmddyyyy}")
        raise SystemExit

    ls = line_score[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"]].drop_duplicates()
    abbr_lookup = ls.set_index(["GAME_ID", "TEAM_ID"])["TEAM_ABBREVIATION"].to_dict()

    elos = advance_elo(asof)
    form = team_form_asof(team_games, asof)

    print(f"\nPredictions w/ Market Blend for {today_mmddyyyy}")
    print(f"Model file: {MODEL_PATH}")
    print("-" * 70)

    out_rows = []

    for _, g in game_header.iterrows():
        game_id = g["GAME_ID"]
        home_id = int(g["HOME_TEAM_ID"])
        away_id = int(g["VISITOR_TEAM_ID"])

        home_abbr = abbr_lookup.get((game_id, home_id), str(home_id))
        away_abbr = abbr_lookup.get((game_id, away_id), str(away_id))

        r_home = float(elos.get(home_id, 1500.0))
        r_away = float(elos.get(away_id, 1500.0))

        hf = form.get(home_id, {"REST_ASOF": 3, "B2B_ASOF": 0, "NET_L10": 0.0, "NET_L5": 0.0})
        af = form.get(away_id, {"REST_ASOF": 3, "B2B_ASOF": 0, "NET_L10": 0.0, "NET_L5": 0.0})

        feats = {
            "ELO_DIFF_PRE": (r_home + 65.0) - r_away,
            "REST_DIFF": float(hf["REST_ASOF"] - af["REST_ASOF"]),
            "B2B_DIFF": float(hf["B2B_ASOF"] - af["B2B_ASOF"]),
            "NET10_DIFF": float(hf["NET_L10"] - af["NET_L10"]),
            "NET5_DIFF": float(hf["NET_L5"] - af["NET_L5"]),
        }

        X = np.array([[feats[c] for c in feature_cols]], dtype=float)
        p_model = float(model.predict_proba(X)[0][1])

        # ✅ Correct call: pass a LIST of date candidates
        p_mkt, matched_date = get_market_prob(odds, home_abbr, away_abbr, date_candidates)

        if p_mkt is None:
            p_final = None
            edge = None
            print(f"{away_abbr} @ {home_abbr} | model={p_model:.3f} | market=NA (no odds match)")
        else:
            p_final = 0.65 * p_mkt + 0.35 * p_model
            edge = p_final - p_mkt
            print(f"{away_abbr} @ {home_abbr}")
            print(f"  Market : {p_mkt:.3f}  (odds date {matched_date})")
            print(f"  Model  : {p_model:.3f}")
            print(f"  Final  : {p_final:.3f}   (edge vs mkt {edge:+.3f})")

        out_rows.append({
            "date": today_yyyy_mm_dd,
            "game_id": game_id,
            "away": away_abbr,
            "home": home_abbr,
            "p_market_home": p_mkt,
            "p_model_home": p_model,
            "p_final_home": p_final,
            "edge_final_minus_market": edge,
            "market_game_date": matched_date,
        })

        sleep(0.05)

    out = pd.DataFrame(out_rows)
    out_path = "data/processed/preds_with_market_log.csv"
    import os
    out.to_csv(out_path, mode="a", index=False, header=not os.path.exists(out_path))


    print("-" * 70)
    print(f"Saved: {out_path}")
