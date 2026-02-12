import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from time import sleep
from joblib import load
from sklearn.metrics import accuracy_score
from nba_api.stats.endpoints import scoreboardv2
import requests_cache
import requests

# Cache responses so reruns don't re-hit the NBA site
requests_cache.install_cache("nba_cache", expire_after=60 * 60 * 24)

# ---------- Elo helpers (walk-forward, no peeking) ----------
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

# ---------- Team “form as of date” ----------
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

def daterange(start: datetime, end: datetime):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

# ---------- Robust fetch with retries/backoff ----------
def fetch_scoreboard_frames(game_date_str: str, timeout=90, max_retries=4, base_sleep=1.25):
    """
    Returns (game_header_df, line_score_df) or (None, None) if fails.
    """
    for attempt in range(max_retries):
        try:
            # nba_api endpoints accept timeout=
            frames = scoreboardv2.ScoreboardV2(game_date=game_date_str, timeout=timeout).get_data_frames()
            return frames[0], frames[1]
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            wait = base_sleep * (2 ** attempt)
            print(f"[{game_date_str}] timeout/conn error (attempt {attempt+1}/{max_retries}) -> sleeping {wait:.1f}s")
            sleep(wait)
        except Exception as e:
            # any other error: log and give up for that date
            print(f"[{game_date_str}] error: {type(e).__name__}: {e}")
            return None, None
    print(f"[{game_date_str}] failed after {max_retries} retries")
    return None, None

if __name__ == "__main__":
    bundle = load("models/win_model_v2.joblib")
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    games = pd.read_parquet("data/processed/games.parquet")
    team_games = pd.read_parquet("data/raw/team_games.parquet")

    advance_elo = build_elo_stream(games, k=20.0, home_adv=65.0, base=1500.0)

    today = datetime.today().date()
    end_date = today - timedelta(days=1)       # yesterday (more likely complete)
    start_date = end_date - timedelta(days=29) # last 30 days

    all_rows = []
    days_hit = 0
    days_failed = 0

    for day in daterange(datetime.combine(start_date, datetime.min.time()),
                         datetime.combine(end_date, datetime.min.time())):
        game_date_str = day.strftime("%m/%d/%Y")
        asof = pd.to_datetime(day.date())

        # gentle pacing to reduce throttling
        sleep(0.6)

        elos = advance_elo(asof)
        game_header, line_score = fetch_scoreboard_frames(game_date_str, timeout=90)

        if game_header is None:
            days_failed += 1
            continue

        if game_header.empty:
            continue

        days_hit += 1

        form = team_form_asof(team_games, asof)

        ls = line_score[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"]].drop_duplicates()
        abbr = ls.set_index(["GAME_ID", "TEAM_ID"])["TEAM_ABBREVIATION"].to_dict()
        pts = ls.set_index(["GAME_ID", "TEAM_ID"])["PTS"].to_dict()

        for _, g in game_header.iterrows():
            game_id = g["GAME_ID"]
            home_id = int(g["HOME_TEAM_ID"])
            away_id = int(g["VISITOR_TEAM_ID"])

            home_pts = pts.get((game_id, home_id), None)
            away_pts = pts.get((game_id, away_id), None)

            # only evaluate finished games
            if home_pts is None or away_pts is None:
                continue

            actual_home_win = 1 if float(home_pts) > float(away_pts) else 0

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
            p_home = float(model.predict_proba(X)[0][1])
            pred_home_win = 1 if p_home > 0.5 else 0

            home_ab = abbr.get((game_id, home_id), str(home_id))
            away_ab = abbr.get((game_id, away_id), str(away_id))

            all_rows.append({
                "date": game_date_str,
                "game_id": game_id,
                "away": away_ab,
                "home": home_ab,
                "away_pts": float(away_pts),
                "home_pts": float(home_pts),
                "p_home": p_home,
                "pred_home_win": pred_home_win,
                "actual_home_win": actual_home_win,
                "correct": int(pred_home_win == actual_home_win),
            })

    if len(all_rows) == 0:
        print("No finished games found in the last 30 days (or NBA endpoint blocked).")
        print("Try rerunning (cache will help), or widen dates to an in-season month.")
        raise SystemExit

    df = pd.DataFrame(all_rows)
    df.to_csv("data/processed/last_month_eval.csv", index=False)

    acc = accuracy_score(df["actual_home_win"], df["pred_home_win"])
    print("\n========== Last 30 days report ==========")
    print(f"Days requested: 30 | days with games fetched: {days_hit} | days failed: {days_failed}")
    print(f"Finished games evaluated: {len(df)}")
    print(f"Accuracy: {acc:.4f}")
    print("Saved: data/processed/last_month_eval.csv")
