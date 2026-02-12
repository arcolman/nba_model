import pandas as pd
import numpy as np
from collections import defaultdict, deque
from joblib import load
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

# ---------------- Elo ----------------
def elo_expected(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

# ---------------- Main ----------------
if __name__ == "__main__":
    MODEL_PATH = "models/win_model_v2.joblib"  # change if needed
    GAMES_PATH = "data/processed/games.parquet"
    TEAM_GAMES_PATH = "data/raw/team_games.parquet"

    # Elo hyperparams (should match your features_elo.py)
    K = 20.0
    HOME_ADV = 65.0
    BASE = 1500.0

    # Rolling form windows
    W5 = 5
    W10 = 10

    # Load model bundle
    bundle = load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # Load data
    games = pd.read_parquet(GAMES_PATH).copy()
    team_games = pd.read_parquet(TEAM_GAMES_PATH).copy()

    # Normalize dates
    games["GAME_ID"] = games["GAME_ID"].astype(str)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="coerce")
    if games["GAME_DATE"].isna().any():
        raise ValueError("games.parquet has invalid GAME_DATE values (NaT).")

    # Expect these columns in games.parquet
    needed = {"GAME_ID", "GAME_DATE", "TEAM_ID_HOME", "TEAM_ID_AWAY", "HOME_WIN"}
    missing = needed - set(games.columns)
    if missing:
        raise ValueError(f"games.parquet missing columns: {missing}")

    games["TEAM_ID_HOME"] = games["TEAM_ID_HOME"].astype(int)
    games["TEAM_ID_AWAY"] = games["TEAM_ID_AWAY"].astype(int)
    games["HOME_WIN"] = games["HOME_WIN"].astype(int)

    # Build lookup for team_game margin (PLUS_MINUS) per team per game
    # We'll use PLUS_MINUS as "margin" for rolling NET features
    team_games["GAME_ID"] = team_games["GAME_ID"].astype(str)
    team_games["TEAM_ID"] = team_games["TEAM_ID"].astype(int)

    # Deduplicate to one row per (GAME_ID, TEAM_ID)
    tg = team_games.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="last").copy()

    if "PLUS_MINUS" not in tg.columns:
        raise ValueError("team_games.parquet missing PLUS_MINUS column (needed for rolling NET).")

    margin_lookup = {(r.GAME_ID, int(r.TEAM_ID)): float(r.PLUS_MINUS) for r in tg.itertuples(index=False)}

    # Determine "last month" window based on your dataset's latest game date
    start_date = games["GAME_DATE"].min().normalize()
    max_date = games["GAME_DATE"].max().normalize()


    print(f"Backtesting full dataset: {start_date.date()} -> {max_date.date()}")
    print(f"Model: {MODEL_PATH}")

    # Sort games chronologically
    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    # State
    elo = defaultdict(lambda: BASE)
    last_played = {}  # TEAM_ID -> last game date (normalized)
    last5 = defaultdict(lambda: deque(maxlen=W5))   # TEAM_ID -> margins
    last10 = defaultdict(lambda: deque(maxlen=W10)) # TEAM_ID -> margins

    rows = []

    for g in games.itertuples(index=False):
        game_id = g.GAME_ID
        d = pd.Timestamp(g.GAME_DATE).normalize()
        home = int(g.TEAM_ID_HOME)
        away = int(g.TEAM_ID_AWAY)
        y = int(g.HOME_WIN)

        # --- Build features using ONLY prior info (no leakage) ---
        r_home = float(elo[home])
        r_away = float(elo[away])

        # Rest days
        def rest_days(team_id):
            if team_id not in last_played:
                return 3.0  # default for first game
            return float((d - last_played[team_id]).days)

        rest_h = rest_days(home)
        rest_a = rest_days(away)
        b2b_h = 1.0 if rest_h == 0.0 else 0.0
        b2b_a = 1.0 if rest_a == 0.0 else 0.0

        # Rolling NET = average margin over last N games (margin = PLUS_MINUS)
        def avg_deque(q):
            return float(np.mean(q)) if len(q) > 0 else 0.0

        net5_h = avg_deque(last5[home])
        net5_a = avg_deque(last5[away])
        net10_h = avg_deque(last10[home])
        net10_a = avg_deque(last10[away])

        feats = {
            "ELO_DIFF_PRE": (r_home + HOME_ADV) - r_away,
            "REST_DIFF": rest_h - rest_a,
            "B2B_DIFF": b2b_h - b2b_a,
            "NET5_DIFF": net5_h - net5_a,
            "NET10_DIFF": net10_h - net10_a,
        }

        # Build X in expected column order; if model expects extra cols, fill with 0
        X = np.array([[feats.get(c, 0.0) for c in feature_cols]], dtype=float)
        p_home = float(model.predict_proba(X)[0][1])

        # Record if within last-month window
        if d >= start_date and d <= max_date:
            rows.append({
                "GAME_ID": game_id,
                "GAME_DATE": str(d.date()),
                "TEAM_ID_HOME": home,
                "TEAM_ID_AWAY": away,
                "HOME_WIN": y,
                "P_HOME": p_home,
                **{k: feats.get(k, 0.0) for k in feats.keys()},
            })

        # --- Update state AFTER the game (so future games can use it) ---
        # Elo update
        exp_home = elo_expected(r_home + HOME_ADV, r_away)
        elo[home] = r_home + K * (y - exp_home)
        elo[away] = r_away + K * ((1 - y) - (1 - exp_home))

        # Rolling margins update (use PLUS_MINUS from team_games)
        m_home = margin_lookup.get((game_id, home))
        m_away = margin_lookup.get((game_id, away))

        # If missing, skip update (rare; but safe)
        if m_home is not None:
            last5[home].append(m_home)
            last10[home].append(m_home)
        if m_away is not None:
            last5[away].append(m_away)
            last10[away].append(m_away)

        # Last played update
        last_played[home] = d
        last_played[away] = d

    if not rows:
        print("No games found in the last-month window. (Dataset may be too small.)")
        raise SystemExit

    out = pd.DataFrame(rows)
    y_true = out["HOME_WIN"].astype(int).values
    p = out["P_HOME"].astype(float).values
    y_pred = (p >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, p, labels=[0, 1])
    brier = brier_score_loss(y_true, p)

    print("\nLAST MONTH MODEL PERFORMANCE")
    print("--------------------------------------------------")
    print(f"Games: {len(out)}")
    print(f"Accuracy: {acc:.3f}")
    print(f"LogLoss : {ll:.3f}")
    print(f"Brier   : {brier:.3f}")

    # Simple calibration buckets
    out["bucket"] = pd.cut(out["P_HOME"], bins=np.linspace(0, 1, 11), include_lowest=True)
    cal = out.groupby("bucket").agg(
        n=("HOME_WIN", "size"),
        avg_p=("P_HOME", "mean"),
        win_rate=("HOME_WIN", "mean"),
    ).reset_index()
    print("\nCALIBRATION (binned)")
    print("--------------------------------------------------")
    print(cal.to_string(index=False))

    out_path = "data/processed/backtest_last_month_model.csv"
    out.to_csv(out_path, index=False)
    print("\nSaved detailed rows to:", out_path)
