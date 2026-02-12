import pandas as pd
import numpy as np

# ---------- odds conversions ----------
def american_to_decimal(ml: float) -> float:
    ml = float(ml)
    if ml > 0:
        return 1.0 + (ml / 100.0)
    else:
        return 1.0 + (100.0 / (-ml))

def ev_per_1u(p: float, dec: float) -> float:
    # stake 1 unit, win profit = dec-1, lose = -1
    return p * (dec - 1.0) - (1.0 - p)

def pnl_per_1u(win: int, dec: float) -> float:
    return (dec - 1.0) if win == 1 else -1.0

if __name__ == "__main__":
    PRED_PATH = "data/processed/preds_with_market.csv"   # <- your current file
    ODDS_PATH = "data/raw/odds_snapshots.parquet"
    GAMES_PATH = "data/processed/games.parquet"

    # Filters
    MIN_EV = 0.02          # +0.02 = +2% expected value per 1u stake (reasonable)
    PRINT_TOP = 15         # show top opportunities by EV even if no results yet

    preds = pd.read_csv(PRED_PATH)
    odds = pd.read_parquet(ODDS_PATH)
    games = pd.read_parquet(GAMES_PATH)

    # Normalize dates
    preds["date"] = pd.to_datetime(preds["date"]).dt.date.astype(str)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"]).dt.date.astype(str)

    # Odds: keep latest snapshot per matchup/date
    odds = odds.copy()
    odds["snapshot_utc"] = pd.to_datetime(odds["snapshot_utc"], errors="coerce")
    odds["game_date"] = pd.to_datetime(odds["game_date"]).dt.date.astype(str)

    odds = odds.sort_values("snapshot_utc").drop_duplicates(
        subset=["game_date", "home_abbr", "away_abbr"],
        keep="last"
    )

    # Join preds -> odds by exact date
    df = preds.merge(
        odds[["game_date", "home_abbr", "away_abbr", "ml_home", "ml_away", "p_home_mkt"]],
        left_on=["date", "home", "away"],
        right_on=["game_date", "home_abbr", "away_abbr"],
        how="left"
    )

    # Fallback: match odds by date+1 (UTC rollover)
    missing_odds = df["ml_home"].isna()
    if missing_odds.any():
        preds2 = preds.copy()
        preds2["date_plus1"] = (pd.to_datetime(preds2["date"]) + pd.Timedelta(days=1)).dt.date.astype(str)
        df_fb = preds2.merge(
            odds[["game_date", "home_abbr", "away_abbr", "ml_home", "ml_away", "p_home_mkt"]],
            left_on=["date_plus1", "home", "away"],
            right_on=["game_date", "home_abbr", "away_abbr"],
            how="left"
        )
        for c in ["ml_home", "ml_away", "p_home_mkt"]:
            df.loc[missing_odds, c] = df_fb.loc[missing_odds, c].values

    # Join to results
    df = df.merge(
        games[["GAME_DATE", "TEAM_ABBREVIATION_HOME", "TEAM_ABBREVIATION_AWAY", "HOME_WIN"]],
        left_on=["date", "home", "away"],
        right_on=["GAME_DATE", "TEAM_ABBREVIATION_HOME", "TEAM_ABBREVIATION_AWAY"],
        how="left"
    )

    # Diagnostics BEFORE filtering to completed games
    print("\nDIAGNOSTICS")
    print("--------------------------------------------------")
    print(f"Pred rows: {len(preds)}")
    print(f"Matched odds: {df['ml_home'].notna().mean():.1%} ({df['ml_home'].notna().sum()}/{len(df)})")
    print(f"Matched results (completed games): {df['HOME_WIN'].notna().mean():.1%} ({df['HOME_WIN'].notna().sum()}/{len(df)})")

    # Show top candidates even if results missing (so you can act today)
    tmp = df.dropna(subset=["ml_home", "ml_away", "p_final_home"]).copy()
    if not tmp.empty:
        tmp["dec_home"] = tmp["ml_home"].apply(american_to_decimal)
        tmp["dec_away"] = tmp["ml_away"].apply(american_to_decimal)
        tmp["p_home"] = tmp["p_final_home"].astype(float)
        tmp["p_away"] = 1.0 - tmp["p_home"]

        tmp["ev_home"] = tmp.apply(lambda r: ev_per_1u(r["p_home"], r["dec_home"]), axis=1)
        tmp["ev_away"] = tmp.apply(lambda r: ev_per_1u(r["p_away"], r["dec_away"]), axis=1)

        # Pick best side by EV
        tmp["side"] = np.where(tmp["ev_home"] >= tmp["ev_away"], "HOME", "AWAY")
        tmp["ev_best"] = np.where(tmp["side"] == "HOME", tmp["ev_home"], tmp["ev_away"])

        print("\nTOP TODAY (by EV, even if not finished yet)")
        print("--------------------------------------------------")
        show = tmp.sort_values("ev_best", ascending=False).head(PRINT_TOP)
        for _, r in show.iterrows():
            matchup = f"{r['away']} @ {r['home']}"
            side = r["side"]
            evb = r["ev_best"]
            p = r["p_home"] if side == "HOME" else (1.0 - r["p_home"])
            ml = r["ml_home"] if side == "HOME" else r["ml_away"]
            print(f"{matchup} | bet {side} | p={p:.3f} | ml={ml:.0f} | EV={evb:+.3f}")
    else:
        print("\nNo rows with odds + p_final yet to score EV.")

    # Now filter to completed games for true backtest
    bt = df.dropna(subset=["HOME_WIN", "ml_home", "ml_away", "p_final_home"]).copy()
    if bt.empty:
        print("\nBACKTEST")
        print("--------------------------------------------------")
        print("No completed games with (odds + prediction + result) yet.")
        print("Run this again after games finish AND after you update your games.parquet ingestion.")
        raise SystemExit

    bt["HOME_WIN"] = bt["HOME_WIN"].astype(int)
    bt["dec_home"] = bt["ml_home"].apply(american_to_decimal)
    bt["dec_away"] = bt["ml_away"].apply(american_to_decimal)

    bt["p_home"] = bt["p_final_home"].astype(float)
    bt["p_away"] = 1.0 - bt["p_home"]

    bt["ev_home"] = bt.apply(lambda r: ev_per_1u(r["p_home"], r["dec_home"]), axis=1)
    bt["ev_away"] = bt.apply(lambda r: ev_per_1u(r["p_away"], r["dec_away"]), axis=1)

    # Choose best side per game, bet only if EV >= MIN_EV
    bt["side"] = np.where(bt["ev_home"] >= bt["ev_away"], "HOME", "AWAY")
    bt["ev_best"] = np.where(bt["side"] == "HOME", bt["ev_home"], bt["ev_away"])
    bt = bt[bt["ev_best"] >= MIN_EV].copy()

    print("\nBACKTEST")
    print("--------------------------------------------------")
    if bt.empty:
        print(f"No bets triggered at MIN_EV={MIN_EV:.2f}. Lower MIN_EV or gather more days.")
        raise SystemExit

    # Realized pnl for chosen side
    bt["win_side"] = np.where(bt["side"] == "HOME", bt["HOME_WIN"], 1 - bt["HOME_WIN"])
    bt["dec_side"] = np.where(bt["side"] == "HOME", bt["dec_home"], bt["dec_away"])
    bt["pnl_1u"] = bt.apply(lambda r: pnl_per_1u(int(r["win_side"]), float(r["dec_side"])), axis=1)

    n = len(bt)
    total_units = float(bt["pnl_1u"].sum())
    roi_per_bet = total_units / n
    win_rate = float(bt["win_side"].mean())
    avg_ev = float(bt["ev_best"].mean())

    print(f"Bets: {n}")
    print(f"Win rate: {win_rate:.3f}")
    print(f"Avg EV (per 1u): {avg_ev:+.3f}")
    print(f"Total units: {total_units:+.2f}")
    print(f"ROI per bet: {roi_per_bet:+.3f}")

    out_path = "data/processed/backtest_results.csv"
    bt.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
