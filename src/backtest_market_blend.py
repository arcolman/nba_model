import pandas as pd
import numpy as np

# ---------- odds conversions ----------
def american_to_decimal(ml: float) -> float:
    ml = float(ml)
    if ml > 0:
        return 1.0 + (ml / 100.0)
    else:
        return 1.0 + (100.0 / (-ml))

def profit_per_unit_stake_from_decimal(dec: float) -> float:
    # stake 1 unit; win profit = dec-1; lose = -1
    return dec - 1.0

def kelly_fraction(p: float, dec: float) -> float:
    # f* = (bp - q)/b where b=dec-1, q=1-p
    b = dec - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f)

# ---------- main ----------
if __name__ == "__main__":
    # ---- Inputs ----
    PRED_PATH = "data/processed/preds_with_market.csv"
    ODDS_PATH = "data/raw/odds_snapshots.parquet"
    GAMES_PATH = "data/processed/games.parquet"

    EDGE_THRESHOLD = 0.04          # only bet if p_final - p_market >= this
    MIN_KELLY = 0.0                # floor
    MAX_KELLY = 0.05               # cap per bet (5% bankroll)
    KELLY_MULT = 0.25              # quarter-kelly (safer)
    START_BANKROLL = 100.0

    # ---- Load ----
    preds = pd.read_csv(PRED_PATH)
    odds = pd.read_parquet(ODDS_PATH)
    games = pd.read_parquet(GAMES_PATH)

    # Normalize dates
    preds["date"] = pd.to_datetime(preds["date"]).dt.date.astype(str)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"]).dt.date.astype(str)

    # Ensure abbreviations exist in games table
    # Your games.parquet columns (from earlier): TEAM_ABBREVIATION_HOME/AWAY, HOME_WIN
    needed_cols = {"GAME_DATE", "TEAM_ABBREVIATION_HOME", "TEAM_ABBREVIATION_AWAY", "HOME_WIN"}
    missing = needed_cols - set(games.columns)
    if missing:
        raise ValueError(f"games.parquet missing columns: {missing}")

    # Keep latest snapshot per (game_date, home_abbr, away_abbr) (acts like “closest to now”)
    odds = odds.copy()
    odds["snapshot_utc"] = pd.to_datetime(odds["snapshot_utc"], errors="coerce")
    odds["game_date"] = pd.to_datetime(odds["game_date"]).dt.date.astype(str)

    odds = odds.sort_values("snapshot_utc").drop_duplicates(
        subset=["game_date", "home_abbr", "away_abbr"],
        keep="last"
    )

    # Join preds -> odds (to get moneylines)
    df = preds.merge(
        odds[["game_date", "home_abbr", "away_abbr", "ml_home", "ml_away", "p_home_mkt"]],
        left_on=["date", "home", "away"],
        right_on=["game_date", "home_abbr", "away_abbr"],
        how="left"
    )

    # If UTC date mismatch happened in your logs, try tomorrow match fallback:
    # (This catches the common “local date vs UTC date” issue.)
    missing_odds = df["ml_home"].isna()
    if missing_odds.any():
        df2 = preds.copy()
        df2["date_plus1"] = (pd.to_datetime(df2["date"]) + pd.Timedelta(days=1)).dt.date.astype(str)
        df2 = df2.merge(
            odds[["game_date", "home_abbr", "away_abbr", "ml_home", "ml_away", "p_home_mkt"]],
            left_on=["date_plus1", "home", "away"],
            right_on=["game_date", "home_abbr", "away_abbr"],
            how="left"
        )
        # fill missing from fallback
        for c in ["ml_home", "ml_away", "p_home_mkt", "game_date"]:
            df.loc[missing_odds, c] = df2.loc[missing_odds, c].values

    # Join to results
    df = df.merge(
        games[["GAME_DATE", "TEAM_ABBREVIATION_HOME", "TEAM_ABBREVIATION_AWAY", "HOME_WIN"]],
        left_on=["date", "home", "away"],
        right_on=["GAME_DATE", "TEAM_ABBREVIATION_HOME", "TEAM_ABBREVIATION_AWAY"],
        how="left"
    )

    # Keep only rows with actual results + odds
    df = df.dropna(subset=["HOME_WIN", "ml_home", "ml_away", "p_final_home"]).copy()
    df["HOME_WIN"] = df["HOME_WIN"].astype(int)

    # Decide bet side based on edge
    df["p_final_home"] = df["p_final_home"].astype(float)
    df["p_market_home"] = df["p_market_home"].astype(float)

    df["edge"] = df["p_final_home"] - df["p_market_home"]
    df["bet_home"] = (df["edge"] >= EDGE_THRESHOLD).astype(int)

    # Also allow betting away if strongly negative (optional)
    # If you want BOTH sides, uncomment:
    # df["bet_away"] = (df["edge"] <= -EDGE_THRESHOLD).astype(int)
    # For now: only home-side bets to keep it simple
    df = df[df["bet_home"] == 1].copy()

    if df.empty:
        print("No bets triggered. Lower EDGE_THRESHOLD or gather more days.")
        raise SystemExit

    # Compute payout / EV / realized
    df["dec_home"] = df["ml_home"].apply(american_to_decimal)
    df["win_profit_per_unit"] = df["dec_home"].apply(profit_per_unit_stake_from_decimal)
    df["p"] = df["p_final_home"]

    # Realized PnL for 1u stake on home
    df["pnl_1u"] = np.where(df["HOME_WIN"] == 1, df["win_profit_per_unit"], -1.0)

    # Expected value (per 1u stake)
    df["ev_1u"] = df["p"] * df["win_profit_per_unit"] - (1.0 - df["p"])

    # Kelly sizing (fraction of bankroll)
    df["kelly_f"] = df.apply(lambda r: kelly_fraction(r["p"], r["dec_home"]), axis=1)
    df["stake_kelly"] = (KELLY_MULT * df["kelly_f"]).clip(lower=MIN_KELLY, upper=MAX_KELLY)

    # Simulate bankroll over time (sort chronologically)
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date_dt", "game_id"]).reset_index(drop=True)

    bankroll_flat = START_BANKROLL
    bankroll_kelly = START_BANKROLL
    eq_flat = []
    eq_kelly = []

    for _, r in df.iterrows():
        # flat: 1 unit stake
        bankroll_flat += r["pnl_1u"]
        eq_flat.append(bankroll_flat)

        # kelly: stake as fraction of bankroll
        stake = bankroll_kelly * float(r["stake_kelly"])
        pnl = stake * (r["win_profit_per_unit"] if r["HOME_WIN"] == 1 else -1.0)
        bankroll_kelly += pnl
        eq_kelly.append(bankroll_kelly)

    df["bankroll_flat"] = eq_flat
    df["bankroll_kelly"] = eq_kelly

    # Metrics
    n_bets = len(df)
    win_rate = df["HOME_WIN"].mean()
    roi_flat = (df["pnl_1u"].sum() / n_bets)  # per bet units
    total_units = df["pnl_1u"].sum()
    avg_ev = df["ev_1u"].mean()
    avg_edge = df["edge"].mean()

    # Max drawdown (flat)
    peak = np.maximum.accumulate(df["bankroll_flat"].values)
    dd = (df["bankroll_flat"].values - peak)
    max_dd_flat = dd.min()

    # Max drawdown (kelly)
    peakk = np.maximum.accumulate(df["bankroll_kelly"].values)
    ddk = (df["bankroll_kelly"].values - peakk)
    max_dd_kelly = ddk.min()

    print("\nBACKTEST (home bets only)")
    print("--------------------------------------------------")
    print(f"Bets: {n_bets}")
    print(f"Win rate: {win_rate:.3f}")
    print(f"Avg edge (p_final - p_mkt): {avg_edge:.3f}")
    print(f"Avg EV (per 1u): {avg_ev:.3f}")
    print("---- Flat (1u) ----")
    print(f"Total units: {total_units:.2f}")
    print(f"ROI per bet: {roi_flat:.3f}")
    print(f"Max drawdown (units): {max_dd_flat:.2f}")
    print("---- Kelly (fractional) ----")
    print(f"Start bankroll: {START_BANKROLL:.2f}")
    print(f"End bankroll:   {df['bankroll_kelly'].iloc[-1]:.2f}")
    print(f"Max drawdown ($): {max_dd_kelly:.2f}")

    out_path = "data/processed/backtest_results.csv"
    df.to_csv(out_path, index=False)
    print("--------------------------------------------------")
    print(f"Saved detailed results: {out_path}")
