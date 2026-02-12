import os
from datetime import datetime, timezone
import pandas as pd
import requests

from nba_api.stats.static import teams as nba_teams


from dotenv import load_dotenv

load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

if not ODDS_API_KEY:
    raise ValueError("ODDS_API_KEY not found in .env")

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"

def moneyline_to_prob(ml: float) -> float:
    ml = float(ml)
    if ml < 0:
        return (-ml) / ((-ml) + 100.0)
    return 100.0 / (ml + 100.0)

def team_name_to_abbr():
    # odds-api uses team full names like "Los Angeles Lakers"
    mapping = {}
    for t in nba_teams.get_teams():
        mapping[t["full_name"]] = t["abbreviation"]
    return mapping

def pick_consensus_h2h(bookmakers):
    """
    Return (home_ml, away_ml) using a simple consensus:
      - use the first bookmaker that has both sides
    You can upgrade this to median across books later.
    """
    for bk in bookmakers:
        for mkt in bk.get("markets", []):
            if mkt.get("key") != "h2h":
                continue
            outs = mkt.get("outcomes", [])
            if len(outs) < 2:
                continue
            # outcomes have {name, price}
            return outs
    return None

if __name__ == "__main__":
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    url = f"{BASE_URL}/sports/{SPORT_KEY}/odds/"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    name2abbr = team_name_to_abbr()

    rows = []
    snapshot_ts = datetime.now(timezone.utc).isoformat()

    for ev in data:
        home_name = ev.get("home_team")
        away_name = ev.get("away_team")
        commence = ev.get("commence_time")  # ISO string

        if home_name not in name2abbr or away_name not in name2abbr:
            continue

        outs = pick_consensus_h2h(ev.get("bookmakers", []))
        if not outs:
            continue

        # outcomes include both teams; order may vary
        prices = {o["name"]: o["price"] for o in outs if "name" in o and "price" in o}
        if home_name not in prices or away_name not in prices:
            continue

        ml_home = float(prices[home_name])
        ml_away = float(prices[away_name])

        p_home_raw = moneyline_to_prob(ml_home)
        p_away_raw = moneyline_to_prob(ml_away)
        denom = p_home_raw + p_away_raw
        if denom <= 0:
            continue
        p_home_novig = p_home_raw / denom

        rows.append({
            "snapshot_utc": snapshot_ts,
            "event_id": ev.get("id"),
            "commence_time_utc": commence,
            "game_date": commence[:10],  # YYYY-MM-DD
            "home_team": home_name,
            "away_team": away_name,
            "home_abbr": name2abbr[home_name],
            "away_abbr": name2abbr[away_name],
            "ml_home": ml_home,
            "ml_away": ml_away,
            "p_home_mkt": p_home_novig,
        })

    df_new = pd.DataFrame(rows)
    if df_new.empty:
        print("No odds returned (maybe offseason or no listed games).")
        raise SystemExit

    out_path = "data/raw/odds_snapshots.parquet"
    if os.path.exists(out_path):
        df_old = pd.read_parquet(out_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    # Dedupe by snapshot + event_id
    df = df.drop_duplicates(subset=["snapshot_utc", "event_id"], keep="last")
    df.to_parquet(out_path, index=False)

    print(f"Saved {len(df_new)} events -> {out_path}")
    print(df_new[["game_date","away_abbr","home_abbr","ml_away","ml_home","p_home_mkt"]].head())
