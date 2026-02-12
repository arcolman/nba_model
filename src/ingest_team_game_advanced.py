import os
import time
import random
import pandas as pd
import requests
import requests_cache

from nba_api.stats.endpoints import boxscoreadvancedv2, boxscorefourfactorsv2
from nba_api.stats.library.http import NBAStatsHTTP

# Cache so reruns don't re-hit the NBA API for the same requests
requests_cache.install_cache("nba_cache", expire_after=60 * 60 * 24)

# Browser-like headers (stats.nba.com often blocks "non-browser" traffic)
NBAStatsHTTP.headers = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-token": "true",
    "x-nba-stats-origin": "stats",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept-Language": "en-US,en;q=0.9",
}

OUT_PATH = "data/raw/team_game_advanced.parquet"

def safe_frames(make_call, label="", max_retries=5, base_sleep=1.5):
    """
    Returns list[pd.DataFrame] from nba_api .get_data_frames() or None.
    Retries on timeout / connection errors with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            return make_call()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            wait = base_sleep * (2 ** attempt) + random.uniform(0, 0.6)
            print(f"{label} timeout/conn (try {attempt+1}/{max_retries}) -> sleep {wait:.1f}s", flush=True)
            time.sleep(wait)
        except Exception as e:
            # Often caused by blocked responses / unexpected payload shape
            print(f"{label} error: {type(e).__name__}: {e}", flush=True)
            return None
    print(f"{label} failed after retries", flush=True)
    return None

def fetch_one_game(game_id: str, timeout=90):
    """
    Returns 2-row dataframe (one per team) with advanced stats + four factors for this game_id, or None.
    """
    adv_frames = safe_frames(
        lambda: boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id, timeout=timeout).get_data_frames(),
        label=f"[ADV {game_id}]"
    )
    if adv_frames is None:
        return None

    ff_frames = safe_frames(
        lambda: boxscorefourfactorsv2.BoxScoreFourFactorsV2(game_id=game_id, timeout=timeout).get_data_frames(),
        label=f"[FF  {game_id}]"
    )
    if ff_frames is None:
        return None

    # Team tables are usually index 1
    adv_team = adv_frames[1].copy()
    ff_team = ff_frames[1].copy()

    adv_keep = ["GAME_ID", "TEAM_ID", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]
    ff_keep = ["GAME_ID", "TEAM_ID", "EFG_PCT", "TM_TOV_PCT", "OREB_PCT", "FT_RATE"]

    if not set(adv_keep).issubset(adv_team.columns):
        return None
    if not set(ff_keep).issubset(ff_team.columns):
        return None

    out = adv_team[adv_keep].merge(ff_team[ff_keep], on=["GAME_ID", "TEAM_ID"], how="inner")
    out = out.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="last")
    return out

def load_existing_done():
    if not os.path.exists(OUT_PATH):
        return set(), None
    df = pd.read_parquet(OUT_PATH)
    done = set(df["GAME_ID"].astype(str).unique())
    return done, df

def checkpoint_save(rows):
    """
    Append rows to OUT_PATH (create if needed), dedupe, and save.
    Returns combined df.
    """
    new = pd.concat(rows, ignore_index=True)
    if os.path.exists(OUT_PATH):
        old = pd.read_parquet(OUT_PATH)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new

    combined = combined.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="last")
    combined.to_parquet(OUT_PATH, index=False)
    return combined

if __name__ == "__main__":
    # Load game ids from your games table
    games = pd.read_parquet("data/processed/games.parquet")
    games["GAME_ID"] = games["GAME_ID"].astype(str)

    # Keep regular season + playoffs only (skip preseason 001...)
    games = games[games["GAME_ID"].str.startswith(("002", "004"))].copy()

    game_ids = sorted(games["GAME_ID"].unique())
    total = len(game_ids)
    print(f"Will attempt {total} games (regular season + playoffs only)", flush=True)

    done, existing_df = load_existing_done()
    if done:
        print(f"Resuming: already have {len(done)} game_ids saved in {OUT_PATH}", flush=True)

    rows = []
    ok_games = 0
    skipped_games = 0
    attempted = 0

    for idx, gid in enumerate(game_ids, start=1):
        if gid in done:
            continue

        attempted += 1
        print(f"attempt {idx}/{total} {gid}", flush=True)

        # Slow down to reduce throttling
        time.sleep(2.2 + random.uniform(0.0, 0.8))

        df = fetch_one_game(gid, timeout=90)
        if df is None or df.empty:
            skipped_games += 1
        else:
            rows.append(df)
            ok_games += 1

        # Checkpoint every 50 *attempts* (not successes) so you get a file quickly
        if attempted % 50 == 0 and rows:
            combined = checkpoint_save(rows)
            rows = []
            done = set(combined["GAME_ID"].astype(str).unique())
            print(
                f"CHECKPOINT: rows_saved={len(combined)} | unique_games={combined['GAME_ID'].nunique()} "
                f"| ok_games={ok_games} | skips={skipped_games}",
                flush=True
            )

        # Light progress message even if no rows yet
        if attempted % 50 == 0 and not rows:
            print(f"PROGRESS: attempted={attempted} | ok_games={ok_games} | skips={skipped_games} | file_exists={os.path.exists(OUT_PATH)}", flush=True)

    # Final save
    if rows:
        combined = checkpoint_save(rows)
        print(f"FINAL SAVE: rows_saved={len(combined)} | unique_games={combined['GAME_ID'].nunique()}", flush=True)

    if os.path.exists(OUT_PATH):
        final = pd.read_parquet(OUT_PATH)
        print(f"DONE: {len(final)} rows | {final['GAME_ID'].nunique()} unique games -> {OUT_PATH}", flush=True)
        print(final.head(), flush=True)
    else:
        print("DONE: no parquet created (NBA endpoint likely blocking). Try again later.", flush=True)
