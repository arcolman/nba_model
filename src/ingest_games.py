import pandas as pd
import requests_cache
from nba_api.stats.endpoints import leaguegamefinder

def fetch_season_team_games(season: str) -> pd.DataFrame:
    # Cache requests so reruns are fast and you hit fewer rate limits
    requests_cache.install_cache("nba_cache", expire_after=60 * 60 * 24)

    games = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",  # NBA
    ).get_data_frames()[0]

    # Each row is a TEAM's view of a game (so every GAME_ID appears twice)
    keep = [
        "SEASON_ID", "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION",
        "MATCHUP", "WL", "PTS", "PLUS_MINUS"
    ]
    games = games[keep].copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    return games

def make_game_level_table(team_games: pd.DataFrame) -> pd.DataFrame:
    # Split into home/away by MATCHUP string: "LAL vs. BOS" or "LAL @ BOS"
    tg = team_games.copy()
    tg["IS_HOME"] = tg["MATCHUP"].str.contains("vs.")
    # For each GAME_ID, create one row with home + away teams
    home = tg[tg["IS_HOME"]].copy()
    away = tg[~tg["IS_HOME"]].copy()

    df = home.merge(
        away,
        on="GAME_ID",
        suffixes=("_HOME", "_AWAY"),
        how="inner"
    )

    # Target: home win
    df["HOME_WIN"] = (df["WL_HOME"] == "W").astype(int)

    keep_cols = [
        "GAME_ID", "GAME_DATE_HOME",
        "TEAM_ID_HOME", "TEAM_ABBREVIATION_HOME", "PTS_HOME",
        "TEAM_ID_AWAY", "TEAM_ABBREVIATION_AWAY", "PTS_AWAY",
        "HOME_WIN"
    ]
    df = df[keep_cols].rename(columns={"GAME_DATE_HOME": "GAME_DATE"})
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df

if __name__ == "__main__":
    # Start with 2 seasons for speed; add more later
    seasons = ["2023-24", "2024-25"]
    all_team_games = []

    for s in seasons:
        print(f"Fetching {s}...")
        all_team_games.append(fetch_season_team_games(s))

    team_games = pd.concat(all_team_games, ignore_index=True)
    team_games.to_parquet("data/raw/team_games.parquet", index=False)

    game_level = make_game_level_table(team_games)
    game_level.to_parquet("data/processed/games.parquet", index=False)

    print("Saved:")
    print("- data/raw/team_games.parquet")
    print("- data/processed/games.parquet")
    print(game_level.head())
