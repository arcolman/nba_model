import pandas as pd

def elo_expected(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

def add_elo_features(games: pd.DataFrame, k=20.0, home_adv=65.0, base=1500.0) -> pd.DataFrame:
    ratings = {}
    rows = []

    games = games.sort_values("GAME_DATE")

    for _, g in games.iterrows():
        h = int(g["TEAM_ID_HOME"])
        a = int(g["TEAM_ID_AWAY"])
        y = int(g["HOME_WIN"])

        r_h = ratings.get(h, base)
        r_a = ratings.get(a, base)

        exp_h = elo_expected(r_h + home_adv, r_a)

        # store pregame elos
        rows.append({
            "GAME_ID": g["GAME_ID"],
            "GAME_DATE": g["GAME_DATE"],
            "TEAM_ID_HOME": h,
            "TEAM_ID_AWAY": a,
            "HOME_WIN": y,
            "ELO_HOME_PRE": r_h,
            "ELO_AWAY_PRE": r_a,
            "ELO_DIFF_PRE": (r_h + home_adv) - r_a,
        })

        # update
        r_h_new = r_h + k * (y - exp_h)
        r_a_new = r_a + k * ((1 - y) - (1 - exp_h))

        ratings[h] = r_h_new
        ratings[a] = r_a_new

    return pd.DataFrame(rows)

if __name__ == "__main__":
    games = pd.read_parquet("data/processed/games.parquet")
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    elo = add_elo_features(games, k=20.0, home_adv=65.0)
    elo.to_parquet("data/processed/games_with_elo.parquet", index=False)

    print("Saved data/processed/games_with_elo.parquet")
    print(elo.head())
