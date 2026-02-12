import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from joblib import dump

def time_split(df, split_date):
    split_date = pd.to_datetime(split_date)
    train = df[df["GAME_DATE"] < split_date].copy()
    test  = df[df["GAME_DATE"] >= split_date].copy()
    return train, test

if __name__ == "__main__":
    base = pd.read_parquet("data/processed/games_features.parquet")      # has REST_DIFF/B2B_DIFF (clean)
    adv  = pd.read_parquet("data/processed/games_features_v3.parquet")   # has advanced diffs

    base["GAME_DATE"] = pd.to_datetime(base["GAME_DATE"])
    adv["GAME_DATE"] = pd.to_datetime(adv["GAME_DATE"])

    # Join on GAME_ID (both are 1 row per game)
    df = adv.merge(base[["GAME_ID","REST_DIFF","B2B_DIFF"]], on="GAME_ID", how="left")

    # Feature set
    feature_cols = [
        "ELO_DIFF_PRE",
        "REST_DIFF","B2B_DIFF",
        "NET_RATING10_DIFF","OFF_RATING10_DIFF","DEF_RATING10_DIFF","PACE10_DIFF",
        "EFG_PCT10_DIFF","TM_TOV_PCT10_DIFF","OREB_PCT10_DIFF","FT_RATE10_DIFF",
        # optional: include L5 diffs too
        "NET_RATING5_DIFF","EFG_PCT5_DIFF","TM_TOV_PCT5_DIFF"
    ]

    df = df.dropna(subset=feature_cols + ["HOME_WIN"]).copy()

    train, test = time_split(df, "2024-10-01")

    X_train = train[feature_cols].to_numpy()
    y_train = train["HOME_WIN"].to_numpy()
    X_test  = test[feature_cols].to_numpy()
    y_test  = test["HOME_WIN"].to_numpy()

    base_model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05)
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_test  = model.predict_proba(X_test)[:, 1]

    print("Train:")
    print("  logloss:", log_loss(y_train, p_train))
    print("  brier  :", brier_score_loss(y_train, p_train))
    print("  acc    :", accuracy_score(y_train, (p_train > 0.5).astype(int)))

    print("Test:")
    print("  logloss:", log_loss(y_test, p_test))
    print("  brier  :", brier_score_loss(y_test, p_test))
    print("  acc    :", accuracy_score(y_test, (p_test > 0.5).astype(int)))

    dump({"model": model, "feature_cols": feature_cols}, "models/win_model_v3.joblib")
    print("Saved models/win_model_v3.joblib")
