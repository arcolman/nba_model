import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from joblib import dump

def time_split(df: pd.DataFrame, split_date: str):
    split_date = pd.to_datetime(split_date)
    train = df[df["GAME_DATE"] < split_date].copy()
    test  = df[df["GAME_DATE"] >= split_date].copy()
    return train, test

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/games_with_elo.parquet")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Split between seasons you downloaded
    train, test = time_split(df, "2024-10-01")

    X_train = train[["ELO_DIFF_PRE"]].to_numpy()
    y_train = train["HOME_WIN"].to_numpy()
    X_test  = test[["ELO_DIFF_PRE"]].to_numpy()
    y_test  = test["HOME_WIN"].to_numpy()

    base = LogisticRegression(max_iter=2000)
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
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

    dump(model, "models/win_model.joblib")
    print("Saved models/win_model.joblib")
