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
    df = pd.read_parquet("data/processed/games_features.parquet")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    train, test = time_split(df, "2024-10-01")

    feature_cols = ["ELO_DIFF_PRE", "REST_DIFF", "B2B_DIFF", "NET10_DIFF", "NET5_DIFF"]
    X_train = train[feature_cols].to_numpy()
    y_train = train["HOME_WIN"].to_numpy()
    X_test  = test[feature_cols].to_numpy()
    y_test  = test["HOME_WIN"].to_numpy()

    base = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06)
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

    dump({"model": model, "feature_cols": feature_cols}, "models/win_model_v2.joblib")
    print("Saved models/win_model_v2.joblib")
