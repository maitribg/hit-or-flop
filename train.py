import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import mlflow
import os

def preprocess(df):
    df["is_hit"] = ((df["averageRating"] >= 7.0) & (df["numVotes"] >= 1000)).astype(int)
    df["genres"] = df["genres"].fillna("").str.split(",")

    all_genres = set(g for genre_list in df["genres"] for g in genre_list)
    for genre in all_genres:
        df[f"genre_{genre.strip()}"] = df["genres"].apply(lambda x: int(genre.strip() in x))

    features = ["runtimeMinutes", "startYear", "numVotes"] + [f"genre_{g.strip()}" for g in all_genres]
    df = df.dropna(subset=features)
    return df[features], df["is_hit"]

def train(data_path):
    df = pd.read_csv(data_path)
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metrics({
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1_score": report["1"]["f1-score"],
            "accuracy": report["accuracy"]
        })
        mlflow.sklearn.log_model(clf, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    args = parser.parse_args()
    train(args.data)
