import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_features import ML_LOGS_FEATURE_NAMES, compute_rdkit_descriptors, make_feature_vector


def train_and_save(csv_path: str = "data/logs_dataset.csv", output_model: str = "models/logs_model.pkl") -> None:
    df = pd.read_csv(csv_path)
    if not {"smiles", "logS"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: smiles, logS")

    records = []
    for _, row in df.iterrows():
        try:
            desc = compute_rdkit_descriptors(row["smiles"])
            records.append((make_feature_vector(desc), row["logS"]))
        except Exception:
            continue

    if not records:
        raise RuntimeError("No valid SMILES found in dataset.")

    X = [r[0] for r in records]
    y = [r[1] for r in records]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            ),
        ]
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    print("Model trained on:", len(X), "samples")
    print("MAE:", mean_absolute_error(y, preds))
    print("RMSE:", mean_squared_error(y, preds) ** 0.5)
    print("R2:", r2_score(y, preds))

    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": ML_LOGS_FEATURE_NAMES,
            "target": "logS",
        },
        output_model,
    )
    print("Saved model to", output_model)


if __name__ == "__main__":
    train_and_save()
