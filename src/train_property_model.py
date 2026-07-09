import os
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from src import features


def main(csv_path: str, target_col: str, test_size: float = 0.2, random_state: int = 42, output_model: str = 'models/property_model.pkl'):
    df = pd.read_csv(csv_path)
    if 'smiles' not in df.columns:
        raise ValueError('CSV must contain a "smiles" column')
    if target_col not in df.columns or df[target_col].isna().all():
        print(f'Warning: target column "{target_col}" missing or empty. Falling back to RDKit-calculated LogP as target for demonstration.')
        # compute RDKit LogP on the fly
        df[target_col] = df['smiles'].apply(lambda s: features.compute_rdkit_descriptors(s)['LogP'] if features.compute_rdkit_descriptors(s) is not None else np.nan)

    # drop rows without valid target
    df = df.dropna(subset=[target_col])
    X, descs, valid_idx = featurize_series(df['smiles'])
    if X is None:
        raise ValueError('No valid SMILES found to featurize')
    y = df.loc[valid_idx, target_col].astype(float).values

    # compute training fingerprints for applicability domain
    train_fps = []
    for idx in valid_idx:
        s = df.loc[idx, 'smiles']
        fp = features.morgan_fingerprint_array(s)
        if fp is None:
            fp = np.zeros((2048,), dtype=np.uint8)
        train_fps.append(fp)
    train_fps = np.vstack(train_fps) if train_fps else np.zeros((0, 2048), dtype=np.uint8)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1))
    ])

    print('Training model...')
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f'Metrics on test set - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}')

    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    joblib.dump({'pipeline': pipeline, 'target': target_col, 'train_fps': train_fps}, output_model)
    print(f'Model saved to {output_model} (target: {target_col})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a property regression model from SMILES')
    parser.add_argument('--csv', type=str, default='data/molecule_dataset.csv')
    parser.add_argument('--target', type=str, default='target_property')
    parser.add_argument('--out', type=str, default='models/property_model.pkl')
    args = parser.parse_args()
    main(args.csv, args.target, output_model=args.out)
