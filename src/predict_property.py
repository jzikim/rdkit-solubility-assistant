import argparse
import joblib
import numpy as np
from src import features


def predict(smiles: str, model_path: str = 'models/property_model.pkl'):
    data = joblib.load(model_path)
    pipeline = data.get('pipeline')
    model_target = data.get('target', 'target_property')
    train_fps = data.get('train_fps', None)

    feat, desc = features.featurize_smiles(smiles)
    if feat is None:
        print('Invalid SMILES provided')
        return
    feat_aligned = features.align_features_for_pipeline(pipeline, feat)
    X = feat_aligned.reshape(1, -1)
    pred = pipeline.predict(X)[0]

    print('RDKit descriptors:')
    for k, v in desc.items():
        print(f'  {k}: {v}')

    print(f'Predicted target ({model_target}) (model): {pred:.4f}')

    # estimate uncertainty from RandomForest (std across trees) if available
    try:
        estimator = pipeline.named_steps.get('model') if hasattr(pipeline, 'named_steps') else None
        if estimator is not None and hasattr(estimator, 'estimators_'):
            tree_preds = np.array([e.predict(X) for e in estimator.estimators_])
            std = float(tree_preds.std())
            print(f'Prediction std (approx.): {std:.4f}')
        else:
            std = None
    except Exception:
        std = None

    # applicability domain via Tanimoto similarity if training fingerprints available
    try:
        fp = features.morgan_fingerprint_array(smiles)
        if fp is not None and train_fps is not None:
            max_sim = features.compute_max_tanimoto(fp, train_fps)
            print(f'Max Tanimoto similarity to training set: {max_sim:.3f}')
        else:
            max_sim = None
    except Exception:
        max_sim = None

    # user-friendly confidence label
    conf = 'Unknown'
    if max_sim is not None and std is not None:
        if max_sim >= 0.7 and std is not None and std < 0.2:
            conf = 'High'
        elif max_sim >= 0.4 and (std is None or std < 0.5):
            conf = 'Medium'
        else:
            conf = 'Low'
    elif max_sim is not None:
        if max_sim >= 0.7:
            conf = 'High'
        elif max_sim >= 0.4:
            conf = 'Medium'
        else:
            conf = 'Low'

    print(f'Confidence: {conf}')
    # If RDKit-calculated value exists for the target (e.g. LogP), show difference
    # We don't know the name of the target here, so we simply display RDKit LogP if available
    # If the model's target is one of RDKit-calculated descriptors, show the RDKit value and diff
    if model_target in desc:
        try:
            rd_val = desc[model_target]
            print(f'RDKit {model_target}: {rd_val:.4f} — model vs RDKit diff: {pred - rd_val:.4f}')
        except Exception:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict property for a SMILES string')
    parser.add_argument('--smiles', type=str, required=True)
    parser.add_argument('--model', type=str, default='models/property_model.pkl')
    args = parser.parse_args()
    predict(args.smiles, args.model)
