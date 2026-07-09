import argparse
import joblib
import numpy as np
from src import features


def interpret_logS(logS: float) -> str:
    # Simple educational interpretation thresholds
    if logS >= -1.0:
        return 'Very soluble'
    if logS >= -3.0:
        return 'Moderately soluble'
    return 'Poorly soluble'


def predict(smiles: str, model_path: str = 'models/logs_model.pkl'):
    data = joblib.load(model_path)
    pipeline = data.get('pipeline')
    model_target = data.get('target', 'logS')
    train_fps = data.get('train_fps', None)

    feat, desc = features.featurize_smiles(smiles)
    if feat is None:
        print('Invalid SMILES provided')
        return
    # align features to the pipeline's expected input size (pad/truncate if needed)
    feat_aligned = features.align_features_for_pipeline(pipeline, feat)
    X = feat_aligned.reshape(1, -1)
    pred = pipeline.predict(X)[0]

    print('Molecule SMILES:', smiles)
    print('Predicted logS:', f'{pred:.4f}')
    print('Solubility interpretation:', interpret_logS(pred))

    # show key descriptors
    rd_desc = features.compute_rdkit_descriptors(smiles)
    if rd_desc:
        keys = ['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'RingCount', 'HeavyAtomCount']
        print('Key descriptors:')
        for k in keys:
            print(f'  {k}: {rd_desc.get(k)}')

    # uncertainty and applicability
    std = None
    try:
        estimator = pipeline.named_steps.get('model') if hasattr(pipeline, 'named_steps') else None
        if estimator is not None and hasattr(estimator, 'estimators_'):
            tree_preds = np.array([e.predict(X) for e in estimator.estimators_])
            std = float(tree_preds.std())
            print(f'Prediction std (approx.): {std:.4f}')
    except Exception:
        pass

    try:
        fp = features.morgan_fingerprint_array(smiles)
        if fp is not None and train_fps is not None:
            max_sim = features.compute_max_tanimoto(fp, train_fps)
            print(f'Max Tanimoto similarity to training set: {max_sim:.3f}')
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict logS for a SMILES string')
    parser.add_argument('--smiles', type=str, required=True)
    parser.add_argument('--model', type=str, default='models/logs_model.pkl')
    args = parser.parse_args()
    predict(args.smiles, args.model)
