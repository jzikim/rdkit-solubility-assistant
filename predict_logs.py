from typing import Optional

import joblib

from ml_features import compute_rdkit_descriptors, make_feature_vector
from pubchem_utils import get_canonical_smiles


def interpret_logS(logS: float) -> str:
    if logS > -1:
        return "relatively soluble"
    if logS >= -3:
        return "moderately soluble"
    return "poorly soluble"


def predict_logS_from_smiles(smiles: str, model_path: str = "models/logs_model.pkl") -> dict:
    descriptors = compute_rdkit_descriptors(smiles)
    feature_vector = make_feature_vector(descriptors)

    model_data = joblib.load(model_path)
    pipeline = model_data["pipeline"]
    pred = pipeline.predict([feature_vector])[0]

    return {
        "smiles": smiles,
        "descriptors": descriptors,
        "predicted_logS": float(pred),
        "interpretation": interpret_logS(pred),
    }


def predict_logS(drug_name_or_smiles: str, model_path: str = "models/logs_model.pkl") -> dict:
    smiles = get_canonical_smiles(drug_name_or_smiles)
    if smiles is None:
        smiles = drug_name_or_smiles

    return predict_logS_from_smiles(smiles, model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict logS from a drug name or SMILES")
    parser.add_argument("input", help="Drug name or SMILES string")
    parser.add_argument("--model", default="models/logs_model.pkl", help="Model file path")
    args = parser.parse_args()

    try:
        result = predict_logS(args.input, args.model)
        print(result)
    except Exception as error:
        print("Error:", error)
