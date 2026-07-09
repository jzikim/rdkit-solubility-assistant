import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen, Lipinski
from rdkit import DataStructs

def mol_from_smiles(smiles: str):
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None

def compute_rdkit_descriptors(smiles: str):
    """Return a dict of a few common RDKit-calculated descriptors.

    Returns keys: MolWt, LogP, TPSA, NumHDonors, NumHAcceptors
    If SMILES invalid, returns None.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    desc = {}
    desc['MolWt'] = rdMolDescriptors.CalcExactMolWt(mol)
    desc['LogP'] = Crippen.MolLogP(mol)
    desc['TPSA'] = rdMolDescriptors.CalcTPSA(mol)
    desc['NumHDonors'] = rdMolDescriptors.CalcNumHBD(mol)
    desc['NumHAcceptors'] = rdMolDescriptors.CalcNumHBA(mol)
    # additional descriptors
    try:
        desc['NumRotatableBonds'] = Lipinski.NumRotatableBonds(mol)
    except Exception:
        desc['NumRotatableBonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol) if hasattr(rdMolDescriptors, 'CalcNumRotatableBonds') else 0
    desc['RingCount'] = mol.GetRingInfo().NumRings()
    desc['HeavyAtomCount'] = mol.GetNumHeavyAtoms()
    return desc

def morgan_fingerprint_array(smiles: str, radius: int = 2, nBits: int = 2048):
    """Return a numpy array (0/1) of the Morgan fingerprint for a SMILES.
    Returns None on invalid SMILES.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    arr = np.zeros((nBits,), dtype=np.uint8)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    # Use DataStructs.ConvertToNumpyArray for compatibility across RDKit versions
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_max_tanimoto(fp: np.ndarray, train_fps: np.ndarray):
    """Compute maximum Tanimoto similarity between a fingerprint and an array of train fingerprints.

    fp: 1D numpy array of bits (0/1)
    train_fps: 2D numpy array shape (n_train, nBits)
    Returns max_similarity (float between 0 and 1)
    """
    if train_fps is None or len(train_fps) == 0:
        return 0.0
    # vectorized intersection and union
    a = fp.astype(np.uint8)
    b = train_fps.astype(np.uint8)
    intersection = np.bitwise_and(b, a).sum(axis=1)
    union = a.sum() + b.sum(axis=1) - intersection
    # avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        sims = np.where(union > 0, intersection / union, 0.0)
    return float(np.max(sims))

def featurize_smiles(smiles: str, use_fp: bool = True, fp_radius: int = 2, fp_bits: int = 2048):
    """Return (feature_vector, descriptor_dict) for a smiles string.
    feature_vector is a 1D numpy array combining descriptors and fingerprint (if enabled).
    descriptor_dict contains human-readable RDKit descriptors.
    Returns (None, None) if SMILES invalid.
    """
    desc = compute_rdkit_descriptors(smiles)
    if desc is None:
        return None, None
    desc_vals = np.array([
        desc['MolWt'],
        desc['LogP'],
        desc['TPSA'],
        desc['NumHDonors'],
        desc['NumHAcceptors'],
        desc.get('NumRotatableBonds', 0),
        desc.get('RingCount', 0),
        desc.get('HeavyAtomCount', 0),
    ], dtype=float)
    if use_fp:
        fp = morgan_fingerprint_array(smiles, radius=fp_radius, nBits=fp_bits)
        if fp is None:
            return None, None
        feat = np.concatenate([desc_vals, fp.astype(float)])
    else:
        feat = desc_vals
    return feat, desc

def featurize_series(smiles_series: pd.Series, **kwargs):
    """Featurize a pandas Series of SMILES. Returns X (2D numpy) and list of descriptor dicts.
    Invalid SMILES rows are dropped.
    """
    feats = []
    descs = []
    valid_idx = []
    for idx, s in smiles_series.items():
        f, d = featurize_smiles(s, **kwargs)
        if f is None:
            continue
        feats.append(f)
        descs.append(d)
        valid_idx.append(idx)
    if not feats:
        return None, None, None
    X = np.vstack(feats)
    return X, descs, valid_idx


def align_features_for_pipeline(pipeline, feat: np.ndarray):
    """Pad or truncate a single feature vector to match a fitted pipeline's expected input size.

    This helps avoid shape errors when models were trained with a different featurizer.
    If the pipeline has a fitted scaler, we read `scaler.n_features_in_`.
    Otherwise we try `pipeline.n_features_in_` or the model's attribute.
    """
    if feat is None:
        return None
    expected = None
    try:
        if hasattr(pipeline, 'named_steps') and 'scaler' in pipeline.named_steps:
            scaler = pipeline.named_steps['scaler']
            expected = getattr(scaler, 'n_features_in_', None)
        if expected is None:
            expected = getattr(pipeline, 'n_features_in_', None)
        if expected is None and hasattr(pipeline, 'named_steps') and 'model' in pipeline.named_steps:
            expected = getattr(pipeline.named_steps['model'], 'n_features_in_', None)
    except Exception:
        expected = None

    if expected is None:
        return feat.astype(float)

    feat = np.asarray(feat, dtype=float)
    if feat.size == expected:
        return feat
    if feat.size < expected:
        pad = np.zeros((expected - feat.size,), dtype=float)
        return np.concatenate([feat, pad])
    # truncate if too long
    return feat[:expected].astype(float)
