from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

ML_LOGS_FEATURE_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "HeavyAtomCount",
]


class InvalidSmilesError(ValueError):
    pass


def compute_rdkit_descriptors(smiles: str) -> dict:
    """Compute the RDKit descriptors used by the LogS model."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise InvalidSmilesError(f"Invalid SMILES: {smiles}")

    return {
        "MolWt": Descriptors.MolWt(molecule),
        "MolLogP": Crippen.MolLogP(molecule),
        "TPSA": rdMolDescriptors.CalcTPSA(molecule),
        "NumHDonors": Lipinski.NumHDonors(molecule),
        "NumHAcceptors": Lipinski.NumHAcceptors(molecule),
        "NumRotatableBonds": Lipinski.NumRotatableBonds(molecule),
        "RingCount": Lipinski.RingCount(molecule),
        "HeavyAtomCount": molecule.GetNumHeavyAtoms(),
    }


def make_feature_vector(descriptors: dict, feature_names=ML_LOGS_FEATURE_NAMES):
    """Build a feature vector from descriptor dict in model feature order."""
    return [descriptors[name] for name in feature_names]
