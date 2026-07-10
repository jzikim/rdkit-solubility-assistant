from typing import Optional

import pubchempy as pcp


def get_canonical_smiles(drug_name: str) -> Optional[str]:
    """Return canonical SMILES for a given drug name using PubChem."""
    if not drug_name or not drug_name.strip():
        return None

    try:
        compounds = pcp.get_compounds(drug_name.strip(), "name")
    except Exception:
        return None

    if not compounds:
        return None

    compound = compounds[0]
    return compound.canonical_smiles if compound and compound.canonical_smiles else None
