from functools import lru_cache

import pandas as pd

from backend.config import DATA_DIR, DRUG_NAME_MAP_PATH, project_relative


SMILES_COLUMNS = ("SMILES", "smiles", "canonical_smiles", "CanonicalSMILES", "canonical_smiles")
MOLECULE_FILE_CANDIDATES = (
    DRUG_NAME_MAP_PATH,
    DATA_DIR / "molecule_smiles.csv",
    DATA_DIR / "molecules.csv",
)


def lookup_single(nsc):
    return _lookup_molecule(str(nsc or "").strip())


def lookup_pair(payload):
    payload = payload or {}
    nsc1 = str(payload.get("NSC1") or payload.get("drug1_id") or "").strip()
    nsc2 = str(payload.get("NSC2") or payload.get("drug2_id") or "").strip()
    molecule_1 = _lookup_molecule(nsc1)
    molecule_2 = _lookup_molecule(nsc2)

    return {
        "status": "success",
        "NSC1": molecule_1,
        "NSC2": molecule_2,
        "molecule_1": molecule_1,
        "molecule_2": molecule_2,
    }


def _lookup_molecule(nsc):
    record = _find_smiles_record(nsc)
    if not record:
        return _placeholder_molecule(nsc)

    smiles = record["smiles"]
    source = record["source"]
    try:
        return _rdkit_molecule(nsc, smiles, source)
    except Exception as exc:
        response = _placeholder_molecule(nsc)
        response.update({
            "smiles": smiles,
            "source": source,
            "message": f"Molecule SMILES is configured, but RDKit rendering is unavailable: {exc}",
        })
        return response


def _rdkit_molecule(nsc, smiles, source):
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import Draw, rdMolDescriptors  # type: ignore

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("SMILES could not be parsed.")

    drawer = Draw.MolDraw2DSVG(320, 220)
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    formula = rdMolDescriptors.CalcMolFormula(molecule)

    return {
        "status": "success",
        "requested_nsc": nsc,
        "used_nsc": nsc,
        "resolved_nsc": nsc,
        "alias_used": False,
        "used_alias": False,
        "molecule_found": True,
        "found": True,
        "molecular_formula": formula,
        "formula": formula,
        "smiles": smiles,
        "source": source,
        "structure_svg": svg,
        "svg": svg,
        "message": "Molecule details loaded from configured SMILES data.",
    }


def _placeholder_molecule(nsc):
    return {
        "status": "success",
        "requested_nsc": nsc,
        "used_nsc": nsc,
        "resolved_nsc": nsc,
        "alias_used": False,
        "used_alias": False,
        "molecule_found": False,
        "found": False,
        "molecular_formula": "",
        "formula": "",
        "smiles": "",
        "source": "not_configured",
        "structure_svg": "",
        "svg": "",
        "message": "Molecule details are not configured for this new project.",
    }


def _find_smiles_record(nsc):
    requested = str(nsc or "").strip()
    if not requested:
        return None

    for path in MOLECULE_FILE_CANDIDATES:
        table = _load_optional_csv(path)
        if table is None or table.empty:
            continue

        id_column = _find_column(table, ("id", "NSC", "nsc", "drug_id", "compound_id"))
        smiles_column = _find_column(table, SMILES_COLUMNS)
        if not id_column or not smiles_column:
            continue

        mask = table[id_column].astype(str).str.replace(r"\.0$", "", regex=True).str.strip().eq(requested)
        if not mask.any():
            continue

        smiles = str(table.loc[mask].iloc[0][smiles_column]).strip()
        if smiles and smiles.lower() != "nan":
            return {"smiles": smiles, "source": project_relative(path)}

    return None


@lru_cache(maxsize=None)
def _load_optional_csv(path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _find_column(frame, aliases):
    normalized = {_normalize(column): column for column in frame.columns}
    for alias in aliases:
        match = normalized.get(_normalize(alias))
        if match:
            return match
    return None


def _normalize(value):
    return str(value or "").strip().lower().replace(" ", "").replace("-", "").replace("_", "")
