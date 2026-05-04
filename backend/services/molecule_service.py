def lookup_pair(payload):
    nsc1 = str((payload or {}).get("NSC1") or (payload or {}).get("drug1_id") or "").strip()
    nsc2 = str((payload or {}).get("NSC2") or (payload or {}).get("drug2_id") or "").strip()

    return {
        "status": "success",
        "NSC1": _placeholder_molecule(nsc1),
        "NSC2": _placeholder_molecule(nsc2),
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
        "source": "not_configured",
        "structure_svg": "",
        "svg": "",
        "message": "Molecule details are not configured for this new project.",
    }
