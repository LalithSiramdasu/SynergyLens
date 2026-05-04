import pandas as pd

from backend.services import data_service


SCORE_ALIASES = (
    "final_predicted_COMBOSCORE",
    "COMBOSCORE",
    "ComboScore",
    "combo_score",
    "synergy_score",
    "score",
    "target",
)


def get_demo_cases():
    real_cases = _cases_from_dataset()
    if real_cases:
        return _ensure_three(real_cases)
    return _placeholder_cases()


def _cases_from_dataset():
    table = data_service.load_dataset()
    if table is None or table.empty:
        return []

    columns = data_service.get_identifier_columns(table)
    if not columns["nsc1"] or not columns["nsc2"]:
        return []

    score_column = _find_score_column(table)
    selected = []

    if score_column:
        scored = table.copy()
        scored["_demo_score"] = pd.to_numeric(scored[score_column], errors="coerce")
        scored = scored.dropna(subset=["_demo_score"])
        if not scored.empty:
            selected = [
                ("strong_synergy", scored.sort_values("_demo_score", ascending=False).iloc[0]),
                ("neutral", scored.iloc[(scored["_demo_score"].abs()).argsort()].iloc[0]),
                ("antagonism", scored.sort_values("_demo_score", ascending=True).iloc[0]),
            ]

    if not selected:
        labels = ("strong_synergy", "neutral", "antagonism")
        selected = list(zip(labels, [row for _, row in table.head(3).iterrows()]))

    cases = []
    for case_type, row in selected:
        cases.append({
            "case_type": case_type,
            "NSC1": _clean(row.get(columns["nsc1"])),
            "NSC2": _clean(row.get(columns["nsc2"])),
            "CELLNAME": _clean(row.get(columns["cell"])) if columns["cell"] else "General",
        })
    return cases


def _placeholder_cases():
    drugs = data_service.get_drugs()
    cell_lines = data_service.get_cell_lines()
    ids = [item["id"] if isinstance(item, dict) else str(item) for item in drugs]
    while len(ids) < 3:
        ids.append(str(1001 + len(ids)))
    cellname = cell_lines[0] if cell_lines else "General"

    return [
        {"case_type": "strong_synergy", "NSC1": ids[0], "NSC2": ids[1], "CELLNAME": cellname},
        {"case_type": "neutral", "NSC1": ids[1], "NSC2": ids[2], "CELLNAME": cellname},
        {"case_type": "antagonism", "NSC1": ids[2], "NSC2": ids[0], "CELLNAME": cellname},
    ]


def _ensure_three(cases):
    output = list(cases[:3])
    placeholders = _placeholder_cases()
    for fallback in placeholders:
        if len(output) >= 3:
            break
        output.append(fallback)
    return output


def _find_score_column(frame):
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for alias in SCORE_ALIASES:
        match = normalized.get(alias.lower())
        if match:
            return match
    return None


def _clean(value):
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text
