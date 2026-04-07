from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

LABELS = [
    "Computer Science",
    "Mathematics",
    "Physics",
    "Quantitative Biology",
    "Quantitative Finance",
    "Statistics",
    "Electrical Engineering and Systems Science",
    "Economics",
]

LABEL_TO_QUERY = {
    "Computer Science": "cat:cs*",
    "Mathematics": "cat:math*",
    "Physics": "("
    "cat:physics* OR cat:astro-ph* OR cat:cond-mat* OR cat:gr-qc "
    "OR cat:hep-ex OR cat:hep-lat OR cat:hep-ph OR cat:hep-th "
    "OR cat:math-ph OR cat:nlin* OR cat:nucl-ex OR cat:nucl-th OR cat:quant-ph"
    ")",
    "Quantitative Biology": "cat:q-bio*",
    "Quantitative Finance": "cat:q-fin*",
    "Statistics": "cat:stat*",
    "Electrical Engineering and Systems Science": "cat:eess*",
    "Economics": "cat:econ*",
}

PHYSICS_PREFIXES = (
    "physics",
    "astro-ph",
    "cond-mat",
    "gr-qc",
    "hep-ex",
    "hep-lat",
    "hep-ph",
    "hep-th",
    "math-ph",
    "nlin",
    "nucl-ex",
    "nucl-th",
    "quant-ph",
)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def build_input_text(title: str, abstract: str | None = None) -> str:
    title = normalize_whitespace(title)
    abstract = normalize_whitespace(abstract or "")
    if title and abstract:
        return f"[TITLE] {title} [ABSTRACT] {abstract}"
    if title:
        return f"[TITLE] {title}"
    if abstract:
        return f"[ABSTRACT] {abstract}"
    return ""


def map_primary_category_to_label(primary_category: str) -> str | None:
    primary_category = (primary_category or "").strip()
    if not primary_category:
        return None

    if primary_category.startswith("cs."):
        return "Computer Science"
    if primary_category.startswith("math."):
        return "Mathematics"
    if primary_category.startswith("q-bio"):
        return "Quantitative Biology"
    if primary_category.startswith("q-fin"):
        return "Quantitative Finance"
    if primary_category.startswith("stat."):
        return "Statistics"
    if primary_category.startswith("eess."):
        return "Electrical Engineering and Systems Science"
    if primary_category.startswith("econ."):
        return "Economics"

    if primary_category == "stat":
        return "Statistics"
    if primary_category == "econ":
        return "Economics"
    if primary_category == "eess":
        return "Electrical Engineering and Systems Science"
    if primary_category == "q-bio":
        return "Quantitative Biology"
    if primary_category == "q-fin":
        return "Quantitative Finance"
    if primary_category == "math":
        return "Mathematics"
    if primary_category == "cs":
        return "Computer Science"

    if primary_category.startswith(PHYSICS_PREFIXES):
        return "Physics"

    return None


def top_p_labels(probabilities: np.ndarray, labels: list[str], threshold: float = 0.95) -> list[tuple[str, float]]:
    order = np.argsort(probabilities)[::-1]
    selected: list[tuple[str, float]] = []
    cumulative = 0.0
    for idx in order:
        prob = float(probabilities[idx])
        selected.append((labels[idx], prob))
        cumulative += prob
        if cumulative >= threshold:
            break
    if not selected:
        return [(labels[int(np.argmax(probabilities))], float(np.max(probabilities)))]
    return selected


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_artifact(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str | Path):
    return joblib.load(path)


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"title", "abstract", "label", "primary_category", "arxiv_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В CSV не хватает колонок: {sorted(missing)}")
    df["title"] = df["title"].fillna("").astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str)
    df = df[df["label"].isin(LABELS)].copy()
    return df


def build_augmented_training_frame(df: pd.DataFrame, title_only_fraction: float = 0.35, random_state: int = 42) -> pd.DataFrame:
    full_df = df.copy()
    full_df["text"] = [
        build_input_text(title, abstract)
        for title, abstract in zip(full_df["title"], full_df["abstract"])
    ]

    if title_only_fraction <= 0:
        return full_df[["text", "label"]].sample(frac=1.0, random_state=random_state)

    title_only_df = (
        full_df.groupby("label", group_keys=False)
        .sample(frac=title_only_fraction, random_state=random_state)
        .reset_index(drop=True)
    )
    title_only_df["text"] = [build_input_text(title, None) for title in title_only_df["title"]]

    augmented = pd.concat(
        [full_df[["text", "label"]], title_only_df[["text", "label"]]],
        ignore_index=True,
    )
    augmented = augmented.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return augmented


def parse_arxiv_id(user_input: str) -> str:
    value = normalize_whitespace(user_input)
    if not value:
        return ""

    patterns = [
        r"arxiv\.org/(?:abs|pdf)/([^/?#]+)",
        r"^([a-z\-]+/\d{7}(?:v\d+)?)$",
        r"^(\d{4}\.\d{4,5}(?:v\d+)?)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            arxiv_id = match.group(1)
            return arxiv_id.replace(".pdf", "")
    return value.replace(".pdf", "")


@dataclass
class PredictorArtifacts:
    classifier_path: Path
    metadata_path: Path

    @property
    def classifier(self):
        return load_artifact(self.classifier_path)

    @property
    def metadata(self) -> dict:
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))
