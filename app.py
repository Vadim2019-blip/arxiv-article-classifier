from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import arxiv
import numpy as np
import pandas as pd
import streamlit as st

from common import build_input_text, parse_arxiv_id, top_p_labels

st.set_page_config(page_title="arXiv article classifier", layout="wide")

_ARTIFACTS_DIR = Path("artifacts/ft_model")
_THRESHOLD = 0.95

_FT_LABELS = [
    "Computer Science",
    "Economics",
    "Electrical Engineering and Systems Science",
    "Mathematics",
    "Physics",
    "Quantitative Biology",
    "Quantitative Finance",
    "Statistics",
]

_SCIBERT_CONFIG = {
    "architectures": ["BertForSequenceClassification"],
    "attention_probs_dropout_prob": 0.1,
    "classifier_dropout": None,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "id2label": {str(i): lbl for i, lbl in enumerate(_FT_LABELS)},
    "label2id": {lbl: i for i, lbl in enumerate(_FT_LABELS)},
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "num_labels": 8,
    "pad_token_id": 0,
    "torch_dtype": "float32",
    "type_vocab_size": 2,
    "vocab_size": 31090,
}

_TAR_MODEL_FILES = {"model.safetensors", "tokenizer_config.json", "tokenizer.json", "training_args.bin"}


def _prepare_ft_model_dir(tar_path: Path = Path("ft_model.tar")) -> None:
    model_dir = _ARTIFACTS_DIR / "model"
    metadata_path = _ARTIFACTS_DIR / "metadata.json"
    config_path = model_dir / "config.json"

    if metadata_path.exists() and config_path.exists():
        return

    if not tar_path.exists():
        return

    model_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            filename = Path(member.name).name
            if filename in _TAR_MODEL_FILES:
                f = tar.extractfile(member)
                if f is not None:
                    with open(model_dir / filename, "wb") as out:
                        shutil.copyfileobj(f, out)

    config_path.write_text(json.dumps(_SCIBERT_CONFIG, indent=2), encoding="utf-8")
    metadata_path.write_text(
        json.dumps({"labels": _FT_LABELS, "max_length": 384}, indent=2),
        encoding="utf-8",
    )


@st.cache_resource
def load_model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _prepare_ft_model_dir()
    model_dir = _ARTIFACTS_DIR / "model"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    return model, tokenizer


def predict(text: str, model, tokenizer) -> list[tuple[str, float]]:
    import torch
    import torch.nn.functional as F

    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=384, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
    return top_p_labels(np.asarray(probs), _FT_LABELS, threshold=_THRESHOLD)


def fetch_arxiv(arxiv_id_or_url: str) -> tuple[str, str]:
    parsed = parse_arxiv_id(arxiv_id_or_url)
    results = list(arxiv.Client().results(arxiv.Search(id_list=[parsed], max_results=1)))
    return results[0].title, results[0].summary

left, right = st.columns([1, 1])

with left:
    st.subheader("Ввод")
    arxiv_link = st.text_input("arXiv ID или ссылка (опционально)")
    if st.button("Подтянуть из arXiv"):
        fetched_title, fetched_abstract = fetch_arxiv(arxiv_link)
        st.session_state["title"] = fetched_title
        st.session_state["abstract"] = fetched_abstract

    title = st.text_input("Title", key="title")
    abstract = st.text_area("Abstract", height=260, key="abstract")
    predict_btn = st.button("Классифицировать", type="primary")

with right:
    st.subheader("Результат")
    if predict_btn:
        if not title.strip() and not abstract.strip():
            st.warning("Нужно ввести хотя бы title или abstract.")
        else:
            text = build_input_text(title, abstract if abstract.strip() else None)
            model, tokenizer = load_model()
            selected = predict(text, model, tokenizer)
            result_df = pd.DataFrame([{"Тема": lbl, "Вероятность": f"{100*p:.1f}%"} for lbl, p in selected])
            st.metric("Тем в топ-95%", len(result_df))
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            st.bar_chart(pd.DataFrame([{"Тема": lbl, "p": p} for lbl, p in selected]).set_index("Тема"))
