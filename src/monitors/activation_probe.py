from __future__ import annotations
import sys
from pathlib import Path

# Make the project root importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.utils.device import pick_device, default_dtype_for_device
from src.utils.model_loader import load_model, load_tokenizer, DEFAULT_MODEL


# ==========================================
# Config
# ==========================================
@dataclass
class ProbeConfig:
    model_id: str = DEFAULT_MODEL
    layer: int = -1                 # which hidden layer to take (supports negative indexing)
    token_pool: str = "last"        # "last" | "mean"
    classifier: str = "logreg"      # "logreg" | "linear_svm"
    C: float = 1.0                  # inverse regularization strength
    max_iter: int = 2000
    class_weight: Optional[str] = "balanced"
    random_state: int = 0


@dataclass
class ProbeArtifact:
    scaler_path: str
    clf_path: str
    meta_path: str


# ==========================================
# Feature extraction
# ==========================================
class ActivationFeaturizer:
    """Extract activation features from a HF CausalLM via output_hidden_states.

    We do not use fragile module-name hooks. Instead, we rely on model outputs
    with `output_hidden_states=True`, which returns per-layer hidden states for
    each token. This is model-agnostic across Llama/Qwen families.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, layer: int = -1, token_pool: str = "last"):
        self.model = model
        self.tok = tokenizer
        self.layer = layer
        assert token_pool in {"last", "mean"}
        self.token_pool = token_pool
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def encode_text(self, text: str, max_len: Optional[int] = None) -> np.ndarray:
        inp = self.tok(text, return_tensors="pt", truncation=True, max_length=max_len)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        out = self.model(**inp, output_hidden_states=True, use_cache=False)
        hs: Tuple[torch.Tensor, ...] = out.hidden_states  # tuple(len=layers+1) of [B,T,H]
        h = hs[self.layer]  # [1, T, H]
        if self.token_pool == "last":
            feat = h[0, -1].float().cpu().numpy()
        else:
            feat = h[0].float().mean(dim=0).cpu().numpy()
        return feat  # shape [H]

    @torch.no_grad()
    def encode_batch(self, texts: Sequence[str], max_len: Optional[int] = None) -> np.ndarray:
        # simple loop to avoid OOM; batch if needed later
        feats = [self.encode_text(t, max_len=max_len) for t in texts]
        return np.stack(feats, axis=0)


# ==========================================
# Training API
# ==========================================
class LinearProbe:
    def __init__(self, cfg: ProbeConfig):
        self.cfg = cfg
        self.device = pick_device()
        self.dtype = default_dtype_for_device(self.device)
        self.model: Optional[PreTrainedModel] = None
        self.tok: Optional[PreTrainedTokenizerBase] = None
        self.featurizer: Optional[ActivationFeaturizer] = None
        self.scaler: Optional[StandardScaler] = None
        self.clf = None

    def _ensure_model(self):
        if self.model is None or self.tok is None:
            # Preserve target device (might have been set externally, e.g., by HybridMonitor)
            target_device = self.device
            # Use force_device=True for specific CUDA devices (e.g., cuda:1) to avoid
            # device_map="auto" placing it on the wrong GPU
            force = target_device.type == "cuda" and target_device.index is not None
            print(f"[LinearProbe] Loading model to {target_device} (force_device={force})")
            self.model, _, self.dtype = load_model(
                self.cfg.model_id, target_device, self.dtype, force_device=force
            )
            # Keep our target device, don't use whatever load_model returned
            self.device = target_device
            self.tok = load_tokenizer(self.cfg.model_id)
            # Only move if not already on device and not distributed via device_map
            if not hasattr(self.model, 'hf_device_map') and self.device.type != "cpu":
                self.model.to(self.device)
            self.model.eval()
            # Verify actual device
            actual_device = next(self.model.parameters()).device
            print(f"[LinearProbe] Model loaded, actual device: {actual_device}")
            self.featurizer = ActivationFeaturizer(self.model, self.tok, layer=self.cfg.layer, token_pool=self.cfg.token_pool)

    def fit(self, texts: Sequence[str], labels: Sequence[int], max_len: Optional[int] = None) -> Dict[str, Any]:
        """Train a linear probe. labels should be 0=benign, 1=sabotage."""
        self._ensure_model()
        X = self.featurizer.encode_batch(texts, max_len=max_len)
        y = np.asarray(labels, dtype=np.int64)

        self.scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = self.scaler.fit_transform(X)

        if self.cfg.classifier == "logreg":
            self.clf = LogisticRegression(
                penalty="l2",
                C=self.cfg.C,
                max_iter=self.cfg.max_iter,
                class_weight=self.cfg.class_weight,
                random_state=self.cfg.random_state,
                n_jobs=None,
            )
        else:
            self.clf = LinearSVC(C=self.cfg.C, class_weight=self.cfg.class_weight, random_state=self.cfg.random_state, max_iter=self.cfg.max_iter)
        self.clf.fit(Xs, y)
        acc = float((self.clf.predict(Xs) == y).mean())
        return {"train_acc": acc, "n": int(len(y)), "dim": int(X.shape[1])}

    def predict_scores(self, texts: Sequence[str], max_len: Optional[int] = None) -> np.ndarray:
        self._ensure_model()
        assert self.scaler is not None and self.clf is not None, "Probe not trained or loaded"
        X = self.featurizer.encode_batch(texts, max_len=max_len)
        Xs = self.scaler.transform(X)
        if hasattr(self.clf, "decision_function"):
            s = self.clf.decision_function(Xs)
        else:
            s = self.clf.predict_proba(Xs)[:, 1]
        # map scores to 0-100 via sigmoid-like normalization for dashboarding
        # keep raw as well
        raw = np.asarray(s, dtype=np.float32)
        scaled = 100.0 / (1.0 + np.exp(-raw))
        return np.stack([raw, scaled], axis=1)

    def save(self, out_dir: str | os.PathLike) -> ProbeArtifact:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        scaler_path = str(Path(out_dir) / "scaler.joblib")
        clf_path = str(Path(out_dir) / "probe.joblib")
        meta_path = str(Path(out_dir) / "meta.json")
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.clf, clf_path)
        meta = {
            "cfg": asdict(self.cfg),
            "model_id": self.cfg.model_id,
            "layer": self.cfg.layer,
            "token_pool": self.cfg.token_pool,
            "sklearn": {
                "scaler": type(self.scaler).__name__ if self.scaler is not None else None,
                "clf": type(self.clf).__name__ if self.clf is not None else None,
            },
        }
        Path(meta_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return ProbeArtifact(scaler_path, clf_path, meta_path)

    def load(self, in_dir: str | os.PathLike) -> None:
        scaler_path = Path(in_dir) / "scaler.joblib"
        clf_path = Path(in_dir) / "probe.joblib"
        self.scaler = joblib.load(scaler_path)
        self.clf = joblib.load(clf_path)


# ==========================================
# Dataset helpers
# ==========================================

def load_text_label_jsonl(paths: Sequence[str | os.PathLike], text_key: str = "text", label_key: str = "label") -> Tuple[List[str], List[int]]:
    X: List[str] = []
    y: List[int] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                X.append(obj[text_key])
                y.append(int(obj[label_key]))
    return X, y


# ==========================================
# CLI
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and use a linear activation probe")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a probe from JSONL")
    p_train.add_argument("--model_id", default=DEFAULT_MODEL)
    p_train.add_argument("--layer", type=int, default=-1)
    p_train.add_argument("--token_pool", choices=["last", "mean"], default="last")
    p_train.add_argument("--classifier", choices=["logreg", "linear_svm"], default="logreg")
    p_train.add_argument("--C", type=float, default=1.0)
    p_train.add_argument("--out_dir", required=True)
    p_train.add_argument("--train", nargs="+", required=True, help="JSONL file(s) with {text, label}")

    p_score = sub.add_parser("score", help="Score text(s) with a trained probe")
    p_score.add_argument("--model_id", default=DEFAULT_MODEL)
    p_score.add_argument("--layer", type=int, default=-1)
    p_score.add_argument("--token_pool", choices=["last", "mean"], default="last")
    p_score.add_argument("--probe_dir", required=True)
    p_score.add_argument("--input", nargs="+", required=True, help="One or more raw text strings or JSONL files")

    args = parser.parse_args()

    if args.cmd == "train":
        cfg = ProbeConfig(
            model_id=args.model_id,
            layer=args.layer,
            token_pool=args.token_pool,
            classifier=args.classifier,
            C=args.C,
        )
        probe = LinearProbe(cfg)
        X, y = load_text_label_jsonl(args.train)
        stats = probe.fit(X, y)
        art = probe.save(args.out_dir)
        print(json.dumps({"stats": stats, "artifact": asdict(art)}, indent=2))

    elif args.cmd == "score":
        cfg = ProbeConfig(model_id=args.model_id, layer=args.layer, token_pool=args.token_pool)
        probe = LinearProbe(cfg)
        probe.load(args.probe_dir)
        texts: List[str] = []
        for item in args.input:
            p = Path(item)
            if p.exists():
                if p.suffix.lower() == ".jsonl":
                    X, _ = load_text_label_jsonl([p], label_key="label")
                    texts.extend(X)
                else:
                    texts.append(p.read_text(encoding="utf-8"))
            else:
                texts.append(item)
        scores = probe.predict_scores(texts)
        out = []
        for t, (raw, scaled) in zip(texts, scores):
            out.append({"text_preview": t[:120], "raw": float(raw), "scaled_0_100": float(scaled)})
        print(json.dumps(out, ensure_ascii=False, indent=2))
