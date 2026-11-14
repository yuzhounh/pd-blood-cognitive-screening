"""Finite local feasibility experiment. Never exports participant rows publicly."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, log_loss, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
LABS = ["Basophils", "Eosinophils", "Hemoglobin", "Lymphocytes", "Monocytes", "NLR",
        "Neutrophils", "Platelets", "RBC", "Serum Chloride", "Serum Glucose",
        "Serum Potassium", "Serum Sodium", "Serum Uric Acid", "Total Protein", "WBC"]
FEATURES = ["age", "sex", "BMI"] + LABS
SUBTYPES = ["Basophils", "Eosinophils", "Lymphocytes", "Monocytes", "Neutrophils"]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_sheet(frame):
    """Apply uniform field rules, preserve patients and propagate missing inputs."""
    d = frame.copy()
    names = ["age", "sex", "height", "weight", "MCATOT", "COG"] + LABS
    for name in names:
        d[name] = pd.to_numeric(d[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    before = d[names].isna().sum()
    d.loc[~d["sex"].isin([0, 1]), "sex"] = np.nan
    d.loc[~d["age"].between(18, 100), "age"] = np.nan
    for name in ["height", "weight", "Hemoglobin", "Platelets", "RBC", "Serum Chloride",
                 "Serum Glucose", "Serum Potassium", "Serum Sodium", "Serum Uric Acid",
                 "Total Protein", "WBC"]:
        d.loc[d[name].le(0), name] = np.nan
    invalid_subtype = pd.DataFrame(False, index=d.index, columns=SUBTYPES)
    for name in SUBTYPES:
        invalid_subtype[name] = d[name].lt(0) | d[name].gt(d["WBC"])
        d.loc[invalid_subtype[name], name] = np.nan
    d["BMI"] = d["weight"] / (d["height"] / 100) ** 2
    d["NLR"] = d["Neutrophils"] / d["Lymphocytes"].where(d["Lymphocytes"].gt(0))
    d.loc[~d["MCATOT"].between(0, 30), "MCATOT"] = np.nan
    d.loc[~d["COG"].isin([0, 1]), "COG"] = np.nan
    info = {
        "n": len(d), "unique_ids": int(d["PATNO"].nunique()),
        "invalid_cbc_people": int(invalid_subtype.any(axis=1).sum()),
        "invalid_cbc_cells": invalid_subtype.sum().astype(int).to_dict(),
        "missing_before": before.astype(int).to_dict(),
        "missing_after": d[FEATURES + ["MCATOT", "COG"]].isna().sum().astype(int).to_dict(),
    }
    if info["unique_ids"] != len(d):
        raise ValueError("Duplicate participant identifiers within a source sheet")
    return d, info


def moca_target(score):
    score = np.asarray(score, dtype=float)
    return np.where(np.isfinite(score), (score < 26).astype(float), np.nan)


class CappedImputer(BaseEstimator, TransformerMixin):
    """Fit caps, median/mode and standardization using fitting rows only."""
    def fit(self, X, y=None):
        x = np.asarray(X, dtype=float)
        if np.isnan(x).all(axis=0).any():
            raise ValueError("Entire training predictor is missing")
        self.lower_ = np.nanquantile(x, .01, axis=0)
        self.upper_ = np.nanquantile(x, .99, axis=0)
        self.lower_[1], self.upper_[1] = 0, 1
        z = np.clip(x, self.lower_, self.upper_)
        self.fill_ = np.nanmedian(z, axis=0)
        vals, counts = np.unique(x[np.isfinite(x[:, 1]), 1], return_counts=True)
        self.fill_[1] = vals[np.argmax(counts)]
        z = np.where(np.isnan(z), self.fill_, z)
        self.mean_, self.scale_ = z.mean(axis=0), z.std(axis=0)
        self.scale_[self.scale_ == 0] = 1
        return self

    def transform(self, X):
        z = np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)
        z = np.where(np.isnan(z), self.fill_, z)
        return (z - self.mean_) / self.scale_


def model_pipeline(kind, config, C=1, params=None):
    seed = config["seed"]
    params = params or {}
    steps = [("prep", CappedImputer())]
    if kind in ["RF_SMOTE_reference", "RF_SMOTE_tuned"]:
        steps += [("smote", SMOTE(k_neighbors=5, random_state=seed)),
                  ("model", RandomForestClassifier(n_estimators=300, max_depth=params.get("max_depth",12),
                                                   min_samples_leaf=params.get("min_samples_leaf",1),
                                                   random_state=seed, n_jobs=config["n_jobs"]))]
    elif kind in ["CatBoost", "CatBoost_tuned"]:
        steps += [("model", CatBoostClassifier(iterations=300, depth=params.get("depth",4), learning_rate=.03,
                    l2_leaf_reg=params.get("l2_leaf_reg",3), loss_function="Logloss", verbose=False, allow_writing_files=False,
                    random_seed=seed, thread_count=config["n_jobs"]))]
    else:
        penalty = "l1" if kind == "LASSO_LR" else "l2"
        steps += [("model", LogisticRegression(C=C, penalty=penalty, solver="liblinear",
                    max_iter=3000, random_state=seed))]
    return Pipeline(steps)


def choose_C(kind, X, y, folds, config):
    candidates = config["lasso_C"] if kind == "LASSO_LR" else config["ridge_C"]
    rows = []
    for c in candidates:
        scores = cross_val_score(model_pipeline(kind, config, c), X, y,
                                 cv=folds, scoring="average_precision", n_jobs=1)
        rows.append({"C": c, "mean_AP": float(scores.mean()),
                     "se_AP": float(scores.std(ddof=1) / np.sqrt(len(scores)))})
    best = max(rows, key=lambda r: r["mean_AP"])
    if kind == "LASSO_LR":
        chosen = min(r["C"] for r in rows if r["mean_AP"] >= best["mean_AP"] - best["se_AP"])
    else:
        chosen = best["C"]
    return chosen, rows


def choose_tree(kind, X, y, folds, config):
    candidates = config["rf_candidates"] if kind == "RF_SMOTE_tuned" else config["catboost_candidates"]
    rows = []
    for params in candidates:
        scores = cross_val_score(model_pipeline(kind,config,params=params),X,y,
                                  cv=folds,scoring="average_precision",n_jobs=1)
        rows.append({"params":params,"mean_AP":float(scores.mean()),
                     "se_AP":float(scores.std(ddof=1)/np.sqrt(len(scores)))})
    best = max(rows,key=lambda r:r["mean_AP"])
    return best["params"], rows


def youden(y, p):
    fpr, tpr, cut = roc_curve(y, p)
    ok = np.isfinite(cut)
    return float(cut[ok][np.argmax((tpr - fpr)[ok])])


def triplet(y, p):
    return np.array([roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p)])


def metrics(y, p, threshold):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    z = logit(p)
    design = np.column_stack([np.ones(len(z)), z])
    fit = minimize(lambda b: np.mean(np.logaddexp(0, design @ b) - y * (design @ b)),
                   [0., 1.], jac=lambda b: design.T @ (expit(design @ b) - y) / len(y),
                   method="BFGS")
    intercept = minimize_scalar(lambda a: np.mean(np.logaddexp(0, z + a) - y * (z + a)),
                                bounds=(-20, 20), method="bounded")
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    auc, ap, brier = triplet(y, p)
    return {"AUROC": float(auc), "AUPRC": float(ap), "Brier": float(brier),
            "log_loss": float(log_loss(y, p)), "threshold": threshold,
            "accuracy": float(accuracy_score(y, p >= threshold)),
            "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
            "calibration_intercept_offset": float(intercept.x),
            "calibration_intercept_free": float(fit.x[0]) if fit.success else None,
            "calibration_slope": float(fit.x[1]) if fit.success else None,
            "calibration_converged": bool(fit.success)}


def tabpfn_worker(input_path, output_path, estimators):
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import ModelSource, prepend_cache_path
    torch.set_num_threads(4)
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    checkpoint = Path(prepend_cache_path(ModelSource.get_v3_5().default_filename))
    if not checkpoint.is_file():
        raise FileNotFoundError("Required locally cached TabPFN v3.5 checkpoint unavailable")
    a = np.load(input_path)
    model = TabPFNClassifier.create_default_for_version(ModelVersion("v3.5"),
                n_estimators=estimators, balance_probabilities=False,
                ignore_pretraining_limits=True, random_state=20261014, device="cpu",
                show_progress_bar=False)
    model.fit(a["X_train"], a["y_train"])
    pred = model.predict_proba(a["X_external"])[:, 1]
    np.save(output_path, pred)
    write_json(Path(output_path).with_suffix(".json"), {"checkpoint_sha256": sha(checkpoint),
               "estimators": estimators, "device": "cpu", "n_predictions": len(pred)})


def main(config_name="config.json"):
    started = time.time()
    config_file = ROOT / "code" / config_name
    config = json.loads(config_file.read_text(encoding="utf-8"))
    source = ROOT / config["input"]
    if sha(source) != config["input_sha256"]:
        raise RuntimeError("Frozen source hash changed")
    results = ROOT / config.get("results_dir", "manuscript/results")
    private = ROOT / config.get("derived_dir", "data/derived")
    results.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    provenance = {"started_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha(source), "code_sha256": sha(__file__),
        "config_sha256": sha(config_file),
        "protocol_sha256": sha(ROOT / config.get("protocol_path", "review/pilot_protocol.md")), "python": sys.version,
        "packages": {x: importlib.metadata.version(x) for x in
                     ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "catboost"]}}
    write_json(results / "launch.json", provenance)
    if source.suffix.lower() == ".csv":
        external_source = ROOT / config["external_input"]
        if sha(external_source) != config["external_sha256"]:
            raise RuntimeError("Frozen external source hash changed")
        sheets = {"PPMI": pd.read_csv(source), "FAHHAUST-PD": pd.read_csv(external_source)}
    else:
        sheets = pd.read_excel(source, sheet_name=["PPMI", "FAHHAUST-PD"])
    ppmi, qc_p = clean_sheet(sheets["PPMI"])
    external, qc_e = clean_sheet(sheets["FAHHAUST-PD"])
    write_json(results / "feasibility.json", {"PPMI": qc_p, "FAHHAUST": qc_e,
                 "features": FEATURES, "n_non_cognitive_features": len(FEATURES)})
    tasks = config.get("tasks", [("PD_MoCA", True, True), ("Published_COG", False, False), ("PD_COG", True, False)])
    all_metrics, differences, selections, task_summary = [], [], {}, {}
    foundations = {"status": "not_started"}
    for task, pd_only, use_moca in tasks:
        train = ppmi.loc[ppmi["COHORT_DEFINITION"].eq("Parkinson's Disease")].copy() if pd_only else ppmi.copy()
        y = moca_target(train["MCATOT"]) if use_moca else train["COG"].to_numpy(float)
        ye = moca_target(external["MCATOT"]) if use_moca else external["COG"].to_numpy(float)
        valid, ve = np.isfinite(y), np.isfinite(ye)
        train, y = train.loc[valid], y[valid].astype(int)
        ext, ye = external.loc[ve], ye[ve].astype(int)
        X, Xe = train[FEATURES].to_numpy(float), ext[FEATURES].to_numpy(float)
        task_summary[task] = {"n_train": len(y), "positive_train": int(y.sum()),
                              "n_external": len(ye), "positive_external": int(ye.sum())}
        expected = config.get("expected_cohorts", {}).get(task)
        if expected is None:
            expected = (749, 202, 197, 75) if task == "PD_MoCA" else ((1279, 491, 197, 122) if not pd_only else (749, 331, 197, 122))
        if tuple(task_summary[task].values()) != tuple(expected):
            raise RuntimeError(f"Task cohort/endpoint changed: {task}")
        print(f"TASK {task}: train {len(y)}/{y.sum()} positive; external {len(ye)}/{ye.sum()} positive", flush=True)
        folds = list(StratifiedKFold(config["cv_folds"], shuffle=True, random_state=config["seed"]).split(X, y))
        predictions, summaries = {}, {}
        for name in config.get("models", ["Demographic_LR", "Blood_Ridge_LR", "LASSO_LR", "RF_SMOTE_reference", "CatBoost"]):
            t0 = time.time()
            xx, xxe = (X[:, :3], Xe[:, :3]) if name == "Demographic_LR" else (X, Xe)
            c, scores = choose_C(name, xx, y, folds, config) if name in ["Blood_Ridge_LR", "LASSO_LR"] else (1, [])
            params = {}
            if name in ["RF_SMOTE_tuned", "CatBoost_tuned"]:
                params, scores = choose_tree(name,xx,y,folds,config)
            estimator = model_pipeline(name, config, c, params)
            oof = cross_val_predict(estimator, xx, y, cv=folds, method="predict_proba", n_jobs=1)[:, 1]
            threshold_youden = youden(y, oof)
            threshold = .5 if config.get("operating_point") == "0.5" else threshold_youden
            estimator.fit(xx, y)
            p = estimator.predict_proba(xxe)[:, 1]
            if not np.isfinite(p).all():
                raise RuntimeError("Nonfinite predictions")
            predictions[name] = p
            summaries[name] = metrics(ye, p, threshold)
            chosen = []
            if hasattr(estimator.named_steps["model"], "coef_"):
                names = FEATURES[:3] if name == "Demographic_LR" else FEATURES
                chosen = [n for n, b in zip(names, estimator.named_steps["model"].coef_[0]) if abs(b) > 1e-9]
            selections[f"{task}/{name}"] = {"C": c, "cv_candidates": scores,
                 "selected_features": chosen, "n_features_input": xx.shape[1], "duration_s": time.time() - t0,
                 "selected_params":params,"training_youden_threshold":threshold_youden}
            print(f"  {name} done in {time.time()-t0:.1f}s", flush=True)
        if task in config.get("foundation_tasks", ["PD_MoCA"]):
            prep = CappedImputer().fit(X)
            np.savez(private / "tfm_input.npz", X_train=prep.transform(X), y_train=y,
                     X_external=prep.transform(Xe))
            command = [sys.executable, __file__, "--tabpfn-worker", str(private / "tfm_input.npz"),
                       str(private / "tfm_predictions.npy"), str(config["tabpfn_estimators"])]
            t0 = time.time()
            try:
                proc = subprocess.run(command, capture_output=True, text=True,
                         timeout=config["tabpfn_timeout_seconds"],
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                (ROOT / "tmp/tabpfn.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
                if proc.returncode != 0:
                    foundations = {"status": "failed", "returncode": proc.returncode,
                                   "reason": proc.stderr[-1200:], "duration_s": time.time() - t0}
                else:
                    p = np.load(private / "tfm_predictions.npy")
                    if len(p) != len(ye) or not np.isfinite(p).all():
                        raise ValueError("Invalid TabPFN prediction array")
                    predictions["TabPFN_pilot"] = p
                    summaries["TabPFN_pilot"] = metrics(ye, p, .5)
                    foundations = {"status": "completed", "duration_s": time.time() - t0,
                         **json.loads((private / "tfm_predictions.json").read_text(encoding="utf-8"))}
            except subprocess.TimeoutExpired:
                foundations = {"status": "timeout", "duration_s": time.time() - t0,
                               "limit_seconds": config["tabpfn_timeout_seconds"]}
            print(f"  TabPFN pilot {foundations['status']}", flush=True)
            write_json(results / "foundation_status.json", foundations)
        row_table = pd.DataFrame({"PATNO": ext["PATNO"].to_numpy(), "y": ye, **predictions})
        row_table.to_csv(private / f"predictions_{task}.csv", index=False)
        names = list(predictions)
        rng = np.random.default_rng(config["seed"])
        positive, negative = np.flatnonzero(ye == 1), np.flatnonzero(ye == 0)
        metric_names = ["AUROC", "AUPRC", "Brier"] + (["accuracy"] if config.get("include_accuracy") else [])
        boot = np.empty((config["bootstrap_replicates"], len(names), len(metric_names)))
        for b in range(config["bootstrap_replicates"]):
            idx = np.r_[rng.choice(positive, len(positive), replace=True),
                        rng.choice(negative, len(negative), replace=True)]
            for j, name in enumerate(names):
                scores = triplet(ye[idx], predictions[name][idx])
                if config.get("include_accuracy"):
                    scores = np.r_[scores,np.mean((predictions[name][idx]>=summaries[name]["threshold"])==ye[idx])]
                boot[b, j] = scores
        for j, name in enumerate(names):
            row = {"task": task, "model": name, **task_summary[task], **summaries[name]}
            for k, metric in enumerate(metric_names):
                row[f"{metric}_lo"], row[f"{metric}_hi"] = np.quantile(boot[:, j, k], [.025, .975]).tolist()
            all_metrics.append(row)
            for base_name in config.get("references",["Demographic_LR", "RF_SMOTE_reference", "Blood_Ridge_LR"]):
                if name == base_name:
                    continue
                base_j = names.index(base_name)
                for k, metric in enumerate(metric_names):
                    lo, hi = np.quantile(boot[:, j, k] - boot[:, base_j, k], [.025, .975])
                    differences.append({"task": task, "model": name, "reference": base_name,
                        "metric": metric, "difference": summaries[name][metric] - summaries[base_name][metric],
                        "lo": float(lo), "hi": float(hi)})
        pd.DataFrame(all_metrics).to_csv(results / "external_metrics.csv", index=False)
        pd.DataFrame(differences).to_csv(results / "paired_differences.csv", index=False)
        write_json(results / "model_selection.json", selections)
        print(f"TASK {task} aggregate results saved", flush=True)
    write_json(results / "completion.json", {"status": "completed", "scope": "exploratory_feasibility_pilot",
          "tasks": task_summary, "foundation_status": foundations,
          "duration_s": time.time() - started, "completed_utc": datetime.now(timezone.utc).isoformat(),
          "provenance": provenance, "interval_scope": "fixed-fitted-model stratified external bootstrap"})
    print("COMPLETION_EVENT: project22_pilot_completed", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--tabpfn-worker":
        tabpfn_worker(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        try:
            with threadpool_limits(limits=4):
                parser = argparse.ArgumentParser()
                parser.add_argument("--config", default="config.json")
                arguments = parser.parse_args()
                main(arguments.config)
        except Exception:
            write_json(ROOT / "manuscript/results/completion.json", {"status": "failed", "traceback": traceback.format_exc()})
            raise
