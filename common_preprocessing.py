"""Frozen pilot preprocessing, extracted without historical SMOTE dependencies."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_curve
from body_rules import valid_height, valid_weight

LABS = ['Basophils', 'Eosinophils', 'Hemoglobin', 'Lymphocytes', 'Monocytes', 'NLR',
        'Neutrophils', 'Platelets', 'RBC', 'Serum Chloride', 'Serum Glucose',
        'Serum Potassium', 'Serum Sodium', 'Serum Uric Acid', 'Total Protein', 'WBC']
FEATURES = ['age', 'sex', 'BMI'] + LABS
SUBTYPES = ['Basophils', 'Eosinophils', 'Lymphocytes', 'Monocytes', 'Neutrophils']


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_sheet(frame):
    """Apply uniform field rules, preserve patients and propagate missing inputs."""
    d = frame.copy()
    names = ['age', 'sex', 'height', 'weight', 'MCATOT', 'COG'] + LABS
    for name in names:
        d[name] = pd.to_numeric(d[name], errors='coerce').replace([np.inf, -np.inf], np.nan)
    before = d[names].isna().sum()
    d.loc[~d['sex'].isin([0, 1]), 'sex'] = np.nan
    d.loc[~d['age'].between(18, 100), 'age'] = np.nan
    for name in ['height', 'weight', 'Hemoglobin', 'Platelets', 'RBC', 'Serum Chloride',
                 'Serum Glucose', 'Serum Potassium', 'Serum Sodium', 'Serum Uric Acid',
                 'Total Protein', 'WBC']:
        d.loc[d[name].le(0), name] = np.nan
    d.loc[~valid_height(d["height"]), "height"] = np.nan
    d.loc[~valid_weight(d["weight"]), "weight"] = np.nan
    invalid_subtype = pd.DataFrame(False, index=d.index, columns=SUBTYPES)
    for name in SUBTYPES:
        invalid_subtype[name] = d[name].lt(0) | d[name].gt(d['WBC'])
        d.loc[invalid_subtype[name], name] = np.nan
    d['BMI'] = d['weight'] / (d['height'] / 100) ** 2
    d['NLR'] = d['Neutrophils'] / d['Lymphocytes'].where(d['Lymphocytes'].gt(0))
    d.loc[~d['MCATOT'].between(0, 30), 'MCATOT'] = np.nan
    d.loc[~d['COG'].isin([0, 1]), 'COG'] = np.nan
    info = {
        'n': len(d), 'unique_ids': int(d['PATNO'].nunique()),
        'invalid_cbc_people': int(invalid_subtype.any(axis=1).sum()),
        'invalid_cbc_cells': invalid_subtype.sum().astype(int).to_dict(),
        'missing_before': before.astype(int).to_dict(),
        'missing_after': d[FEATURES + ['MCATOT', 'COG']].isna().sum().astype(int).to_dict(),
    }
    if info['unique_ids'] != len(d):
        raise ValueError('Duplicate participant identifiers within a source sheet')
    return d, info


class CappedImputer(BaseEstimator, TransformerMixin):
    """Fit caps, median/mode and standardization using fitting rows only."""
    def fit(self, X, y=None):
        x = np.asarray(X, dtype=float)
        if np.isnan(x).all(axis=0).any():
            raise ValueError('Entire training predictor is missing')
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


def youden(y, p):
    fpr, tpr, cut = roc_curve(y, p)
    ok = np.isfinite(cut)
    return float(cut[ok][np.argmax((tpr - fpr)[ok])])
