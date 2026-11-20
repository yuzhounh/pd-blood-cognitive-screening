"""Presentation names; internal model and predictor identifiers remain unchanged."""

import re

MODEL_LABELS = {
    'Age_LR': 'LR - Age-only set',
    'Basic_LR': 'LR - Basic set',
    'LR': 'LR - Full set',
    'SVM': 'SVM',
    'RF': 'RF',
    'XGBoost': 'XGBoost',
    'CatBoost': 'CatBoost',
    'NN': 'NN',
}
PREDICTOR_SET_LABELS = {'Age_LR': 'Age-only set', 'Basic_LR': 'Basic set', 'LR': 'Full set'}

# Legacy labels remain immutable keys in saved analysis files and strategy IDs.
PREDICTOR_SET_DISPLAY = {'Set 1': 'Age-only set', 'Set 2': 'Basic set', 'Full set': 'Full set'}


def display_text(value):
    """Map retained result keys to reader-facing names, including comparisons."""
    return re.sub(r'\bSet(?:\s+|~)([12])\b',
                  lambda match: PREDICTOR_SET_DISPLAY['Set ' + match.group(1)],
                  str(value))


def require_semantic_display_names(text, location):
    """Reject a presentation regression without changing immutable result keys."""
    legacy = re.search(r'\bSet(?:\s+|~)[12]\b', text)
    if legacy:
        raise ValueError(f'Legacy predictor-set display name in {location}: {legacy.group()}')
