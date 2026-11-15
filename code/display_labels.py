"""Presentation names; internal model and predictor identifiers remain unchanged."""

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
    text = str(value)
    for key, label in PREDICTOR_SET_DISPLAY.items():
        text = text.replace(key, label)
    return text
