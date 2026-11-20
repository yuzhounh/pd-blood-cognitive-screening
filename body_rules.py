"""Author-approved, outcome-independent anthropometry validity rules."""
import numpy as np

HEIGHT_MIN_CM = 100.0
HEIGHT_MAX_CM = 250.0
BODY_RULE_VERSION = "B128_height_100_250_preserve_participants"

def valid_height(height):
    return np.isfinite(height) & (height >= HEIGHT_MIN_CM) & (height <= HEIGHT_MAX_CM)

def valid_weight(weight):
    return np.isfinite(weight) & (weight > 0)
