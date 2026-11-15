"""Stable binomial-likelihood calibration, independent of predictive fitting."""
import warnings
import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

def fit_calibration(y,z,weights,center,scale):
    """Likelihood fits for evaluation only; no probabilities are updated."""
    design=np.c_[np.ones(len(y)),(z-center)/scale]
    with warnings.catch_warnings():
        warnings.simplefilter('error',PerfectSeparationWarning)
        joint=sm.GLM(y,design,family=sm.families.Binomial(),freq_weights=weights).fit(
            maxiter=100,tol=1e-10)
        offset=sm.GLM(y,np.ones((len(y),1)),family=sm.families.Binomial(),
                      offset=z,freq_weights=weights).fit(maxiter=100,tol=1e-10)
    if not joint.converged or not offset.converged:
        raise RuntimeError('Calibration likelihood fit did not converge')
    slope=float(joint.params[1]/scale)
    intercept=float(joint.params[0]-slope*center)
    values=np.array([float(offset.params[0]),slope,intercept])
    if not np.isfinite(values).all():
        raise RuntimeError('Nonfinite calibration estimate')
    return values

