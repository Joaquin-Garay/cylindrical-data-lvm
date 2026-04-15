import numpy as np

from ..distributions import (
    Cylindrical,
    ExponentialFamily,
    IndGaussVM,
    MultivariateGaussian,
    UnivariateGaussian,
    VonMises,
    VonMisesFisher,
)


def _num_free_params_for_component(comp: ExponentialFamily | Cylindrical) -> int:
    """Return the number of free parameters for a single component.

    We count independent degrees of freedom (e.g., full symmetric covariance has d(d+1)/2,
    not d^2). This is used for AIC/BIC.
    """
    if isinstance(comp, MultivariateGaussian):
        d = comp.d
        return d + (d * (d + 1)) // 2
    if isinstance(comp, UnivariateGaussian):
        return 2
    if isinstance(comp, VonMises):
        return 2
    if isinstance(comp, VonMisesFisher):
        # mu lies on S^{d-1} -> d-1 DoF, plus concentration kappa.
        return comp.d
    if isinstance(comp, IndGaussVM):
        # composite: count the underlying parts
        return _num_free_params_for_component(comp.gaussian) + _num_free_params_for_component(comp.vonmises)
    if isinstance(comp, Cylindrical):
        d_g = comp.d_gauss
        d_v = comp.d_vmf
        # mu_gauss: d_g
        # cross_cov: d_g * d_v
        # cond_cov symmetric: d_g(d_g+1)/2
        # vmf: d_v (unit-direction + kappa)
        return d_g + (d_g * d_v) + (d_g * (d_g + 1)) // 2 + d_v

    # Fallback: best-effort count
    params = comp.params
    if not isinstance(params, (tuple, list)):
        params = (params,)
    count = 0
    for p in params:
        arr = np.asarray(p)
        if arr.ndim == 0:
            count += 1
        else:
            count += arr.size
    return count
