import numpy as np

BETAS = ['beta1', 'beta2', 'beta3']


def compute(frame):
    result = frame.copy()
    v, dv, s = (frame[c].to_numpy() for c in ['Speed_Follower', 'Relative_speed', 'Spacing'])
    if not np.isfinite(frame.select_dtypes('number')).all().all() or (s <= 1e-6).any():
        raise ValueError('Beta input must be finite and spacing > 1e-6')
    v0, tau, s0, delta, a, b = 39.12, .96, 10., 20., 1.18, 8.9
    q = np.sqrt(a*b)
    star = s0 + v*tau + v*dv/(2*q)
    result['beta1'] = 2*a*star**2/s**3
    result['beta2'] = a*star*v/(s**2*q)
    # Preserve the requested original minus sign; no derivative correction here.
    result['beta3'] = -a*(delta*v**(delta-1)/v0**delta + 2*star/s**2*(tau-dv/(2*q)))
    return result.round(6)


def normal_bounds(normal_beta):
    """Use all already-trimmed rows, without trimming a second time."""
    return {c: [float(normal_beta[c].min()), float(normal_beta[c].max())] for c in BETAS}
