"""
parameter_bounds.py
"""

PARAMETERS = [
    ('b1', 0.001, 0.20),
    ('b2', 5.00, 25.00),
    ('x0', 5, 20),
    ('x1', 14, 20),
    ('c', 0.02, 2.0),
    ('c1', 0.8, 4.0),
    ('gamma1', 0.35, 0.7),
    ('gamma2', 1.167, 1.4),
    ('mu1', 1.167, 1.4),
    ('mu2', 1.75, 14),
    ('T_perceive', 0.5, 20.0),
    ('sensitivity', 0.005, 0.2),
    ('dispose_rate', 2.0, 7.5),
]

PARAM_NAMES = [p[0] for p in PARAMETERS]
BOUNDS = [(p[1], p[2]) for p in PARAMETERS]
K_DOUBLE = 13
K_SINGLE = 11  # single-sigmoid drops x1, c1

_DROPPED_FOR_SINGLE = {'x1', 'c1'}
PARAMETERS_SINGLE = [p for p in PARAMETERS if p[0] not in _DROPPED_FOR_SINGLE]
PARAM_NAMES_SINGLE = [p[0] for p in PARAMETERS_SINGLE]
BOUNDS_SINGLE = [(p[1], p[2]) for p in PARAMETERS_SINGLE]
assert len(PARAMETERS_SINGLE) == K_SINGLE

_DROPPED_FOR_NULL = {'x1', 'c1', 'T_perceive', 'sensitivity'}
PARAMETERS_NULL = [p for p in PARAMETERS if p[0] not in _DROPPED_FOR_NULL]
PARAM_NAMES_NULL = [p[0] for p in PARAMETERS_NULL]
BOUNDS_NULL = [(p[1], p[2]) for p in PARAMETERS_NULL]
K_NULL = 9
assert len(PARAMETERS_NULL) == K_NULL