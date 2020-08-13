def rotation_matrix_2d(theta_radians, dtype=None, backend=None):
    from .base import sin, cos, asarray
    c, s = cos(theta_radians), sin(theta_radians)
    return asarray(((c, -s), (s, c)), dtype=dtype, backend=backend)
