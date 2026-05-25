import numpy as np


def dislocation_field(r):
    nu = 1 / 3
    K_s = 1 / (2 * np.pi * (1 - nu))
    K_d = 1 / (2 * np.pi)

    # Stress calculations
    r_norm = np.linalg.norm(r)
    r4 = r_norm ** 4
    x, y = r[0], r[1]
    r2 = r_norm ** 2
    S_xx = -K_s * (y * (3 * x ** 2 + y ** 2)) / r4
    S_yy = K_s * (y * (x ** 2 - y ** 2)) / r4
    S_xy = K_s * (x * (x ** 2 - y ** 2)) / r4
    S_xz = 0
    S_yz = 0
    S_zz = nu * (S_xx + S_yy)
    stress = np.array([[S_xx, S_xy, S_xz], [S_xy, S_yy, S_yz], [S_xz, S_yz, S_zz]])

    # Displacement calculations
    u_x = K_d * (np.arctan2(y, x) + x * y / (2 * (1 - nu) * (r2)))
    u_y = -K_d * (((1 - 2 * nu) / (4 * (1 - nu))) * np.log(r2) + (x ** 2 - y ** 2) / (4 * (1 - nu) * r2))
    displacement = np.array([u_x, u_y, 0])

    return stress, displacement
