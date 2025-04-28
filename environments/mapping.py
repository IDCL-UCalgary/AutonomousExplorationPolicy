import numpy as np
import math
import numpy as np

from environments.config import (alpha, beta, map_height, map_width, meas_phi,
                    rmax)


def get_range(map_matrix, X, meas_phi, rmax):
    height, width = map_matrix.shape
    x, y, theta = X
    meas_r = rmax * np.ones(meas_phi.shape)

    # Vectorized computation of cell coordinates
    r_values = np.arange(2, rmax)
    xi = x + np.outer(r_values, np.cos(theta + meas_phi)).astype(int)
    yi = y + np.outer(r_values, np.sin(theta + meas_phi)).astype(int)

    # Mask for valid coordinates
    valid_mask = (xi >= 0) & (xi < height) & (yi >= 0) & (yi < width)

    # Map look-up
    for i in range(len(meas_phi)):
        valid_cells = valid_mask[:, i]
        xi_valid = xi[valid_cells, i]
        yi_valid = yi[valid_cells, i]

        # Find the first occupied cell or out-of-bound
        occupied_cells = (map_matrix[xi_valid, yi_valid] == 1)
        if occupied_cells.any():
            meas_r[i] = r_values[valid_cells][occupied_cells.argmax()]
        else:
            meas_r[i] = r_values[valid_cells][-1] if valid_cells.any() else rmax

    return meas_r


def map_update(x, map_matrix, m):
    meas_r = get_range(map_matrix, x, meas_phi, rmax)

    sensor_range = rmax
    min_x, max_x = int(max(0, x[1] - sensor_range)), int(min(map_width, x[1] + sensor_range))
    min_y, max_y = int(max(0, x[0] - sensor_range)), int(min(map_height, x[0] + sensor_range))

    y, x_grid = np.ogrid[min_y:max_y, min_x:max_x]

    r = np.sqrt((y - x[0])**2 + (x_grid - x[1])**2)
    phi = (np.arctan2(x_grid - x[1], y - x[0]) - x[2] + np.pi) % (2*np.pi) - np.pi
    
    k = np.argmin(np.abs(phi - meas_phi[:, None, None]), axis=0)

    condition1 = (r > np.minimum(rmax, meas_r[k] + alpha / 2.0)) | (np.abs(phi - meas_phi[k]) > beta / 2.0)
    condition2 = (meas_r[k] < rmax) & (np.abs(r - meas_r[k]) < alpha / 2.0)
    condition3 = r < meas_r[k]

    update_mask = (m[min_y:max_y, min_x:max_x] == 0.5) & (~condition1)
    m[min_y:max_y, min_x:max_x] = np.where(update_mask & condition2, map_matrix[min_y:max_y, min_x:max_x], m[min_y:max_y, min_x:max_x])
    m[min_y:max_y, min_x:max_x] = np.where(update_mask & condition3, map_matrix[min_y:max_y, min_x:max_x], m[min_y:max_y, min_x:max_x])

    m[int(x[0]), int(x[1])] = map_matrix[int(x[0]), int(x[1])]
    return m



