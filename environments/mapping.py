import numpy as np
import math
import numpy as np

from config import (alpha, beta, map_height, map_width, meas_phi,
                    rmax)


# def inverse_measurement(number_rows,number_columns,x,y,theta,meas_phi,meas_r,rmax,alpha,beta):
#   m = np.zeros((map_height,map_width))
#   for i in range(number_rows):
#     for j in range(number_columns):
#       r = np.sqrt((i-x)**2 + (j-y)**2)
#       phi = (math.atan2(j-y,i-x)- theta + np.pi) % (2*np.pi) - np.pi
#       k = np.argmin(np.abs(np.subtract(phi,meas_phi)))
#       if (r > min(rmax, meas_r[k] + alpha/2.0)) or (abs(phi-meas_phi[k])> beta/2.0) :
#         m[i,j] = 0.5
#       elif (meas_r[k] < rmax) and (abs(r-meas_r[k])<alpha/2.0):
#         m[i,j] = 0.8
#       elif r < meas_r[k]:
#         m[i,j] = 0.2
#   return m 

# def map_update(x,Llocal,localmap,map_matrix):
#   meas_r = get_range(map_matrix , x, meas_phi ,rmax)
#   invmod = inverse_measurement(map_height,map_width, x[0],x[1],x[2],meas_phi,meas_r,rmax,alpha,beta)
#   for i in range(map_height):
#     for j in range(map_width): 
#       if np.divide(invmod[i][j],np.subtract(1, invmod[i][j])) != 0 :
#         Llocal[i][j] += np.log(np.divide(invmod[i][j],np.subtract(1, invmod[i][j])))
#   localmap = np.exp(Llocal)/(1+np.exp(Llocal))
#   return Llocal,localmap


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



