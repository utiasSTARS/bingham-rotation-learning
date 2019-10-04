# 1D test of temporal calibration.
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----- Generate Data -----

# Simple velocimeter and position measurement system.
# Let's do the easiest thing possible.

# Equations are:
#
#  p(k + 1) = p(k) + v*dt
#  t_d(k + 1) = t_d(k)
p0 = 0.0
td = 0.25
dT = 0.1
# Ground truth - velocity update rate is 10 Hz.
t = np.arange(0, 1000.0, dT)
# w = 0.2   # rad/s - use later.
v = 1.0

tru_data = {
    't': t,
    'p': p0 + v * t,  # + v*np.sin(1/10/np.pi*t),
    'v': v * np.ones(t.shape),  # + 1/10/np.pi*v*np.cos(1/10/np.pi*t),
    'td': td}  # Lags relative to position updates of 200 ms.

# fig, ax = plt.subplots(1, 1)
# ax.plot(tru_data['t'], tru_data['p'])
# ax.set_title('Ground Truth')
# plt.show()

np.random.seed(12345)

# Velocity measurements are corrupted by noise.
# Q  = np.array([[0.0025]])  # sigma = 0.05 m/s
sigma_proc = 0.05
Q = np.array([[sigma_proc**2]])
vm = tru_data['v'] + np.random.normal(scale = sigma_proc, size = t.shape)

# Position measurements are corrupted by noise *and* time shifted.
sigma_meas = 0.0000000001
R  = np.array([[sigma_meas**2]])  # sigma = 0.05 m
pm = tru_data['p'] + np.random.normal(scale = sigma_meas, size = t.shape)

# Time shift the noisy position data by chopping the first two entries.
pm = pm[int(round(td/dT)):]
# fig, ax = plt.subplots(1, 1)
# ax.plot(tru_data['t'], vm)
# ax.set_title('Noisy Velocity')
# plt.show()

# fig, ax = plt.subplots(1, 1)
# ax.plot(tru_data['t'][:-2], pm)
# ax.set_title('Noisy Position')
# plt.show()

# ----- Setup Problem -----

def measurement_update(x_check, P_check, v, y):
    # Build measurement Jacobian
    H = np.zeros((1, 2))
    H[0, 0] = 1
    H[0, 1] = v #H_td
    # Compute Kalman Gain
    y_res = y - x_check[0] #np.dot(H, x_check)
    # print('y_res: ')
    # print(y_res)
    S = np.dot(H, np.dot(P_check, H.T)) + R
    K = np.dot(P_check, H.T)/S
    # Correct predicted state.
    x_hat = x_check + K*y_res
    # Compute corrected covariance
    P_hat = np.dot(np.eye(2) - np.dot(K, H), P_check)
    return x_hat, P_hat


if __name__=='__main__':

    # ----- Parameters -----
    x = np.zeros((1, 2))
    x[0] = np.array([p0, 0.0])
    P = np.zeros((1, 2, 2))
    P[0] = np.array([[0.0, 0.0], [0.0, 0.36]])  # initial td sigma is 0.6

    # Why start at j=1?
    j = 0  # POS measurement index
    k = 0  # VEL measurement index

    x_check = np.zeros([2, 1])
    x_check[0] = x[0, 0]
    x_check[1] = x[0, 1]
    P_check = P[0, :, :].copy() #np.zeros([2, 2])
    tc = t[0]
    ts = tc.copy()

    # Loop while we have data to process
    while tc <= t[pm.shape[0] - 1] and k < vm.shape[0]:
        # Figure out how far to propagate the velocity update.
        t_needed = j*dT + x_check[1, 0]
        do_update = tc >= t_needed
        # Have a measurement to process?
        # print(tc)
        if do_update:
            # Interpolate measurement/position
            v_in = np.interp(tc, t, vm)
            x_check, P_check = \
                measurement_update(x_check, P_check, v_in, pm[j])
            tc = j*dT + x_check[1, 0]
            j += 1
        else:
            # Propagate to t_needed
            # t_next = k*dT
            while tc < t_needed and k < vm.shape[0]:
                dT_next = t_needed - tc
                if dT_next > dT:
                    x_check[0] += vm[k] * dT
                    P_check = P_check + np.array([[Q[0, 0] * dT ** 2, 0.0], [0.0, 0.0]])
                    tc += dT
                else:
                    x_check[0] += vm[k] * dT_next
                    P_check = P_check + np.array([[Q[0, 0] * dT_next ** 2, 0.0], [0.0, 0.0]])
                    tc = t_needed
                k += 1

        # # Store the result
        # if do_update:
        #     P[-1, :, :] = P_check
        #     x[-1, :] = x_check.T
        # else:
        pp_check = np.zeros((1, 2, 2))
        pp_check[0] = P_check
        P = np.append(P, pp_check, axis=0)
        xx_check = np.zeros((1, 2))
        xx_check[0] = x_check.T
        x = np.append(x, xx_check, axis=0)
        ts = np.append(ts, tc)

    ## Plots
    # Three-sigma bounds.
    tsig = np.sqrt(P[:, 1, 1])

    fig, ax = plt.subplots(1, 1)
    ax.plot(ts, x[:, 1] - td)
    ax.plot(ts, -3 * tsig, color='red')
    ax.plot(ts, 3 * tsig, color='red')
    ax.set_title('Delay Estimate Error')
    plt.grid()
    plt.show()

    # Compute true positions at known timesteps.
    ptru = p0 + v * ts

    # Three-sigma bounds.
    psig = np.sqrt(P[:, 0, 0])

    fig, ax = plt.subplots(1, 1)
    ax.plot(ts, x[:, 0] - ptru)
    ax.plot(ts, -3 * psig, color='red')
    ax.plot(ts, 3 * psig, color='red')
    ax.set_title('Postion Estimate Error')
    plt.grid()
    plt.show()