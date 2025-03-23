import numpy as np
from gmm.gmm_utils import gmm_2_parameters, parameters_2_gmm, shape_DS, lyapunov
import pickle
from sympy import *
from math import sqrt, pow, acos
import matplotlib



matplotlib.use('Qt5Agg')
def angle_of_vector(v1, v2):
    pi = 3.1415
    vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
    length_prod = sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * sqrt(pow(v2[0], 2) + pow(v2[1], 2))
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    return (acos(cos) / pi) * 180

def GMR(x, Vxf, sigma, mu, priors, nargout=0):      # Ues this regression method as the modest policy
    # if rl is True:
    #     nbData = 1
    # else:
    nbData = x.shape[1]
    nbStates = sigma.shape[2]
    input = np.arange(0, Vxf['d'])
    output = np.arange(Vxf['d'], 2 * Vxf['d'])

    Pxi = []
    for i in range(nbStates):
        Pxi.append(priors[0, i] * gaussPDF(x, mu[input, i],
                                                     sigma[input[0]:(input[1] + 1),
                                                     input[0]:(input[1] + 1), i]))

    Pxi = np.reshape(Pxi, [len(Pxi), -1]).T
    beta = Pxi / np.tile(np.sum(Pxi, axis=1) + 1e-300, [nbStates, 1]).T

    #########################################################################
    y_tmp = []
    for j in range(nbStates):
        a = np.tile(mu[output, j], [nbData, 1]).T
        b = sigma[output, input[0]:(input[1] + 1), j]
        c = x - np.tile(mu[input[0]:(input[1] + 1), j], [nbData, 1]).T
        d = sigma[input[0]:(input[1] + 1), input[0]:(input[1] + 1), j]
        e = np.linalg.lstsq(d, b.T, rcond=-1)[0].T
        y_tmp.append(a + e.dot(c))

    y_tmp = np.reshape(y_tmp, [nbStates, len(output), nbData])
    y_tmp = np.reshape(y_tmp, [nbStates, len(output), nbData])
    # y_tmp = np.reshape(y_tmp, [nbStates, len(self.output)])
    beta_tmp = beta.T.reshape([beta.shape[1], 1, beta.shape[0]])
    y_tmp2 = np.tile(beta_tmp, [1, len(output), 1]) * y_tmp
    y = np.sum(y_tmp2, axis=0)
    ## Compute expected covariance matrices Sigma_y, given input x
    #########################################################################
    Sigma_y_tmp = []
    Sigma_y = []
    if nargout > 1:
        for j in range(nbStates):
            Sigma_y_tmp.append(
                sigma[output, output, j] - (
                            sigma[output, input, j] / (sigma[input, input, j]) *
                            sigma[input, output, j]))

        beta_tmp = beta.reshape(1, 1, beta.shape)
        Sigma_y_tmp2 = np.tile(beta_tmp * beta_tmp, [len(output), len(output), 1, 1]) * np.tile(Sigma_y_tmp,[1, 1, nbData,1])
        Sigma_y = np.sum(Sigma_y_tmp2, axis=3)
    return y, Sigma_y, beta

def gaussPDF(data, mu, sigma, rl=False):
    # if rl is True:
    #     nbdata = 1
    #     nbVar = 2
    # else:
    nbVar, nbdata = data.shape

    data = data.T - np.tile(mu.T, [nbdata, 1])
    prob = np.sum(np.linalg.lstsq(sigma, data.T, rcond=-1)[0].T * data, axis=1)
    prob = np.exp(-0.5 * prob) / np.sqrt((2 * np.pi) ** nbVar * np.abs(np.linalg.det(sigma) + 1e-300))
    return prob.T


def _get_vec_theta_in_car(vec, x, theta):
    vec_obs = vec - x
    vec_obs_in_car = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ]).dot(vec_obs)
    theta_obs_car = np.arctan2(vec_obs_in_car[1],
                                vec_obs_in_car[0])
    return theta_obs_car


def ds_stabilizer(x, obs, obs_theta, delta_obs_theta, theta,phi, option, Vxf, gmm_parameters, predictor, dx_rl, x_obs=None, r=None):
    u_rl = False      ###
    # u_rl = True

    dx_ = dx_rl.copy()
    d = Vxf['d']
    rho0 = 1
    kappa0 = 0.1
    if x.shape[0] == 2*d:
        dx = x[d+1:2*d, :]
        x = x[:d, :]

    if option == 'GMR':
        if u_rl is True:
            x_p = np.reshape(x, [d, 1])
            dx, _, _ = GMR(x_p, Vxf, gmm_parameters['sigma'], gmm_parameters['mu'], gmm_parameters['priors'])
            dx_ = dx[:, 0]
        else:
            dx_, _, _ = GMR(x, Vxf, gmm_parameters['sigma'], gmm_parameters['mu'], gmm_parameters['priors'])
    
    V, Vx = lyapunov(x, obs, Vxf['Priors'], Vxf['Mu'], Vxf['P'], Vxf['L'])
    # V, Vx = gmr_lyapunov(x, obs, Vxf['Priors'], Vxf['Mu'], Vxf['P'], Vxf['L_g'])
    Vx_ = Vx.copy()
    if option == 'rl':
        if obs > 1e-4 and np.abs(obs_theta) > 1e-8:              # takeover
            u = dx_ * 0
            dx_ = dx_ + u
            Vdot = np.sum(Vx * dx_, axis=0)
        else:
            norm_Vx = np.sum(Vx * Vx, axis=0)
            norm_x = np.sum(x * x, axis=0)
            Vdot = np.sum(Vx * dx_, axis=0)
            obs = np.squeeze(obs)
            rho = rho0 * (1 - np.exp(-kappa0 * norm_x)) * np.sqrt(norm_Vx)
            ind = Vdot + rho >= 0
            # ind = Vdot + 0.05*rho >= 0
            # ind = Vdot >= 0
            u = dx_ * 0
            if ind:
                lambder = (Vdot + rho) / (norm_Vx + 1e-8)
                u = -8 * lambder * Vx
                dx_ = dx_ + u

    else:
        mask = obs > 1e-4 
        if np.any(mask):
            norm_Vx = np.sum(Vx_ * Vx_, axis=0)
            norm_x = np.sum(x * x, axis=0)
            Vdot = np.sum(Vx_ * dx_, axis=0)
            obs = np.squeeze(obs)
            rho = rho0 * (1 - np.exp(-kappa0 * norm_x)) * np.sqrt(norm_Vx)
            ind = Vdot + rho >= 0
            # ind = Vdot >= 0
            # ind = Vdot + 0.05*rho >= 0
            u = dx_ * 0
            lambder = (Vdot + rho) / (norm_Vx + 1e-8)
            u = -10 * lambder * Vx_
            dx_ = dx_ + u
        else:
            norm_Vx = np.sum(Vx * Vx, axis=0)
            norm_x = np.sum(x * x, axis=0)
            Vdot = np.sum(Vx * dx_, axis=0)
            obs = np.squeeze(obs)
            rho = rho0 * (1-np.exp(-kappa0 * norm_x)) * np.sqrt(norm_Vx)
            # ind = Vdot >= 0
            ind = Vdot + rho >= 0
            u = dx_ * 0

            if np.sum(ind) > 0:
                lambder = (Vdot[ind] + rho[ind]) / (norm_Vx[ind] + 1e-8)
                u[:, ind] = -np.tile(11*lambder, [d, 1]) * Vx[:, ind]
                dx_[:, ind] = dx_[:, ind] + u[:, ind]
    return dx_, u, Vx, Vdot, V


def euclidean_distance(vectors, vector, Sum=True):
    diff = vectors - vector
    squared_diff = np.square(diff)
    if Sum is True:
        summed = np.sum(squared_diff, axis=0)
    else:
        summed = squared_diff
    distance = np.sqrt(summed)
    return distance  # 1xN

def barrier(x, x_so, r, k=1.6):  # x(state,length)
    g = [np.array(())]
    G = []
    r = r
    R = 0.18
    gain = 1
    xi = np.array(1e-12)

    nbData = np.shape(x)[1]
    num_obs = len(x_so[1, :])
    for i in range(num_obs):
        obs_R = np.array((r[i] + R) * k)
        x_obs = x_so[:, i]
        a = euclidean_distance(x, x_obs[:, np.newaxis])
        a = np.reshape(a, [1, nbData])
        theta = a - obs_R
        c = np.sqrt(np.square(theta) + 4 * xi)
        g = 0.5 * (c - theta)
        # obs_indx = g > 1e-6
        G.append(g * gain)
    G_obs = np.sum(G, axis=0)
    # print(self.G_obs )
    # G_obs_hist.append(self.G_obs)
    return G_obs

def obs_check(x, x_so, r, k=1.1):
    # r = 1.5
    r = r
    R = 0.45
    G = []
    xi = np.array(0.00000000001)

    for i in range(len(x_so[1, :])):
        obs_R = np.array((r[i] + R) * k)
        x_obs = x_so[:, i]
        a = euclidean_distance(x, x_obs[:, np.newaxis])
        a = np.reshape(a, [1, np.shape(x)[1]])
        theta = a - obs_R
        c = np.sqrt(np.square(theta) + 4 * xi)
        g = 0.5 * (c - theta)
        G.append(g.squeeze())
    obs_index = np.array(G) > 1e-7
    return obs_index


def load_V():
    with open('./G/G_202404062243_Ndemo=15000.pkl', 'rb') as g:
        gmm_parameters = pickle.load(g, encoding='bytes')
    with open('./V/V_202404062243_Ndemo=15000.pkl', 'rb') as v:
        Vxf = pickle.load(v, encoding='bytes')

    return gmm_parameters, Vxf













