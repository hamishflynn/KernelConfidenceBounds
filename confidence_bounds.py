import math
import numpy as np
import cvxpy as cp

def ay_gp_ucb_cbs(x_eval, x, y, kernel, c, sigma, B, delta):
    n = x.shape[0]
    alpha = sigma**2/c
    k_nn = kernel(x, x)
    kinv = np.linalg.inv(k_nn + alpha*np.eye(n))
    logdet = np.linalg.slogdet(k_nn/alpha + np.eye(n))[1]
    R = np.sqrt(c*logdet + 2*c*np.log(1/delta)) + B

    k = kernel(x_eval, x)
    mid = np.matmul(k, np.matmul(kinv, y))
    var = np.diag(kernel(x_eval, x_eval)).reshape(mid.shape) - \
    np.diag(np.matmul(k, np.matmul(kinv, k.T))).reshape(mid.shape)
    lcb = mid - R*np.sqrt(var)
    ucb = mid + R*np.sqrt(var)
    return lcb, ucb

def igp_ucb_cbs(x_eval, x, y, kernel, eta, sigma, B, delta):
    n = x.shape[0]
    k_nn = kernel(x, x)
    kinv = np.linalg.inv(k_nn + (1+eta)*np.eye(n))
    logdet = np.linalg.slogdet(k_nn/(1+eta) + np.eye(n))[1]
    R = sigma*np.sqrt(logdet + 2 + 2*np.log(1/delta)) + B

    k = kernel(x_eval, x)
    mid = np.matmul(k, np.matmul(kinv, y))
    var = np.diag(kernel(x_eval, x_eval)).reshape(mid.shape) - \
    np.diag(np.matmul(k, np.matmul(kinv, k.T))).reshape(mid.shape)
    lcb = mid - R*np.sqrt(var)
    ucb = mid + R*np.sqrt(var)
    return lcb, ucb

def amm_cbs(x_eval, x, y, kernel, c, sigma, B, delta):
    n = x.shape[0]
    alpha = sigma**2/c
    k_nn = kernel(x, x)
    kinv = np.linalg.inv(k_nn + alpha*np.eye(n))
    logdet = np.linalg.slogdet(k_nn/alpha + np.eye(n))[1]
    R = np.sqrt(c*logdet + 2*c*np.log(1/delta) + B**2)

    k = kernel(x_eval, x)
    mid = np.matmul(k, np.matmul(kinv, y))
    var = np.diag(kernel(x_eval, x_eval)).reshape(mid.shape) - \
    np.diag(np.matmul(k, np.matmul(kinv, k.T))).reshape(mid.shape)

    lcb = mid - R*np.sqrt(var)
    ucb = mid + R*np.sqrt(var)
    return lcb, ucb

def dmm_cbs(x_eval, x, y, kernel, c, sigma, B, delta, alphas):
    n = x.shape[0]
    n_eval = x_eval.shape[0]
    alpha = sigma**2/c
    k_nn = kernel(x, x)
    kinv = np.linalg.inv(k_nn + alpha*np.eye(n))
    kinvs = [np.linalg.inv(k_nn + alphas[i]*np.eye(n)) for i in range(len(alphas))]
    logdet = np.linalg.slogdet(k_nn/alpha + np.eye(n))[1]
    quad = alpha*np.dot(y.reshape(-1), np.dot(kinv, y.reshape(-1)))
    quads = [alphas[i]*np.dot(y.reshape(-1), np.dot(kinvs[i], y.reshape(-1))) for i in range(len(alphas))]
    R_sq = quad + sigma**2*logdet + 2*sigma**2*np.log(1/delta)
    rads = [math.sqrt(R_sq + alphas[i]*B**2 - quads[i]) for i in range(len(alphas))]

    k = kernel(x_eval, x)
    lcbs = np.zeros((len(alphas), n_eval))
    ucbs = np.zeros((len(alphas), n_eval))
    for i in range(len(alphas)):
        mid = np.matmul(k, np.matmul(kinvs[i], y))
        var = np.diag(kernel(x_eval, x_eval)).reshape(mid.shape) - \
        np.diag(np.matmul(k, np.matmul(kinvs[i], k.T))).reshape(mid.shape)
        ucbs[i] = (mid + (rads[i]/math.sqrt(alphas[i]))*np.sqrt(var)).reshape(-1)
        lcbs[i] = (mid - (rads[i]/math.sqrt(alphas[i]))*np.sqrt(var)).reshape(-1)
    lcb = lcbs.max(0)
    ucb = ucbs.min(0)
    return lcb, ucb

def cmm_cbs(x_eval, x, y, kernel, c, sigma, B, delta, eps):
    n = x.shape[0]
    n_eval = x_eval.shape[0]
    alpha = sigma**2/c
    k_nn = kernel(x, x)
    kinv = np.linalg.inv(k_nn + alpha*np.eye(n))
    logdet = np.linalg.slogdet(k_nn/alpha + np.eye(n))[1]
    quad = alpha*np.dot(y.reshape(-1), np.dot(kinv, y.reshape(-1)))
    R = np.sqrt(quad + sigma**2*logdet + 2*sigma**2*np.log(1/delta))

    ucb = np.zeros(n_eval)
    lcb = np.zeros(n_eval)

    for i in range(n_eval):
        x_t = x_eval[i].reshape(1, -1)
        x_T = np.concatenate((x, x_t), axis=0)
        k_tt = kernel(x_T, x_T) + eps*(np.eye(n+1))
        l_tt = np.linalg.cholesky(k_tt).T
        k_tt1 = k_tt[0:-1]
        k_t = k_tt[:, -1]

        _w = cp.Variable(n+1)
        obj = k_t.T @ _w
        cons = [cp.norm(y.reshape(-1) - k_tt1 @ _w) <= R,
                cp.norm(l_tt @ _w) <= B]
        prob = cp.Problem(cp.Maximize(obj), cons)
        ucb[i] = prob.solve()

        _w = cp.Variable(n+1)
        obj = k_t.T @ _w
        cons = [cp.norm(y.reshape(-1) - k_tt1 @ _w) <= R,
                cp.norm(l_tt @ _w) <= B]
        prob = cp.Problem(cp.Minimize(obj), cons)
        lcb[i] = prob.solve()

    return lcb, ucb
