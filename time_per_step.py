import time
import math
import numpy as np
import cvxpy as cp
from scipy.linalg import solve_triangular

def cholesky_update(l_tt, k_t, k_xt):
    lb = solve_triangular(l_tt, k_t, lower=True)
    schur = k_xt - np.matmul(lb.T, lb)
    ls = np.sqrt(schur)
    upper_block = np.concatenate((l_tt, np.zeros((l_tt.shape[0], 1))), axis=1)
    lower_block = np.concatenate((lb.T, ls), axis=1)
    l_t1t1 = np.concatenate((upper_block, lower_block), axis=0)
    return l_t1t1

def schur_inv_update(kinv_tt, k_t, k_xt, alpha):
    k_22 = 1/(k_xt + alpha - np.matmul(k_t.T, np.matmul(kinv_tt, k_t)))
    kinv_k = np.matmul(kinv_tt, k_t)
    k_11 = kinv_tt + k_22*(np.matmul(kinv_k, kinv_k.T))
    k_12 = -k_22*kinv_k
    k_21 = -k_22*kinv_k.T
    upper_block = np.concatenate((k_11, k_12), axis=1)
    lower_block = np.concatenate((k_21, k_22), axis=1)
    kinv_t1t1 = np.concatenate((upper_block, lower_block), axis=0)
    return kinv_t1t1

def schur_logdet_update(logdet_t, kinv_tt, k_t, k_xt, alpha):
    quad = np.matmul(k_t.T, np.matmul(kinv_tt, k_t))/alpha
    logdet_t1 = logdet_t + np.log(k_xt/alpha + 1.0 - quad)[0][0]
    return logdet_t1

def ay_gp_ucb(env, kernel, a, b, d, K, B, sigma, T, c, delta):
    times = np.zeros(T)

    start = time.time()
    actions = env.get_actions(0)
    action_idx = np.random.randint(0, K)
    reward, regret = env.step(0, action_idx)

    x = actions[action_idx].reshape(1, -1)
    y = reward.reshape(1, 1)
    alpha = sigma**2/c

    k_tt = kernel(x, x)
    kinv = 1 / (k_tt + alpha)
    logdet = np.log(1.0 + k_tt/alpha)[0][0]
    R = np.sqrt(c*logdet + 2*c*np.log(1/delta)) + B
    end = time.time()
    times[0] = end - start

    for t in range(1, T):
        start = time.time()
        actions = env.get_actions(t)
        k = kernel(actions, x)
        mean = np.matmul(k, np.matmul(kinv, y))
        var = np.ones((K,1)) - np.diag(np.matmul(k, np.matmul(kinv, k.T))).reshape(mean.shape)
        ucb = (mean + R*np.sqrt(var)).reshape(-1)
        action_idx = np.argmax(ucb)

        reward, regret = env.step(t, action_idx)

        x_t = actions[action_idx].reshape(1, -1)
        k_t = kernel(x, x_t)
        k_xt = kernel(x_t, x_t)
        logdet = schur_logdet_update(logdet, kinv, k_t, k_xt, alpha)
        kinv = schur_inv_update(kinv, k_t, k_xt, alpha)
        R = np.sqrt(c*logdet + 2*c*np.log(1/delta)) + B

        x = np.concatenate((x, x_t), axis=0)
        y = np.concatenate((y, reward.reshape(1, 1)), axis=0)
        end = time.time()
        times[t] = end - start

    return times

def igp_ucb(env, kernel, a, b, d, K, B, sigma, T, delta):
    times = np.zeros(T)

    start = time.time()
    actions = env.get_actions(0)
    action_idx = np.random.randint(0, K)
    reward, regret = env.step(0, action_idx)

    x = actions[action_idx].reshape(1, -1)
    y = reward.reshape(1, 1)
    lamb = 1 + 2/T

    k_tt = kernel(x, x)
    kinv = 1 / (k_tt + lamb)
    logdet = np.log(1.0 + k_tt/lamb)[0][0]
    R = sigma*np.sqrt(logdet + 2 + 2*np.log(1/delta)) + B
    end = time.time()
    times[0] = end - start

    for t in range(1, T):
        start = time.time()
        actions = env.get_actions(t)
        k = kernel(actions, x)
        mean = np.matmul(k, np.matmul(kinv, y))
        var = np.ones((K,1)) - np.diag(np.matmul(k, np.matmul(kinv, k.T))).reshape(mean.shape)
        ucb = (mean + R*np.sqrt(var)).reshape(-1)
        action_idx = np.argmax(ucb)

        reward, regret = env.step(t, action_idx)

        x_t = actions[action_idx].reshape(1, -1)
        k_t = kernel(x, x_t)
        k_xt = kernel(x_t, x_t)
        logdet = schur_logdet_update(logdet, kinv, k_t, k_xt, lamb)
        kinv = schur_inv_update(kinv, k_t, k_xt, lamb)
        R = sigma*np.sqrt(logdet + 2 + 2*np.log(1/delta)) + B

        x = np.concatenate((x, x_t), axis=0)
        y = np.concatenate((y, reward.reshape(1, 1)), axis=0)
        end = time.time()
        times[t] = end - start

    return times

def amm_ucb(env, kernel, a, b, d, K, B, sigma, T, c, delta):
    times = np.zeros(T)

    start = time.time()
    actions = env.get_actions(0)
    action_idx = np.random.randint(0, K)
    reward, regret = env.step(0, action_idx)

    x = actions[action_idx].reshape(1, -1)
    y = reward.reshape(1, 1)
    alpha = sigma**2/c

    k_tt = kernel(x, x)
    kinv = 1 / (k_tt + alpha)
    logdet = np.log(1.0 + k_tt/alpha)[0][0]
    R = np.sqrt(c*logdet + 2*c*np.log(1/delta) + B**2)
    end = time.time()
    times[0] = end - start

    for t in range(1, T):
        start = time.time()
        actions = env.get_actions(t)
        k = kernel(actions, x)
        mean = np.matmul(k, np.matmul(kinv, y))
        var = np.ones((K,1)) - np.diag(np.matmul(k, np.matmul(kinv, k.T))).reshape(mean.shape)
        ucb = (mean + R*np.sqrt(var)).reshape(-1)
        action_idx = np.argmax(ucb)

        reward, regret = env.step(t, action_idx)

        x_t = actions[action_idx].reshape(1, -1)
        k_t = kernel(x, x_t)
        k_xt = kernel(x_t, x_t)
        logdet = schur_logdet_update(logdet, kinv, k_t, k_xt, alpha)
        kinv = schur_inv_update(kinv, k_t, k_xt, alpha)
        R = np.sqrt(c*logdet + 2*c*np.log(1/delta) + B**2)

        x = np.concatenate((x, x_t), axis=0)
        y = np.concatenate((y, reward.reshape(1, 1)), axis=0)
        end = time.time()
        times[t] = end - start

    return times

def dmm_ucb(env, kernel, a, b, d, K, B, sigma, T, c, alphas, delta):
    times = np.zeros(T)

    start = time.time()
    actions = env.get_actions(0)
    action_idx = np.random.randint(0, K)
    reward, regret = env.step(0, action_idx)

    x = actions[action_idx].reshape(1, -1)
    y = reward.reshape(1, 1)
    alpha = sigma**2/c

    k_tt = kernel(x, x)
    kinv = 1 / (k_tt + alpha)
    kinvs = [1 / (k_tt + alphas[i]) for i in range(len(alphas))]
    logdet = np.log(1.0 + k_tt/alpha)[0][0]
    quad = alpha*np.dot(y.reshape(-1), np.dot(kinv, y.reshape(-1)))
    quads = [alphas[i]*np.dot(y.reshape(-1), np.dot(kinvs[i], y.reshape(-1))) for i in range(len(alphas))]
    R_sq = quad + sigma**2*logdet + 2*sigma**2*np.log(1/delta)
    rads = [math.sqrt(R_sq + alphas[i]*B**2 - quads[i]) for i in range(len(alphas))]
    end = time.time()
    times[0] = end - start

    for t in range(1, T):
        start = time.time()
        actions = env.get_actions(t)
        k = kernel(actions, x)
        ucbs = np.zeros((len(alphas), K))
        for i in range(len(alphas)):
            mean = np.matmul(k, np.matmul(kinvs[i], y))
            var = np.ones((K,1)) - np.diag(np.matmul(k, np.matmul(kinvs[i], k.T))).reshape(mean.shape)
            ucbs[i] = (mean + (rads[i]/math.sqrt(alphas[i]))*np.sqrt(var)).reshape(-1)
        ucb = ucbs.min(0)
        action_idx = np.argmax(ucb)

        reward, regret = env.step(t, action_idx)

        x_t = actions[action_idx].reshape(1, -1)
        k_t = kernel(x, x_t)
        k_xt = kernel(x_t, x_t)
        logdet = schur_logdet_update(logdet, kinv, k_t, k_xt, alpha)
        kinv = schur_inv_update(kinv, k_t, k_xt, alpha)
        for i in range(len(alphas)):
            kinvs[i] = schur_inv_update(kinvs[i], k_t, k_xt, alphas[i])

        x = np.concatenate((x, x_t), axis=0)
        y = np.concatenate((y, reward.reshape(1, 1)), axis=0)

        quad = alpha*np.dot(y.reshape(-1), np.dot(kinv, y.reshape(-1)))
        quads = [alphas[i]*np.dot(y.reshape(-1), np.dot(kinvs[i], y.reshape(-1))) for i in range(len(alphas))]
        R_sq = quad + sigma**2*logdet + 2*sigma**2*np.log(1/delta)
        rads = [math.sqrt(R_sq + alphas[i]*B**2 - quads[i]) for i in range(len(alphas))]
        end = time.time()
        times[t] = end - start

    return times

def cmm_ucb(env, kernel, a, b, d, K, B, sigma, T, c, delta, eps):
    times = np.zeros(T)

    start = time.time()
    actions = env.get_actions(0)
    action_idx = np.random.randint(0, K)
    reward, regret = env.step(0, action_idx)

    x = actions[action_idx].reshape(1, -1)
    y = reward.reshape(1, 1)
    alpha = sigma**2/c

    k_tt = kernel(x, x)
    l_tt = np.sqrt(k_tt + eps)
    kinv = 1 / (k_tt + alpha)
    logdet = np.log(1.0 + k_tt/alpha)[0][0]
    quad = alpha*y*kinv*y
    R = np.sqrt(quad + sigma**2*logdet + 2*sigma**2*np.log(1/delta))
    end = time.time()
    times[0] = end - start

    for t in range(1, T):
        start = time.time()
        actions = env.get_actions(t)
        ucb = np.zeros(K)
        for i in range(K):
            x_t = actions[i].reshape(1, -1)
            k_t = kernel(x, x_t)
            k_xt = kernel(x_t, x_t)
            k_t1 = np.concatenate((k_t, k_xt), axis=0).reshape(-1)
            k_tt1 = np.concatenate((k_tt, k_t), axis=1)
            l_t1t1 = cholesky_update(l_tt, k_t, k_xt + eps)

            _w = cp.Variable(t+1)
            obj = k_t1.T @ _w
            cons = [cp.norm(y.reshape(-1) - k_tt1 @ _w) <= R,
                    cp.norm(l_t1t1.T @ _w) <= B]
            prob = cp.Problem(cp.Maximize(obj), cons)
            ucb[i] = prob.solve()
        action_idx = np.argmax(ucb)

        reward, regret = env.step(t, action_idx)

        y = np.concatenate((y, reward.reshape(1, 1)), axis=0)
        x_t = actions[action_idx].reshape(1, -1)
        k_t = kernel(x, x_t)
        k_xt = kernel(x_t, x_t)
        k_tt = np.concatenate((np.concatenate((k_tt, k_t), axis=1),
                               np.concatenate((k_t.T, k_xt), axis=1)), axis=0)
        l_tt = cholesky_update(l_tt, k_t, k_xt + eps)
        logdet = schur_logdet_update(logdet, kinv, k_t, k_xt, alpha)
        kinv = schur_inv_update(kinv, k_t, k_xt, alpha)
        quad = alpha*np.dot(y.reshape(-1), np.dot(kinv, y.reshape(-1)))
        R = np.sqrt(quad + sigma**2*logdet + 2*sigma**2*np.log(1/delta))
        x = np.concatenate((x, actions[action_idx].reshape(1, -1)), axis=0)
        end = time.time()
        times[t] = end - start

    return times
