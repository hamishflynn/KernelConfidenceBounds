import numpy as np
from scipy.spatial import distance_matrix

class SquaredExponentialKernel:

    def __init__(self, rho):
        self.rho = rho

    def __call__(self, x1, x2):
        d = distance_matrix(x1, x2)
        out = np.exp(-d**2 / (2*self.rho**2))
        return out

class Matern12Kernel:

    def __init__(self, rho):
        self.rho = rho

    def __call__(self, x1, x2):
        d = distance_matrix(x1, x2)
        out = np.exp(-d/self.rho)
        return out

class Matern32Kernel:

    def __init__(self, rho):
        self.rho = rho

    def __call__(self, x1, x2):
        d = distance_matrix(x1, x2)
        out = (1 + 3**0.5 * d/self.rho) * np.exp(-3**0.5 * d/self.rho)
        return out

class Matern52Kernel:

    def __init__(self, rho):
        self.rho = rho

    def __call__(self, x1, x2):
        d = distance_matrix(x1, x2)
        out = (1 + 5**0.5 * d/self.rho + 5*d**2/(3*self.rho**2)) * np.exp(-5**0.5 * d/self.rho)
        return out

class RKHSFunc:

    def __init__(self, kernel, a, b, d, m, B):
        self.kernel = kernel
        self.ind_pts = np.random.uniform(a, b, (m, d))
        k_mm = self.kernel(self.ind_pts, self.ind_pts)
        w = np.random.normal(0.0, 1.0, (m,))
        w = w*B/np.sqrt(np.dot(w, np.dot(k_mm, w)))
        self.weights = w

    def __call__(self, x):
        k = self.kernel(x, self.ind_pts)
        y_hat = np.dot(k, self.weights).reshape(-1, 1)
        return y_hat

class KernelCBEnv:

    def __init__(self, kernel, a, b, m, d, K, B, sigma, T):
        self.f_star = RKHSFunc(kernel, a, b, d, m, B)
        self.sigma = sigma
        self.a = a
        self.b = b
        self.actions = np.random.uniform(a, b, (T, K, d))
        self.max_reward = self.f_star(self.actions.reshape(T*K, d)).reshape(T, K).max(1)

    def get_actions(self, t):
        return self.actions[t]

    def step(self, t, action_idx):
        exp_reward = self.f_star(self.actions[t][action_idx].reshape(1, -1)).reshape(1)
        noise = np.random.normal(0.0, self.sigma, (1,))
        reward = exp_reward + noise
        regret = self.max_reward[t] - exp_reward.reshape(1)
        return reward, regret
