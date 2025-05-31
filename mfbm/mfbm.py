import numpy as np

class MFBM:
    def __init__(self, H, n, rho, eta, sigma):
        self.H = H
        self.p = len(self.H)
        self.n = n
        self.rho = np.array(rho)
        print(self.rho)
        self.eta = np.array(eta)
        self.sigma = np.array(sigma)
        self.GG = np.block([
            [self.construct_G(np.abs(i - j)) for i in range(1, n + 1)]
            for j in range(1, n + 1)
        ])

    def w_func(self, i, j, h):
        if self.H[i] + self.H[j] == 1:
            return self.rho[i, j] - self.eta[i, j] * np.sign(h) * np.abs(h) ** (self.H[i] + self.H[j])
        else:
            return self.rho[i, j] * np.abs(h) + self.eta[i, j] * h * np.log(np.abs(h))

    def gamma_func(self, i, j, h):
        return (self.sigma[i] * self.sigma[j]) / 2 * (
            self.w_func(i, j, h - 1) - 2 * self.w_func(i, j, h) + self.w_func(i, j, h + 1)
        )

    def construct_G(self, h):
        return np.array([
            [self.gamma_func(i, j, h) for j in range(self.p)]
            for i in range(self.p)
        ])

