import numpy as np

def block_circulant(blocks):
    r, c = blocks[0].shape
    B = np.zeros((len(blocks) * r, len(blocks) * c), dtype=blocks[0].dtype)
    for k in range(len(blocks)):
        for l in range(len(blocks)):
            kl = np.mod(k + l, len(blocks))
            B[r * kl : r * (kl + 1), c * k : c * (k + 1)] = blocks[l]
    return B

class MFBM:
    def __init__(self, H, n, rho, eta, sigma):
        self.H = H
        self.p = len(self.H)
        self.n = n
        self.m = 1 << (2 * n - 1).bit_length()  # smallest power of 2 greater than 2(n-1)
        self.rho = np.array(rho)
        self.eta = np.array(eta)
        self.sigma = np.array(sigma)
        self.GG = np.block([
            [self.construct_G(np.abs(i - j)) for i in range(1, n + 1)]
            for j in range(1, n + 1)
        ])
        self.C = block_circulant([self.construct_C(j) for j in range(self.m)])

    def single_cov(self, H, h):
        return ((h + 1) ** (2 * H) + (h - 1) ** (2 * H) - 2 * (h ** (2 * H))) / 2

    def w_func(self, i, j, h):
        if self.H[i] + self.H[j] == 1:
            return self.rho[i, j] * np.abs(h) + self.eta[i, j] * h * np.log(np.abs(h))
        elif h == 0:
            return 0
        else:
            return self.rho[i, j] - self.eta[i, j] * np.sign(h) * np.abs(h) ** (self.H[i] + self.H[j])

    def gamma_func(self, i, j, h):
        return (self.sigma[i] * self.sigma[j]) / 2 * (
            self.w_func(i, j, h - 1) - 2 * self.w_func(i, j, h) + self.w_func(i, j, h + 1)
        )

    def construct_G(self, h):
        return np.array([
            [self.gamma_func(i, j, h) for j in range(self.p)]
            for i in range(self.p)
        ])

    def construct_C(self, j):
        if 0 <= j and j < self.m / 2:
            return self.construct_G(j)
        elif j == self.m / 2:
            return (self.construct_G(j) + self.construct_G(j)) / 2
        elif self.m / 2 < j and j <= self.m - 1:
            return self.construct_G(j - self.m)
        else:
            raise ValueError("argument j must be in the range [0, m-1]")

