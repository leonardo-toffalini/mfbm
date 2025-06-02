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
        self.circulant_row = self.construct_circulant_row()
        self.C = block_circulant(self.circulant_row)

    def single_cov(self, H, h):
        if h == 0: return 1
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
        result = np.ndarray((self.p, self.p))
        for i in range(self.p):
            result[i, i] = self.single_cov(self.H[i], h)
            for j in range(i):
                if i == j:
                    assert False
                result[i, j] = result[j, i] = self.gamma_func(i, j, h)
        return result

        # return np.array([
        #     [self.gamma_func(i, j, h) for j in range(self.p)]
        #     for i in range(self.p)
        # ])

    def construct_C(self, j):
        if 0 <= j and j < self.m / 2:
            return self.construct_G(j)
        elif j == self.m / 2:
            return (self.construct_G(j) + self.construct_G(j)) / 2
        elif self.m / 2 < j and j <= self.m - 1:
            return self.construct_G(self.m - j)  # basic properties paper says other way around, Wood Chan says this way
        else:
            raise ValueError("argument j must be in the range [0, m-1]")

    def construct_circulant_row(self):
        # [self.construct_C(j) for j in range(self.m)]
        circulant_row = np.ndarray((self.m, self.p, self.p))  # m number of p x p matrices
        N = self.m // 2
        circulant_row[:N + 1] = [self.construct_G(i) for i in range(N + 1)]
        circulant_row[-N + 1:] = np.flip(circulant_row[1:N])
        return circulant_row

    def sample(self):
        pass

