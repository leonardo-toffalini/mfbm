from mfbm.mfbm import MFBM
import numpy as np

np.set_printoptions(precision=5, suppress=True, linewidth=1000)

def corr_matrix(p, mean_corr):
    corr = np.full((p, p), mean_corr)
    noise = np.random.normal(scale=0.1, size=(p, p))
    np.fill_diagonal(noise, 0)
    corr += noise
    np.fill_diagonal(corr, 1)
    corr = (corr + corr.T) / 2  # make sure its symmetric
    return np.clip(corr, 0, 1)

H = np.linspace(0.05, 0.25, 2)
n = 4
m = 1 << (2 * n - 1).bit_length()
p = H.size
rho = corr_matrix(len(H), 0.8)
eta = np.ones_like(rho)
sigma = np.ones(len(H))

mfbm = MFBM(H, n, rho, eta, sigma)

print(mfbm.C)
print(mfbm.C.shape)
print(f"{m}·{p} x {m}·{p}")
# print(mfbm.circulant_row)

