from mfbm.mfbm import MFBM
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, suppress=True, linewidth=1000)

def random_corr_matrix(p, mean_corr):
    corr = np.full((p, p), mean_corr)
    noise = np.random.normal(scale=0.1, size=(p, p))
    np.fill_diagonal(noise, 0)
    corr += noise
    np.fill_diagonal(corr, 1)
    corr = (corr + corr.T) / 2  # make sure its symmetric
    return np.clip(corr, 0, 1)

p = 5
H = np.linspace(0.6, 0.9, 5)
n = 100
m = 1 << (2 * n - 1).bit_length()

random_corr = True

if random_corr:
    rho = random_corr_matrix(len(H), 0.7)
else:
    rho = 0.7 * np.ones((p, p))
    np.fill_diagonal(rho, 1)

eta = np.ones_like(rho)
sigma = np.ones(len(H))

mfbm = MFBM(H, n, rho, eta, sigma)
ts = mfbm.sample()

fig = plt.figure(figsize=(14, 6))
ax1 = plt.subplot(1, 3, 1)
ax1.imshow(rho)
ax1.set_title("Correlations")

ax2 = plt.subplot(1, 3, (2, 4))
for i in range(len(H)):
    ax2.plot(range(n), ts[i], label=f'H{i}={H[i]:.2f}', alpha=0.7)
ax2.legend()
ax2.set_title("Realizations")

plt.show()


