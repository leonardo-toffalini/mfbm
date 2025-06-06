from mfbm import MFBM
from mfbm import FBM
import numpy as np
import matplotlib.pyplot as plt
from mfbm.utils import random_corr_matrix

def mfbm_demo():
    p = 5
    H = np.linspace(0.6, 0.9, 5)
    n = 100
    T = 100

    random_corr = True

    if random_corr:
        rho = random_corr_matrix(len(H), 0.85)
    else:
        rho = 0.7 * np.ones((p, p))
        np.fill_diagonal(rho, 1)

    eta = np.ones_like(rho)
    sigma = np.ones(len(H))

    mfbm = MFBM(H, rho, eta, sigma)
    ts = mfbm.sample(n, T)

    fig = plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 4, 4)
    im = ax1.imshow(rho)
    fig.colorbar(im, orientation="vertical")
    ax1.set_title("Correlations")

    ax2 = plt.subplot(1, 4, (1, 3))
    for i in range(len(H)):
        ax2.plot(range(n), ts[i], label=f'H{i}={H[i]:.2f}', alpha=0.7)
    ax2.legend()
    ax2.set_title("Realizations")

    plt.show()

def fbm_demo():
    Hs, n, T = [0.1, 0.3], 100, 100
    for H in Hs:
        process = FBM(H=H, n=n, T=T)
        ts = process.sample()
        plt.plot(ts, label=f"{H=}", alpha=0.7)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    mfbm_demo()
    fbm_demo()

