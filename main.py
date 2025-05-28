import numpy as np
import matplotlib.pyplot as plt
from mfbm import mfbm
import time

def corr_matrix(n, mean_corr):
    corr = np.full((n, n), mean_corr)
    noise = 0.01 * np.random.normal(scale=1, size=(n, n))
    np.fill_diagonal(noise, 0)
    corr += noise
    np.fill_diagonal(corr, 1)
    return np.clip(corr, 0, 1)

if __name__ == "__main__":
    p = 1
    H = np.linspace(0.01, 0.7, p)
    sigma = np.ones_like(H)
    mean_corr = 0.6

    print(f"{H = }")

    n = 1_000_000
    T = float(n)

    rho_matrix = corr_matrix(p, mean_corr)

    start = time.perf_counter()
    times, X = mfbm(H, sigma, rho_matrix, n, T, method="cholesky_jax")
    print(f"took {time.perf_counter() - start:.4f} seconds to simulate")
    
    plt.figure(figsize=(12, 8))

    viridis = plt.get_cmap("viridis")
    colors = viridis(np.linspace(0, 1, p))
    for i in range(p):
        plt.plot(times, X[i, :], color=colors[i], linewidth=1.0, alpha=0.7, label=f'Component {i} (H={H[i]:.02f})')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Multivariate Fractional Brownian Motion')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Simulation completed with {p} components and {n} time steps")
    print(f"Final values: Component 1 = {X[0, -1]:.3f}, Component 2 = {X[1, -1]:.3f}")


