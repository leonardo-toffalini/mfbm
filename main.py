import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mfbm import simulate_mfbm_efficient
import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

if __name__ == "__main__":
    p = 5
    H = np.linspace(0.8, 0.9, p)
    sigma = np.ones_like(H)
    rho_matrix = 0.6 * np.ones((p, p)) + 0.4 * np.eye(p)
    eta_matrix = np.array([[0.0, 0.2], [-0.2, 0.0]])  # Asymmetry parameters
    
    n = 1000
    T = 100.0
    
    start = time.perf_counter()
    times, X = simulate_mfbm_efficient(H, sigma, rho_matrix, eta_matrix, n, T)
    print(f"took {time.perf_counter() - start:.4f} seconds to simulate")

    
    plt.figure(figsize=(12, 8))

    viridis = cm.get_cmap("viridis")
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


