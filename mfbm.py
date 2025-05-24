import numpy as np
from scipy.linalg import cholesky

def validate_parameters(H, rho_matrix):
    if not np.all((H > 0) & (H < 1)):
        raise ValueError("All Hurst parameters must be in the open interval (0, 1)")
    
    eigenvals = np.linalg.eigvals(rho_matrix)
    if not np.all(eigenvals > 0):
        raise ValueError("Correlation matrix must be positive definite")
    
    for i in range(len(H)):
        for j in range(len(H)):
            if abs(H[i] + H[j] - 1) < 1e-10:
                print(f"Warning: H[{i}] + H[{j}] ≈ 1, may cause numerical issues")

def mfbm(H, sigma, rho_matrix, n, T=1.0):
    times, X = np.zeros(n), np.zeros((p, n))
    for i, (t, x) in enumerate(mfbm_generator(H, sigma, rho_matrix, n, T)):
        times[i] = t
        X[:, i] = x
    return times, X

def mfbm_generator(H, sigma, rho_matrix, n, T=1.0):
    validate_parameters(H, rho_matrix)

    p = len(H)
    dt = T / n
    times = np.linspace(0, T, n+1)
    
    X = np.zeros((p, n+1))
    
    for k in range(n):
        cov_inc = np.zeros((p, p))
        
        for i in range(p):
            for j in range(p):
                if H[i] + H[j] == 1:
                    cov_inc[i, j] = sigma[i] * sigma[j] * rho_matrix[i, j] * dt
                else:
                    cov_inc[i, j] = sigma[i] * sigma[j] * rho_matrix[i, j] * \
                                   (dt**(H[i] + H[j])) / (H[i] + H[j])
        
        # Generate correlated increments
        try:
            L = cholesky(cov_inc, lower=True)
            Z = np.random.standard_normal(p)
            increments = L @ Z
        except:
            # Fallback to independent increments if correlation matrix is problematic
            increments = np.random.multivariate_normal(np.zeros(p), cov_inc)
        
        X[:, k+1] = X[:, k] + increments
        yield times[k+1], X[:, k+1]

