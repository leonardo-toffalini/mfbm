import numpy as np
from scipy.linalg import cholesky

def simulate_mfbm_wood_chan(H, sigma, rho_matrix, eta_matrix, n, T=1.0):
    """
    Simulate multivariate fractional Brownian motion using Wood-Chan algorithm.
    
    Parameters:
    -----------
    H : array-like, shape (p,)
        Hurst parameters for each component (0 < H_i < 1)
    sigma : array-like, shape (p,)
        Scaling coefficients for each component
    rho_matrix : array-like, shape (p, p)
        Cross-correlation matrix (symmetric, diagonal = 1)
    eta_matrix : array-like, shape (p, p)
        Asymmetry parameters matrix
    n : int
        Number of time steps
    T : float
        Total time horizon
        
    Returns:
    --------
    times : ndarray, shape (n+1,)
        Time grid
    X : ndarray, shape (p, n+1)
        Simulated mfBm paths
    """
    p = len(H)  # Number of components
    times = np.linspace(0, T, n+1)
    dt = T / n
    
    # Initialize output
    X = np.zeros((p, n+1))
    
    # Covariance function for mfBm
    def cov_mfbm(s, t, Hi, Hj, sigma_i, sigma_j, rho_ij, eta_ij):
        """Covariance function between components i and j"""
        if Hi + Hj == 1:
            # Special case when Hi + Hj = 1
            if t == s:
                return sigma_i * sigma_j * rho_ij * np.abs(t)
            else:
                return sigma_i * sigma_j * rho_ij * 0.5 * (np.abs(s) + np.abs(t))
        else:
            # General case
            sign_diff = np.sign(t - s) if t != s else 0
            return (sigma_i * sigma_j / 2) * (rho_ij - eta_ij * sign_diff) * \
                   (np.abs(s)**(Hi + Hj) + np.abs(t)**(Hi + Hj) - np.abs(t - s)**(Hi + Hj))
    
    # Build covariance matrix for all time points and components
    cov_size = p * (n + 1)
    C = np.zeros((cov_size, cov_size))
    
    for i in range(p):
        for j in range(p):
            for k in range(n + 1):
                for l in range(n + 1):
                    idx_i = i * (n + 1) + k
                    idx_j = j * (n + 1) + l
                    C[idx_i, idx_j] = cov_mfbm(
                        times[k], times[l], 
                        H[i], H[j], 
                        sigma[i], sigma[j], 
                        rho_matrix[i, j], eta_matrix[i, j]
                    )
    
    # Ensure positive definiteness by adding small regularization
    C += 1e-10 * np.eye(cov_size)
    
    # Generate multivariate normal sample
    try:
        L = cholesky(C, lower=True)
        Z = np.random.standard_normal(cov_size)
        Y = L @ Z
        
        # Reshape to get mfBm paths
        Y = Y.reshape(p, n + 1)
        X = Y
        
    except np.linalg.LinAlgError:
        # Fallback: use eigenvalue decomposition if Cholesky fails
        eigenvals, eigenvecs = np.linalg.eigh(C)
        eigenvals = np.maximum(eigenvals, 1e-10)  # Ensure positive eigenvalues
        sqrt_eigenvals = np.sqrt(eigenvals)
        
        Z = np.random.standard_normal(cov_size)
        Y = eigenvecs @ (sqrt_eigenvals[:, np.newaxis] * (eigenvecs.T @ Z))
        X = Y.reshape(p, n + 1)
    
    return times, X

def simulate_mfbm_efficient(H, sigma, rho_matrix, eta_matrix, n, T=1.0):
    """
    More efficient implementation using increments approach.
    """
    p = len(H)
    dt = T / n
    times = np.linspace(0, T, n+1)
    
    # Initialize paths
    X = np.zeros((p, n+1))
    
    # Generate increments
    for k in range(n):
        t_curr = times[k]
        t_next = times[k+1]
        
        # Covariance matrix for increments at this step
        cov_inc = np.zeros((p, p))
        
        for i in range(p):
            for j in range(p):
                # Increment covariance
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
    
    return times, X

