import numpy as np
import jax.numpy as jnp
import jax

outer_add = jax.vmap(jax.vmap(lambda x, y: x+y, in_axes=(None, 0)), in_axes=(0, None))

def validate_parameters(H, rho_matrix):
    if not np.all((H > 0) & (H < 1)):
        raise ValueError("All Hurst parameters must be in the open interval (0, 1)")
    
    eigenvals = np.linalg.eigvals(rho_matrix)
    if not np.all(eigenvals > 0):
        raise ValueError("Correlation matrix must be positive definite")

def mfbm(H, sigma, rho_matrix, n, T=1.0, method="cholesky"):
    methods = {"cholesky": _cholesky, "cholesky_jax": _cholesky_jax}
    generator = methods[method]

    times, X = np.zeros(n), np.zeros((len(H), n))
    for i, (t, x) in enumerate(generator(H, sigma, rho_matrix, n, T)):
        times[i] = t
        X[:, i] = x
    return times, X

def _cholesky(H, sigma, rho_matrix, n, T=1.0):
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
            L = np.linalg.cholesky(cov_inc)
            Z = np.random.standard_normal(p)
            increments = L @ Z
        except:
            # Fallback to independent increments if correlation matrix is problematic
            increments = np.random.multivariate_normal(np.zeros(p), cov_inc)
        
        X[:, k+1] = X[:, k] + increments
        yield times[k+1], X[:, k+1]

def _cholesky_jax(H, sigma, rho_matrix, n, T=1.0):
    validate_parameters(H, rho_matrix)

    p = len(H)
    dt = T / n
    times = np.linspace(0, T, n+1)
    
    X = np.zeros((p, n+1))

    outer_add_H = outer_add(H, H)
    condition = outer_add_H == 1
    first_case = jnp.outer(sigma, sigma) * rho_matrix * dt
    second_case = jnp.outer(sigma, sigma) * rho_matrix * (dt ** outer_add_H) / outer_add_H
    cov_inc = jnp.where(condition, first_case, second_case)

    # Generate correlated increments
    L = np.linalg.cholesky(cov_inc)

    for k in range(n):
        Z = np.random.standard_normal(p)
        increments = L @ Z
        
        X[:, k+1] = X[:, k] + increments
        yield times[k+1], X[:, k+1]

