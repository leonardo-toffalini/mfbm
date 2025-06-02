import numpy as np

def random_corr_matrix(p, mean_corr):
    corr = np.full((p, p), mean_corr)
    noise = np.random.normal(scale=0.1, size=(p, p))
    np.fill_diagonal(noise, 0)
    corr += noise
    np.fill_diagonal(corr, 1)
    corr = (corr + corr.T) / 2  # make sure its symmetric
    return np.clip(corr, 0, 1)
