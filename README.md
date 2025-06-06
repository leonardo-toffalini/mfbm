# Multivariate fractional Brownian motion (mfBm)
This package provides implementation for the multivariate fractional Brownian motion and simple fractional brownian motion with the method described in [1].

### Installation
The mfbm package is available on PyPI and can be install via pip with the following command:
```bash
pip install mfbm
```

### Example usage
```python
import numpy as np
from mfbm import MFBM

p = 5
H = np.linspace(0.6, 0.9, 5)
n = 100
T = 100

rho = 0.7 * np.ones((p, p))
np.fill_diagonal(rho, 1)

eta = np.ones_like(rho)
sigma = np.ones(len(H))

mfbm = MFBM(H, rho, eta, sigma)
ts = mfbm.sample(n, T)
```

Explanation of the parameters:
- `H: np.ndarray` is a one dimensional array of the Hurst parameters, where all Hursts are in the range (0, 1).
- `rho: np.ndarray` is a two dimensional array of the cross correlations between the multiple fBms, default is the identity matrix.
- `sigma: np.ndarray` is a one dimensional array of the standard deviations of the single fBms, default is all ones.
- `n: int` is the number of increaments to generate for the multivariate fractional Gaussian noise.
- `T: float` is the time horizon of the mfBm, default is `n`.

![mfbm](https://github.com/user-attachments/assets/2fb0f6bf-58a5-4cd1-99e9-824bd8b17315)


[1] Andrew T. A. Wood & Grace Chan (1994): Simulation of Stationary
Gaussian Processes in [0, 1]^d , Journal of Computational and Graphical Statistics, 3:4,
409-432
