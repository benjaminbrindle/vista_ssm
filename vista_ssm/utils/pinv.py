import numpy as np

def pseudo_inverse(a, rcond=1e-8, hermitian=False):
    return np.linalg.pinv(a, rcond=rcond, hermitian=hermitian)
