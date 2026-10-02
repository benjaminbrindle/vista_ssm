from numpy import asarray, newaxis, sqrt, multiply, amax, matmul, divide, dot, diag, swapaxes
from numpy.linalg import svd, eigh


def _makearray(a):
    # Same helper numpy.linalg uses internally (private there, so copied here).
    new = asarray(a)
    wrap = getattr(a, "__array_wrap__", new.__array_wrap__)
    return new, wrap


def transpose(a):
    # numpy.linalg's transpose: swap the last two axes (not numpy.transpose).
    return swapaxes(a, -1, -2)


def diagonalization(a):
    w, v = eigh(dot(a.T, a))

    w = w[::-1]; v = v[:,::-1]
    s = sqrt(w)
    u = dot(a, v); u = dot(u, diag(s**(-1)))
    vt = v.T

    return u, s, vt


def pseudo_inverse(a, rcond=1e-8, hermitian=False):
    a, wrap = _makearray(a)
    rcond = asarray(rcond)
    a = a.conjugate()

    try:
        u, s, vt = svd(a, full_matrices=False, hermitian=hermitian)
        cutoff = rcond[..., newaxis] * amax(s, axis=-1, keepdims=True)
        large = s > cutoff
        s = divide(1, s, where=large, out=s)
        s[~large] = 0
        res = matmul(transpose(vt), multiply(s[..., newaxis], transpose(u)))
        return wrap(res)

    except:
        u, s, vt = diagonalization(a)
        cutoff = rcond[..., newaxis] * amax(s, axis=-1, keepdims=True)
        large = s > cutoff
        s = divide(1, s, where=large, out=s)
        s[~large] = 0
        res = matmul(transpose(vt), multiply(s[..., newaxis], transpose(u)))
        return wrap(res)