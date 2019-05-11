## Block Krylov Singular Value Decomposition (NumPy)

Python implementation of [https://github.com/cpmusco/bksvd](https://github.com/cpmusco/bksvd).

Example
-------

```python
import numpy as np

# Generate some low rank matrix
v = np.random.random(500)
a = v.reshape(-1, 1) @ v.reshape(1, -1)

# Note unlike MATLAB this returns V transpose
# to be more consistent with np.linalg.svd
u, s, vt = bksvd(a, 20)

# Check quality
assert np.allclose(a, u @ np.diag(s) @ vt)
print(np.linalg.norm(a - u @ np.diag(s) @ vt) / np.linalg.norm(a))
```
