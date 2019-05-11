from scipy.linalg import qr
import numpy as np

def bksvd(tensor, k = 6, block_size = None, num_iter = 3):
    """
    Block Krylov SVD

    Parameters
    ----------
    tensor : array (M, N) array_like
    k : int

    Returns
    -------
    u : (M, k) array_like
    s : (k,) array_like
    v : (k, N) array_like

    """
    if block_size is None:
        block_size = k
        
    k = min(k, min(tensor.shape))
    u = np.zeros((1, tensor.shape[1]))
        
    l = np.ones((tensor.shape[0], 1))
    
    K = np.zeros((tensor.shape[1], block_size * num_iter))
    block = np.random.randn(tensor.shape[1], block_size)
    
    block, _ = qr(block, mode='economic')
    
    T = np.zeros((tensor.shape[1], block_size))
    
    for i in range(num_iter):
        T = tensor @ block - l * (u @ block)
        block = tensor.T @ T - (u.T * (l.T @ T))
        block, _ = qr(block, mode='economic')
        K[:, i * block_size: (i + 1) * block_size] = block
    Q, _ = qr(K, mode='economic')
    
    # Rayleigh-Ritz
    T = tensor @ Q - l @ (u @ Q)
    
    Ut, St, Vt = np.linalg.svd(T, full_matrices=False)
    U = Ut[:, :k]
    S = St[:k]
    V = Q @ Vt.T
    return U, S, V[:, :k].T
