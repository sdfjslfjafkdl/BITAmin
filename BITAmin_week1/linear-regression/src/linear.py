import numpy as np

class Linear:
    def __init__(self, in_dim, out_dim=1, seed=0):
        rng = np.random.default_rng(seed)
        # Weight initialization
        self.W = 0.01 * rng.normal(size=(in_dim, out_dim))
        self.b = np.zeros((out_dim,))
        #grads
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        #cache
        self.X = None

    def forward(self, X):
        """
        X: (N, D)
        returns: (N, out_dim)
        """
        self.X = X # backward를 위해 저장
        return X @ self.W + self.b

    def backward(self, dY):
        """
        dY: (N, out_dim)
        returns dX: (N, D)

        out = WX + b
        dY = dL / d(out) 
        dL / dW = (dL / d(out)) * (d(out) / dW)
        (d(out) / dW)) = X
        dL / db = (dL / d(out)) * (d(out) / db)
        """
        self.dW = self.X.T @ dY # (D, N) @ (N, out_dim)
        self.db = np.sum(dY, axis=0) # 행끼리 더해서 차원 맞춰줌 (out_dim, 1)

        dX = dY @ self.W.T # (N, out_dim) @ (out_dim, D)
        return dX