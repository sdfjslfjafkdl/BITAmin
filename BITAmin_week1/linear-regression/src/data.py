import numpy as np

def make_regression_data(n=256, d=3, noise_std=0.1, seed=0):
    rng = np.random.default_rng(seed) # 항상 같은 seed를 줘서 매번 같은 난수를 뽑게 하는 난수 생성기를 만듦
    X = rng.normal(size=(n, d))

    W_true = rng.normal(size=(d, 1))
    b_true = rng.normal(size=(1, ))

    y = X @ W_true + b_true
    y = y + rng.normal(scale=noise_std, size=y.shape)

    return X, y, W_true, b_true