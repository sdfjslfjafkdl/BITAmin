import numpy as np
from data import make_regression_data
from linear import Linear
from losses import MSELoss
from optim import SGD

def main():
    X, y, W_true, b_true = make_regression_data(n=512, d=5, noise_std=0.2, seed=42)

    model = Linear(in_dim=X.shape[1], out_dim=1, seed=1)
    loss_fn = MSELoss()
    opt = SGD(lr=0.05)

    for epoch in range(1, 501):
        #forward
        y_pred = model.forward(X)
        loss = loss_fn.forward(y_pred, y)

        #backward
        dY = loss_fn.backward()
        model.backward(dY)

        #update
        opt.step(model)

        if epoch % 50 == 0 or epoch == 1:
            # 파라미터가 true에 가까워지는지 대충 보기
            w_err = float(np.linalg.norm(model.W - W_true))
            b_err = float(np.linalg.norm(model.b - b_true))
            print(f"epoch {epoch:04d} | loss {loss:.6f} | ||W-W*|| {w_err:.4f} | ||b-b*|| {b_err:.4f}")

if __name__=="__main__":
    main()