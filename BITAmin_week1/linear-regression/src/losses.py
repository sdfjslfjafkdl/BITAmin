import numpy as np

class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        diff = self.y_pred - self.y_true
        mse = float(np.mean(diff * diff))
        return mse
    
    def backward(self):
        N = self.y_pred.shape[0]
        return (2.0 / N) * (self.y_pred - self.y_true)