class SGD:
    def __init__(self, lr=1e-2):
        self.lr = lr
    
    def step(self, layer):
        # layer가 W, b를 가진다고 가정
        layer.W -= self.lr * layer.dW
        layer.b -= self.lr * layer.db