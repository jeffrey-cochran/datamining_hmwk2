from torch import zeros, ones, normal, norm, manual_seed
from torch import addmv, mm, device, cuda
from math import sqrt

class ForwardModel(object):

    def __init__(self, seed=0):
        manual_seed(0)
        cuda0 = device('cuda:0')
        self.W = normal(
            zeros((10,784)),
            ones((10,784))
        ).to(device=cuda0)
        self.w0 = normal(
            zeros((10, 1)),
            ones((10, 1))
        ).to(device=cuda0)
        cuda.synchronize()
        return

    def __call__(self, X):
        N = X.size()[1]
        return mm(self.W, X) + self.w0.expand(-1, N)

    def update(self, W_grad, w0_grad, step_size):
        self.W -= W_grad * step_size
        self.w0 -= w0_grad * step_size
        cuda.synchronize()
        return

    def weight_norm(self):
        self.norm = sqrt(norm(self.W)**2 + norm(self.w0)**2)
        cuda.synchronize()
        return self.norm

    def error(self, X, Y):
        N = float(X.size()[1])
        Yhat = self.__call__(X)
        #
        # Model prediction as largest entry
        Yhat_idx = Yhat.max(0).indices
        Y_idx = Y.max(0).indices
        #
        # Compare predictions
        Y_diff = Yhat_idx - Y_idx
        error_idx = Y_diff.nonzero()
        #
        return len(error_idx) / N
