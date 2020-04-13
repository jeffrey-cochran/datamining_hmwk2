from torch import zeros, ones, normal, t, cuda
from torch import addmv, mm, mv, dot, ger, add, unsqueeze, transpose, bmm
from torch import sum as torch_sum, norm

class LossFunction(object):

    def __init__(self, forward_model, ridge_coeff=None):
        self.forward_model = forward_model
        self.ridge_coeff = ridge_coeff
        self.gradient = self.gradient_default if ridge_coeff is None else self.gradient_ridge
        return

    def __call__(self, X, Y):
        N = X.size()[1]
        Yhat = self.forward_model(X)
        #
        #
        # print(Y[:, :5], Yhat[:,:5])
        return norm(self.forward_model(X) - Y, dim=0).sum()/float(N)

    def gradient_default(self, X, Y):
        N = X.size()[1]
        W_ext = unsqueeze(
                    self.forward_model.W,
                    0
                ).expand(N, -1, -1)
        w0_ext = unsqueeze(
                    self.forward_model.w0,
                    0
                ).expand(N, -1, -1)
        X_ext = transpose(
                    unsqueeze(X, 0),
                    0,
                    2
                )
        Y_ext = transpose(
                    unsqueeze(Y, 0),
                    0,
                    2
                )
        cuda.synchronize()
        return (
            torch_sum(
                bmm(
                    bmm(W_ext, X_ext) + w0_ext - Y_ext,
                    transpose(
                        X_ext,
                        1,
                        2
                    )
                ),
                dim=0
            ) * 2 / N, # W gradient
            unsqueeze(
                torch_sum(
                    self.forward_model(X) - Y,
                    dim=1
                ) * 2 / N,
                1
            ) # w0 gradient
        )

    def gradient_ridge(self, X, Y):
        N = X.size()[1]
        #
        # Compute ridge component of grad
        ridge_W = self.forward_model.W * 2 * self.ridge_coeff
        ridge_w0 = self.forward_model.w0 * 2 * self.ridge_coeff
        #
        # Compute default component of grad
        W, w0 = self.gradient_default(X, Y)
        #
        return (W + ridge_W, w0 + ridge_w0)
