from theano import tensor as T
from blocks.bricks.base import application
from blocks.bricks.cost import CostMatrix


class ExpectedLikelihood(CostMatrix):
    """Multi-classes Classification:

    - log_{Likelihood}
    = - E_post(Likelihood) - KL divergence(Entropy)
    -----------------
    Params
    ======
        y_hat : (n_steps, batch_size, n_classes)
        y : (batch_size, answer)
    """
    @application
    def cost_matrix(self, y, y_hat):
        y_hat_shape = y_hat.shape
        y = T.repeat(y, y_hat_shape[0], axis=0)
        y_hat_flat = y_hat.flatten()
        log_p = -T.log(y_hat_flat[T.arange(y.shape[0]) * y_hat_shape[0] + y] +
                       1e-5)
        return log_p.reshape((y.shape[0], 1))


class KLdivergence(CostMatrix):
    @application
    def cost_matrix(self, y, y_hat):
        pass

if __name__ == '__main__':
    import numpy as np
    y = T.ivector('y')
    y_hat = T.tensor3('hat')
    n_classes = 10
    batch_size = 11
    n_steps = 9
    test = {y: np.random.randint(n_classes, size=(batch_size,)
                                 ).astype(np.int32),
            y_hat: np.random.rand(n_steps, batch_size, n_classes
                                  ).astype('float32')}
    rval = ExpectedLikelihood().cost_matrix(y, y_hat)
    assert rval.eval(test).shape == (batch_size * n_steps, 1)
    # assert np.allclose(rval.eval(test)
    rval = ExpectedLikelihood().apply(y, y_hat)
    reval = rval.eval(test)
    assert reval.shape == ()  # scalar
