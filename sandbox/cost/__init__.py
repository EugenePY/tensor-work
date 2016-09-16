from theano import tensor as T
from blocks.bricks.base import application
from blocks.bricks.cost import CostMatrix


class NeLogLikelihood(CostMatrix):
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
        n_steps, batch_size, n_classes = y_hat.shape
        y = T.repeat(y[None, :], n_steps, axis=0).flatten()
        y_hat_ = y_hat.reshape([n_steps*batch_size, n_classes])
        return -T.log(y_hat_[T.arange(n_steps*batch_size), y]).reshape(
            [n_steps, batch_size])

    @application
    def apply(self, y, y_hat):
        return self.cost_matrix(y, y_hat).sum(0).mean()


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
            y_hat: np.random.uniform(size=(n_steps, batch_size, n_classes)
                                     ).astype('float32')}
    rval = NeLogLikelihood().cost_matrix(y, y_hat)
    print rval.eval(test)
    assert rval.eval(test).shape == (batch_size * n_steps, 1)
    # assert np.allclose(rval.eval(test)
    rval = NeLogLikelihood().apply(y, y_hat)
    reval = rval.eval(test)
    print reval
    assert reval.shape == ()  # scalar
