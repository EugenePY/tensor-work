"""
REINFORCE algorithms for Expected log Likelihood and log Likelihood
Deriviation : refer to REINFORCE.pdf
"""

import theano.tensor as T
import numpy as np
from blocks.bricks.base import Brick, application
from cost import NeLogLikelihood
# from blocks.algorithms import GradientDescent


class REINFORCE(Brick):
    """
    REINFORCE algorithm : Episodic trial
        -- Version: ExpectedLikelihood version
    -------------------

    Parameters
    ----------
    base_line : int
    y_dis : (n_steps, batch_size, n_classes)
    y     : (batch_size,)

    Return
    --
    theano.op.gradient
    """
    def __init__(self, **kwargs):
        super(REINFORCE, self).__init__(**kwargs)
        self.cost_brick = NeLogLikelihood()
        self.children = [self.cost_brick]

    @application
    def reward(self, y_, y_dis_, n_steps, batch_size):
        y_pred = T.argmax(y_dis_, axis=1)[:, None]
        # reward = T.cast(T.eq(y_pred, y_), 'float32')
        reward = T.cast(T.eq(y_pred, y_).reshape((n_steps, batch_size)),
                        'float32')
        # reward = T.repeat(reward[-1][None, :], n_steps, axis=0)
        return reward

    def _pre_process(self, y, y_dis):
        y_ = T.repeat(y[None, :], y_dis.shape[0], axis=0).reshape(
            (y_dis.shape[0]*y_dis.shape[1], 1))
        y_dis_ = y_dis.reshape((y_dis.shape[0]*y_dis.shape[1], y_dis.shape[2]))
        return y_, y_dis_

    @application
    def baseline(self, reward, y_dis, y_dis_, n_steps, batch_size):
        # idx = T.repeat(T.argmax(y_dis[-1], axis=1), n_steps, axis=0)
        # base = reward * \
        #     y_dis_[T.arange(batch_size*n_steps),
        #            idx].reshape((n_steps, batch_size))
        base = reward * T.max(y_dis_, axis=1).reshape((n_steps, batch_size))
        # base = reward.mean(1)
        # base = base.mean(1)
        # base = T.extra_ops.cumsum(base[::-1], axis=0)[::-1].mean(1)
        base = base.mean(1)[:, None]
        return base

    def cost_matrix(self, y, y_dis):
        return self.cost_brick.cost_matrix(y, y_dis)

    def _logmulti_normal_pdf(self, mean, samples, var):
        norm2d_var = (-0.5*((samples - mean)/var)**2).sum(2)
        return norm2d_var

    def _gaussian_pdf(self, mean, samples, var):
        return T.prod(T.exp(((samples-mean)/var)**2 / 2) /
                      (T.sqrt(2 * np.pi) * var), axis=2)

    def expected_reward(self, reward):
        # reward_cum = T.extra_ops.cumsum(reward[::-1], axis=0)[::-1]
        return reward

    @application(inputs=['y', 'y_dis'], outputs=['reinforce_cost',
                                                 'reward_cum', 'baseline'])
    def build_reinforce_cost_reward_base(self, y, y_dis):
        n_steps, batch_size, n_classes = y_dis.shape
        y_, y_dis_ = self._pre_process(y, y_dis)
        # reward = self.reward(y_, y_dis_, n_steps, batch_size)
        reward = self.reward(y_, y_dis_, n_steps, batch_size)
        # n_steps, batch_size
        reward_cum = self.expected_reward(reward)
        base = self.baseline(reward, y_dis, y_dis_, n_steps, batch_size)
        cost_re = (self.cost_matrix(y, y_dis)).sum(0).mean()
        return cost_re, reward_cum, base

    def build_reinforce_grad(self, cost_re, l_u, l_sample,
                             wrt, reward_cum, base, std):
        """
        REINFORCE with normal distribution
        """

        # grads = T.extra_ops.cumsum((l_sample - l_u)/(std**2), axis=0) * \
        #     (reward_cum - base)[:, :, None]  # backprop-through-time
        # base = reward_cum * self._gaussian_pdf(l_u, l_sample, std)
        # base = base.mean(1)[:, None]

        cost_re = -self._logmulti_normal_pdf(l_u, l_sample, std)
        cost_re = (cost_re * (reward_cum - base)).sum(0).mean()
        return T.grad(cost_re, wrt,
                      consider_constant=[reward_cum, base, l_sample])


if __name__ == '__main__':
    from blocks.bricks.simple import Softmax, Linear
    from blocks.initialization import Constant, IsotropicGaussian
    from blocks.graph import ComputationGraph
    from blocks.roles import PARAMETER
    from blocks.filter import VariableFilter
    from model.RAM import CoreNetwork

    x = T.tensor3('x')
    y = T.ivector('y')
    n_steps = 4
    batch_size = 5
    input_dim = 6
    output_dim = 8
    n_classes = 10
    test_data = {x: np.random.normal(size=(n_steps, batch_size, input_dim)
                                     ).astype(np.float32),
                 y: np.random.randint(n_classes, size=(batch_size, )
                                      ).astype(np.int32)}
    inits = {
        'weights_init': IsotropicGaussian(0.1),
        'biases_init': Constant(0.),
    }

    core = CoreNetwork(input_dim=input_dim, dim=output_dim, **inits)
    core.initialize()
    proj = Linear(input_dim=output_dim*2, output_dim=n_classes, **inits)
    proj.initialize()
    out = Softmax()

    state, cell = core.apply(x)

    a = T.concatenate([state, cell], axis=2)
    a = a.reshape((a.shape[0]*a.shape[1], a.shape[2]))

    a = proj.apply(a)
    prop = out.apply(a).reshape((n_steps, batch_size, n_classes))
    pred = prop[-1]
    prop = prop.reshape((n_steps * batch_size, n_classes))

    print prop.eval({x: test_data[x]})
    y_reat = T.repeat(y[None, :], n_steps, axis=0).reshape(
        (n_steps * batch_size, 1))
    print y_reat.eval({y: test_data[y]})

    cost = T.nnet.categorical_crossentropy(pred, y).mean()
    print cost.eval(test_data)
    cost_re = cost.reshape((n_steps, batch_size))
    reward = T.cast(T.eq(T.argmax(prop, axis=1)[:, None],
                         y_reat).reshape((n_steps, batch_size)), 'float32')
    print reward.eval(test_data)
    base = (reward * T.max(prop, axis=1).reshape((n_steps, batch_size))).sum(0)
    base = base[None, :]
    print base.eval(test_data)
    reward_cum = T.extra_ops.cumsum(reward[::-1], axis=0)[::-1]
    print reward_cum.eval(test_data)
    cost_new = (cost_re * (reward_cum - base)).sum(0).mean()
    # print cost_new.eval(test_data)
    cg = ComputationGraph(cost_new)
    params = VariableFilter(roles=[PARAMETER])(cg.variables)
    grad = T.grad(cost_new, params, consider_constant=[reward_cum, base])
    # fn = theano.function(inputs=[x, y], outputs=grad)
    # print fn(test_data[x], test_data[y])

    a = REINFORCE()
    o = a.build_reinforce_cost_reward_base(
        y, prop.reshape((n_steps, batch_size, n_classes)))
    print [i.eval(test_data) for i in o]
