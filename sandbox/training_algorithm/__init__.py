"""
REINFORCE algorithms for Expected log Likelihood and log Likelihood
Deriviation : refer to REINFORCE.pdf
"""

import theano
import theano.tensor as T
from blocks.bricks.base import Brick, application
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
    @application
    def reward(self, y_, y_dis_, n_steps, batch_size):
        y_pred = T.argmax(y_dis_, axis=1)[:, None]
        reward = T.cast(T.eq(y_pred, y_).reshape((n_steps, batch_size)),
                        'float32')
        return reward

    def _pre_process(self, y, y_dis):
        y_ = T.repeat(y, y_dis.shape[0], axis=0)[:, None]
        y_dis_ = y_dis.reshape((y_dis.shape[0]*y_dis.shape[1], y_dis.shape[2]))
        return y_, y_dis_

    @application
    def baseline(self, reward, y_dis_, n_steps, batch_size):
        base = (reward * T.max(y_dis_, axis=1).reshape((n_steps, batch_size))
                ).sum(0)
        base = base[None, :]
        return base

    def cost_matrix(self, y_, y_dis_, n_steps, batch_size):
        cost_m = T.nnet.categorical_crossentropy(y_dis_, y_)
        return cost_m.reshape((n_steps, batch_size))

    def expected_reward(self, reward):
        reward_cum = T.extra_ops.cumsum(reward[::-1, :], axis=0)[::-1, :]
        return reward_cum

    @application(inputs=['y', 'y_dis'], outputs=['reinforce_cost',
                                                 'reward_cum', 'baseline'])
    def build_reinforce_cost_reward_base(self, y, y_dis):
        n_steps, batch_size, n_classes = y_dis.shape
        y_, y_dis_ = self._pre_process(y, y_dis)
        cost_m = self.cost_matrix(y_, y_dis_, n_steps, batch_size)
        reward = self.reward(y_, y_dis_, n_steps, batch_size)  # n_steps, batch_size
        base = self.baseline(reward, y_dis_, n_steps, batch_size)
        reward_cum = self.expected_reward(reward)
        return (cost_m * (reward_cum - base)).sum(0).mean(), reward_cum, base

    def build_reinforce_grad(self, reinforce_cost, wrt, reward, baseline):
        return T.grad(reinforce_cost, wrt,
                      consider_constant=[reward, baseline])


if __name__ == '__main__':
    import numpy as np
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
    prop = out.apply(a)
    # print prop.eval({x: test_data[x]})
    y_reat = T.repeat(y, n_steps, axis=0)[:, None]
    # print y_reat.eval({y: test_data[y]})
    cost = T.nnet.categorical_crossentropy(prop, y_reat)
    # print cost.eval(test_data)
    cost_re = cost.reshape((n_steps, batch_size))
    reward = T.cast(T.eq(T.argmax(prop, axis=1)[:, None],
                         y_reat).reshape((n_steps, batch_size)), 'float32')
    base = (reward * T.max(prop, axis=1).reshape((n_steps, batch_size))).sum(0)
    base = base[None, :]
    print base.eval(test_data)
    reward_cum = T.extra_ops.cumsum(reward[::-1, :], axis=0)[::-1, :]
    print reward_cum.eval(test_data)
    cost_new = (cost_re * (reward_cum - base)).sum(0).mean()
    cg = ComputationGraph(cost_new)
    params = VariableFilter(roles=[PARAMETER])(cg.variables)
    grad = T.grad(cost_new, params, consider_constant=[reward_cum, base])
    #fn = theano.function(inputs=[x, y], outputs=grad)
    #print fn(test_data[x], test_data[y])

    print prop
    a = REINFORCE()
    o = a.build_reinforce_cost_reward_base(y, prop.reshape((n_steps, batch_size, n_classes)))
    print [i.eval(test_data) for i in o]
