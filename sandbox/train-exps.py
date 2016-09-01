from __future__ import print_function

import logging
import yaml

FORMAT = '[%(asctime)s] %(name)s %(message)s'
DATEFMT = "%M:-%D:-%S"

logger = logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

import os
import theano
import theano.tensor as T
import fuel
import numpy as np
import ipdb
import time
import cPickle as pickle

from theano import tensor

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam
from blocks.bricks import Tanh, Identity
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import PARAMETER
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model
# from blocks.extras import Plot

import datasets
from model.RAM import (RAM, GlimpseNetwork, LocationNetwork, ActionNetwork,
                   CoreNetwork)
from training_algorithm import REINFORCE

def main():
    dataset = 'mnist'
    batch_size = 100

    n_classes = 10

    learning_rate = 1e-3
    name = dataset

    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    # -----------------------------------------------------------------------
    image_size, channels, data_train, data_valid, data_test = \
        datasets.get_data(dataset)
    img_height, img_width = image_size

    train_stream = Flatten(DataStream.default_stream(
        data_train, iteration_scheme=SequentialScheme(
            data_train.num_examples, batch_size)))

    valid_stream = Flatten(DataStream.default_stream(
        data_valid, iteration_scheme=SequentialScheme(
            data_valid.num_examples, batch_size)))

    test_stream  = Flatten(DataStream.default_stream(
        data_test,  iteration_scheme=SequentialScheme(
            data_test.num_examples, batch_size)))

    def lr_tag(value):
        """ Convert a float into a short tag-usable string representation. E.g.:
            0.1   -> 11
            0.01  -> 12
            0.001 -> 13
            0.005 -> 53
        """
        exp = np.floor(np.log10(value))
        leading = ("%e"%value)[0]
        return "%s%d" % (leading, -exp)

    lr_str = lr_tag(learning_rate)

    print("     dataset: %s" % dataset)
    # --------------- Building Model ----------------

    glim_net = GlimpseNetwork(dim=50, loc_emb=50,
                              n_channels=channels, img_height=img_height,
                              img_width=img_width, N=40, name='glimpse_net',
                              **inits) # output (n)

    core = CoreNetwork(input_dim=50, dim=50, name='core_net',
                       **inits)

    loc_net = LocationNetwork(input_dim=50, loc_emb=50,
                              sensor=glim_net.sensor, name='loc_net', **inits)

    action = ActionNetwork(input_dim=50, n_classes=n_classes,
                           name='action',**inits)

    ram = RAM(core, glim_net, loc_net, action, n_steps=10, name='RAM', **inits)
    ram.initialize()
    # -------------------------------------------------------------

    img = T.matrix('features')
    y = T.imatrix('targets')
    y = y.flatten()

    ls, actions, _, _ = ram.out(img)

    # get loc network param
    # ---------------- Building Reinforce alforithm ----------------
    loc_params = reduce(lambda x, y: x+y,
                        [list(net.parameters) for net in
                         list(loc_net.children) + list(glim_net.children)])

    others_bricks = list(core.children) + list(action.children)

    reinforce = REINFORCE()

    cost_re, reward, baseline = \
        reinforce.build_reinforce_cost_reward_base(y, actions)

    loc_grad = reinforce.build_reinforce_grad(cost_re, loc_params, reward,
                                              baseline)

    y_dis = actions[-1]
    cost_true = CategoricalCrossEntropy().apply(y, y_dis)
    cg = ComputationGraph([cost_true])

    # filter out initial_state
    others_params = VariableFilter(roles=[PARAMETER],
                                   bricks=others_bricks)(cg.variables)

    other_grad = T.grad(cost_true, others_params)
    # Hybrid Cost
    all_grad = loc_grad + other_grad
    all_params = loc_params + others_params
    gradients = {param: grad for param, grad in zip(all_params, all_grad)}

    algorithm = GradientDescent(
        cost=cost_true,
        gradients=gradients,
        parameters=all_params,
        step_rule=CompositeRule([
            StepClipping(10.),
            Adam(learning_rate),
        ])
    )
    # ------------------------------------------------------------------------
    # Setup monitors
    cost_true.name = 'cost_true'
    cost_re.name = 'reinforce cost'
    # avg_action = actions.mean(1)
    # avg_action.name = 'avg_action'
    acc = T.cast(T.eq(y, T.argmax(y_dis, axis=1)), 'float32').mean()
    acc.name = 'accuratcy'
    monitors = [cost_true, cost_re, acc]
    train_monitors = monitors
    # Live plotting...
    #plot_channels = [
    #    ["cost_true", "reinforce_cost"]
    #]

    # ------------------------------------------------------------

    #plotting_extensions = [
    #        Plot(name, channels=plot_channels)
    #    ]
    main_loop = MainLoop(
        model=Model(cost_true),
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            Timing(),
            FinishAfter(after_n_epochs=5),
            TrainingDataMonitoring(
                train_monitors,
                prefix="train",
                after_epoch=True),
            DataStreamMonitoring(
                monitors,
                test_stream,
                prefix="test"),
            ProgressBar(),
            Printing()])

    main_loop.run()


if __name__ == '__main__':
    main()
