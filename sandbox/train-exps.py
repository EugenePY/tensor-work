from __future__ import print_function

import logging
import sys
import os
import time

from argparse import ArgumentParser
import cPickle as pickle

import theano.tensor as T

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

from blocks.algorithms import (GradientDescent, CompositeRule,
                               Adam, StepClipping)
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.initialization import Constant, IsotropicGaussian
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import PARAMETER
from blocks.extensions import (FinishAfter, Timing, Printing, ProgressBar)
from util import EarlyStopping
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
from blocks.model import Model
# from blocks.extras import Plot

import datasets
from model.RAM import (RAM, GlimpseNetwork, LocationNetwork, ActionNetwork,
                       CoreNetwork)
from checkpoints import PartsOnlyCheckpoint, SampleAttentionCheckpoint
from training_algorithm import REINFORCE

FORMAT = '[%(asctime)s] %(name)s %(message)s'
DATEFMT = "%M:%D:%S"

logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)
sys.setrecursionlimit(100000)


def main():
    dataset = 'mnist'
    batch_size = 100

    n_classes = 10

    learning_rate = 1e-3
    name = dataset
    oldmodel = None

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

    test_stream = Flatten(DataStream.default_stream(
        data_test,  iteration_scheme=SequentialScheme(
            data_test.num_examples, batch_size)))

    logging.info("experiment dataset: %s" % dataset)

    # --------------- Building Model ----------------

    glim_net = GlimpseNetwork(dim=128,
                              n_channels=channels, img_height=img_height,
                              img_width=img_width, N=7, name='glimpse_net',
                              **inits)  # output (n)

    core = CoreNetwork(input_dim=128, dim=256, name='core_net',
                       **inits)

    loc_net = LocationNetwork(input_dim=256, loc_emb=2, std=0.18,
                              non_hetro=False, name='loc_net', **inits)

    action = ActionNetwork(input_dim=256, n_classes=n_classes,
                           name='action', **inits)

    ram = RAM(core=core, glimpes_network=glim_net,
              location_network=loc_net, action_network=action,
              n_steps=8, name='RAM', random_init_loc=False, **inits)
    ram.initialize()
    # -------------------------------------------------------------

    img = T.matrix('features')
    y = T.imatrix('targets')
    y = y.flatten()

    loc_sample, actions, _, _, loc_u = ram.out(img)

    loc_sample = loc_sample[:-1]
    loc_u = loc_u[:-1]

    # classification task
    actions = T.repeat(actions[1:][-1][None, :, :], loc_sample.shape[0], axis=0)
    # get loc network param
    # ---------------- Building Reinforce alforithm ----------------
    loc_bricks = list(loc_net.children)  # + list(glim_net.children)

    others_bricks = list(core.children) + list(action.children) + \
        list(glim_net.children)

    reinforce = REINFORCE()

    cost_re, reward, baseline = \
        reinforce.build_reinforce_cost_reward_base(y, actions)

    cg_rein = ComputationGraph([cost_re])
    loc_params = VariableFilter(roles=[PARAMETER],
                                bricks=loc_bricks)(cg_rein.variables)

    loc_grad = reinforce.build_reinforce_grad(cost_re, loc_u,
                                              loc_sample, loc_params, reward,
                                              baseline, loc_net.std)

    y_dis = actions[-1]
    cost_true = CategoricalCrossEntropy().apply(y, y_dis)
    # cost_true = cost_re
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
            Adam(learning_rate)]),
    )
    algorithm.add_updates(cg_rein.updates)
    # ------------------------------------------------------------------------
    # Setup monitors
    cost_true.name = 'cost_true'
    cost_re.name = 'reinforce cost'
    # avg_action = actions.mean(1)
    # avg_action.name = 'avg_action'
    acc = T.cast(T.eq(y, T.argmax(y_dis, axis=1)), 'float32').mean()
    acc.name = 'accuratcy'
    monitors = [cost_re, acc, cost_true]
    train_monitors = monitors

    monitor_img = data_train.data_sources[0][:10*20].reshape((200, -1))
    # Live plotting...
    # plot_channels = [
    #    ["cost_true", "reinforce_cost"]
    # ]

    # ------------------------------------------------------------

    # plotting_extensions = [
    #        Plot(name, channels=plot_channels)
    # ]
    subdir = './exp/' + name + "-" + time.strftime("%Y%m%d-%H%M%S")

    if not os.path.exists(subdir):
        os.makedirs(subdir)

    main_loop = MainLoop(
        model=Model(cost_true),
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            Timing(),
            EarlyStopping('test_cost_true', iterations=100),
            FinishAfter(after_n_epochs=500),
            TrainingDataMonitoring(
                train_monitors,
                prefix="train",
                after_epoch=True),
            DataStreamMonitoring(
                monitors,
                test_stream,
                updates=cg_rein.updates,
                prefix="test"),
            ProgressBar(),
            Printing(),
            SampleAttentionCheckpoint(monitor_img=monitor_img,
                                      image_size=image_size[0],
                                      channels=channels,
                                      save_subdir='{}'.format(subdir),
                                      before_training=True,
                                      after_epoch=True),
            PartsOnlyCheckpoint("{}/{}".format(subdir, name),
                                before_training=True, after_epoch=True,
                                save_separately=['log', 'model'])])

    if oldmodel is not None:
        print("Initializing parameters with old model %s" % oldmodel)
        with open(oldmodel, "rb") as f:
            oldmodel = pickle.load(f)
            main_loop.model.set_parameter_values(
                oldmodel.get_parameter_values())
        del oldmodel

    main_loop.run()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--mode", "--mode",
                        default="debuging",
                        help="Simple model to test the REINFORCE algorithm")
    parser.add_argument("--name", type=str,
                        dest="name", default=None,
                        help="Name for this experiment")
    parser.add_argument("--dataset", type=str, dest="dataset",
                        default="mnist",
                        help="Dataset to use: [transform_mnist|mnist]")
    parser.add_argument("--epochs", type=int, dest="epochs",
                        default=100,
                        help="Number of training epochs to do")
    parser.add_argument("--bs", "--batch-size", type=int,
                        dest="batch_size",
                        default=100, help="Size of each mini-batch")
    parser.add_argument("--oldmodel", type=str,
                        help="Use a model pkl file created by a previous run"
                        " as a starting point for all parameters")
    args = parser.parse_args()

    main()
