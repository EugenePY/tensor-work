#!/usr/bin/env python

from __future__ import print_function, division

import logging
import theano
import theano.tensor as T
from blocks.model import Model
from blocks.graph import ComputationGraph
import numpy as np
import os
from skimage.draw import line

from PIL import Image

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

ROWS = 10
COLS = 20


def scale_norm(arr):
    arr = arr - arr.min()
    scale = (arr.max() - arr.min())
    return arr / scale


# these aren't paramed yet in a generic way, but these values work
def img_ratangle(shape):
    row, col = shape
    row = row - 1
    col = col - 1

    comb = [(0, 0, row, 0),
            (0, 0, 0, col),
            (row, 0, row, col),
            (0, col, row, col)]
    row_idx = []
    col_idx = []
    for i in comb:
        a, b = line(*i)
        row_idx.extend(a.tolist())
        col_idx.extend(b.tolist())

    return row_idx, col_idx


def img_grid(arr, global_scale=True):
    N, channels, height, width = arr.shape

    global ROWS, COLS
    rows = ROWS
    cols = COLS

    total_height = rows * height + 9
    total_width = cols * width + 19

    if global_scale:
        arr = scale_norm(arr)

    I = np.zeros((channels, total_height, total_width))
    I.fill(1)

    for i in xrange(N):
        r = i // cols
        c = i % cols

        if global_scale:
            this = arr[i]
        else:
            this = scale_norm(arr[i])

        offset_y, offset_x = r*height+r, c*width+c
        I[0:channels, offset_y:(offset_y+height),
          offset_x:(offset_x+width)] = this

    I = (255*I).astype(np.uint8)
    if(channels == 1):
        out = I.reshape((total_height, total_width))
    else:
        out = np.dstack(I).astype(np.uint8)
    return Image.fromarray(out)


def generate_samples(p, batch, subdir, output_size, channels):
    if isinstance(p, Model):
        model = p
    else:
        print("Don't know how to handle unpickled %s" % type(p))

    ram = model.get_top_bricks()[0]
    sensor = ram.glimpes_network.sensor

    # ------------------------------------------------------------
    logging.info("Compiling sample function...")
    img = T.matrix('img')
    img_loc, a, _, _, _ = ram.out(img)
    img_loc = T.clip(img_loc, -1., 1.)

    img_loc_cg = ComputationGraph([img_loc])
    updates = img_loc_cg.updates

    img_loc, a = img_loc[:-1], a[1:]

    n_steps = img_loc.shape[0]
    batch_size = img.shape[0]

    # flattening the outputs into (batch_size * n_step, dim)
    img_loc_flat = img_loc.reshape((n_steps * batch_size, sensor.emb_dim))
    img_flat = T.repeat(img, n_steps, axis=0)
    action_flat = a.reshape((n_steps * batch_size, a.shape[-1]))

    action_flat = T.argmax(action_flat, axis=1)  # this is for coloring

    img_sampling = sensor.att_mark(img_flat, img_loc_flat[:, 0],
                                   img_loc_flat[:, 1])

    do_sample = theano.function([img], outputs=img_sampling,
                                updates=updates,
                                allow_input_downcast=True)

    # ------------------------------------------------------------
    logging.info("Sampling and saving images...")
    samples = do_sample(batch)

    n_steps_non_symbolic = ram.n_steps - 1 # we have drop one-step
    img_width = sensor.img_width
    img_height = sensor.img_height
    channels = 3  # RGB imgae
    batch_size_non_symbolic = samples.shape[0] // n_steps_non_symbolic

    samples = samples.reshape((batch_size_non_symbolic, n_steps_non_symbolic,
                               channels, img_height, img_width))

    if(n_steps_non_symbolic > 0):
        img = img_grid(samples[:, n_steps_non_symbolic-1, :, :])
        img.save("{0}/sample.png".format(subdir))

    for i in xrange(n_steps_non_symbolic):
        img = img_grid(samples[:, i, :, :])
        img.save("{0}/time-{1:03d}.png".format(subdir, i))

    os.system("convert -delay 5 {0}/time-*.png -delay 300 "
              "{0}/sample.png {0}/sequence.gif".format(subdir))


if __name__ == '__main__':
    from model.RAM import GlimpseSensorBeta
    batch_size = 10
    img_width = 28
    img_height = 28
    channels = 1
    N = 4
    img = T.matrix('img')
    action = T.vector('action')
    center_x = T.vector('x')
    center_y = T.vector('y')

    img_test = {img: np.random.randn(
        batch_size, channels * img_height * img_width).astype('float32'),
        center_x: np.random.uniform(high=1, low=-1,
                                    size=(batch_size,)).astype('float32'),
        center_y: np.random.uniform(high=1, low=-1,
                                    size=(batch_size,)).astype('float32'),
        action: np.random.uniform(high=1, low=-1,
                                  size=(batch_size,)).astype('float32')
    }

    sensor = GlimpseSensorBeta(img_width=img_width, img_height=img_height,
                               channels=channels, N=N)
    img = sensor.att_mark(img, action, center_x, center_y)
    print(img.eval(img_test))
