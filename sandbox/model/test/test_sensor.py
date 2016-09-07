import theano
import theano.tensor as T
import numpy as np

import pylab
from PIL import Image
import theano.tensor.nnet.neighbours
from model.RAM import GlimpseSensorBeta, RetinaGlimpse

#theano.config.optimizer='fast_compile'
# theano.config.exception_verbosity='high'
#theano.config.compute_test_value = 'warn'

r = 20
channels = 3
height = 480
width = 640
batch_size = 100

sensor = RetinaGlimpse(img_width=width, img_height=height,
                       channels=channels, radius=r, n_retina=5)
img = T.tensor4('features')
center_x = T.vector('x')
center_y = T.vector('y')

img.tag.test_value = np.random.normal(size=(batch_size, channels * height *
                                      width)).astype('float32')

center_x.tag.test_value = np.random.uniform(low=-1., high=1.,
    size=(batch_size,)).astype('float32')

center_y.tag.test_value = np.random.uniform(low=-1., high=1.,
    size=(batch_size,)).astype('float32')

test = {var: var.tag.test_value for var in [img, center_x, center_y]}
# test graph


# a = sensor.float2center_pixel(center_x, center_y)
# expect the retina to generate a (batch, channel)
# a = sensor.read(img, center_x, center_y)
# print a.eval(test).shape
# plot retina
"""
# test images2neibs
# ------------------------------------------------------------------------
att = GlimpseSensorBeta(channels=channels, img_height=height,
                        img_width=width, N=N)
print att.total_step_y, att.total_step_x

I_ = T.tensor4('img')
center_y_ = T.vector()
center_x_ = T.vector()

# ------------------------------------------------------------------------
test = {img: np.random.normal(size=(batch_size, channels * height *
                                    width)).astype('float32'),
        center_x_: np.random.rand(batch_size,).astype('float32'),
        center_y_: np.random.rand(batch_size,).astype('float32')}
idx = att.map_float_to_index(center_x_, center_y_)
batch_size = img.shape[0]
img_re = img.reshape((batch_size, att.channels, att.img_height,
                    att.img_width))
neighbours = T.nnet.neighbours.images2neibs(img_re, (att.N, att.N),
                                            att.stride_size)
a = att.read(img, center_x_, center_y_)
#print neighbours.eval({img: test[img]}).shape
print a.eval(test).shape


"""
# ------------------------------------------------------------------------
W_ = sensor.read(img, center_y, center_x)

do_read = theano.function(inputs=[img, center_x, center_y],
                          outputs=W_, allow_input_downcast=True)

# ------------------------------------------------------------------------
I = Image.open("model/test/cat.jpg")
I = I.resize((640, 480))  # .convert('L')
I = np.asarray(I).transpose([2, 0, 1])

I = I / 255.

center_y = 0.
center_x = 0.


def vectorize(*args):
    return [a.reshape((1,)+a.shape) for a in args]

I, center_y, center_x = \
    vectorize(I, np.array(center_x), np.array(center_y))

W = do_read(I, center_x, center_y)


def imagify(flat_image, h, w):
    image = flat_image.reshape([channels, h, w])
    image = image.transpose([1, 2, 0])
    return image / image.max()


pylab.figure()
pylab.gray()
pylab.imshow(imagify(I, height, width), interpolation='nearest')

pylab.figure()
pylab.gray()
pylab.imshow(imagify(W[:, :channels], 2*r, 2*r), interpolation='nearest')

pylab.figure()
pylab.gray()
pylab.imshow(imagify(W[:,channels:channels*2], 2*r, 2*r), interpolation='nearest')

pylab.show(block=True)

