import theano
import theano.tensor as T
import numpy as np

import pylab
from PIL import Image
import theano.tensor.nnet.neighbours
from model.RAM import GlimpseSensorBeta


N = 80
channels = 3
height = 480
width = 640
batch_size = 10
# test images2neibs
# ------------------------------------------------------------------------
att = GlimpseSensorBeta(channels=channels, img_height=height,
                        img_width=width, N=N)
print att.total_step_y, att.total_step_x

I_ = T.tensor4('img')
center_y_ = T.vector()
center_x_ = T.vector()

"""
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
W_ = att.read(I_, center_y_, center_x_)

do_read = theano.function(inputs=[I_, center_x_, center_y_],
                          outputs=W_, allow_input_downcast=True)

# ------------------------------------------------------------------------

I = Image.open("model/test/cat.jpg")
I = I.resize((640, 480))  # .convert('L')
I = np.asarray(I).transpose([2, 0, 1])

I = I / 255.

center_y = 1.
center_x = 1.


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
pylab.imshow(imagify(W, N, N), interpolation='nearest')

pylab.show(block=True)
