import theano
from theano import tensor
from theano.gof.op import Op
import numpy

class Fibby(Op):
    __props__ = ()

    def make_node(self, x):
        x_ = tensor.as_tensor_variable(x)
        assert x_.ndim == 1
        return theano.Apply(self, inputs=[x_], outputs=[x_.type()])
        # using x_.type() is dangerous, it copies x's broadcasting behaviour

    def perform(self, node, inputs, output_storage):
        x, = inputs
        y = output_storage[0][0] = x.copy()
        for i in range(2, len(x)):
            y[i] = y[i-1] * y[i-2] + x[i]

    def c_code(self, node, name, inames, onames, sub):
        x, = inames
        y, = onames
        fail = sub['fail']
        return """
Py_XDECREF(%(y)s);
%(y)s = (PyArrayObject*)PyArray_FromArray(
            %(x)s, 0, NPY_ARRAY_ENSURECOPY);
if (!%(y)s)
  %(fail)s;
{//New scope needed to make compilation work
  dtype_%(y)s * y = (dtype_%(y)s*)PyArray_DATA(%(y)s);
  dtype_%(x)s * x = (dtype_%(x)s*)PyArray_DATA(%(x)s);
  for (int i = 2; i < PyArray_DIMS(%(x)s)[0]; ++i)
    y[i] = y[i-1]*y[i-2] + x[i];
}
        """ % locals()

    def c_code_cache_version(self):
        return (1,)


class PoolTensor(Op):
    __props__ = ('ds', 'ignore_border', 'st', 'padding', 'mode')

    def make_node(self, x):
        x_ = tensor.as_tensor_variable(x)
        assert x_.ndim > 4
        return theano.Apply(self, inputs=[x_], outputs=[x_.type()])
        # using x_.type() is dangerous, it copies x's broadcasting behaviour


    @staticmethod
    def out_shape(tensor_shape, ds,
                  n_pool_dim=3, ignore_border=False, st=None,
                  padding=(0, 0, 0)):
        """
        Return the shape of the output from this op, for input of given
        shape and flags.
        Parameters
        ----------
        tensor_shape : tuple, list, or similar of integer or scalar Theano variable
            The shape of a tensor of images. The last two elements are
            interpreted as the number of rows, and the number of cols.
        ds : list or tuple of two ints
            Downsample factor over rows and columns this parameter indicates
            the size of the pooling region.
        n_pool_dim : int
            The number of desired pooling dimension.
        st : list or tuple of two ints
            The stride size. This is the distance between the pooling regions.
            If it's set to None, it equals ds.
        ignore_border : bool
            If ds doesn't divide imgshape, do we include an extra row/col of
            partial downsampling (False) or ignore it (True).
        padding : tuple of two ints
            (pad_h, pad_w), pad zeros to extend beyond four borders
            of the images, pad_h is the size of the top and bottom margins,
            and pad_w is the size of the left and right margins.
        Returns
        -------
        list
            The shape of the output from this op, for input of given shape.
            This will have the same length as imgshape, but with last two
            elements reduced as per the downsampling & ignore_border flags.
        """
        if len(tensor_shape) < n_pool_dim:
            raise TypeError('tensor_shape must have at least %i elements ' %
                            int(n_pool_dim))

        if st is None:
            st = ds

        if any([n_pool_dim != n_dim
                for n_dim in [len(padding), len(ds), len(st)]]):
            raise ValueError('the length of padding, tensor_shape and'
                            ' downsample factor must be the same.')

        tensor_shape_const = [tensor.extract_constant(dim) for dim in
                              tensor_shape[-n_pool_dim:]]


        # add padding
        for i, dim_size in enumerate(tensor_shape_const):
            dim_size += padding[i] * n_pool_dim

        out_dims = []

        for i in range(n_pool_dim):
            if ignore_border:
                if ds[i] == st[i]:
                    out_dims.append(tensor_shape_const[i] // st[i])
                else:
                    out_dims.append((tensor_shape_const[i] - ds[i]) //
                                    st[i] + 1)
                    if isinstance(out_dims[i], theano.Variable):
                        out_dims[i] = tensor.maximum(out_dims[i], 0)
                    else:
                        out_dims[i] = numpy.maximum(out_dims[i], 0)

            else:
                if isinstance(tensor_shape_const[0], theano.Variable):
                    out_dims.append(tensor.switch(tensor.ge(st[i], ds[i]),
                                    (tensor_shape_const[i] - 1) // st[i] + 1,
                                    tensor.maximum(0, (r - 1 - ds[i]) //
                                                    st[i] + 1) + 1))
                elif st[i] >= ds[i]:
                    out_dims.append((tensor_shape_const[i] - 1) // st[i] + 1)
                else:
                    out_dims.append(
                        max(0, (tensor_shape_const[i] - 1 - ds[i] +
                                st[i]) // st[i]) + 1)

        rval = list(tensor_shape[:-n_pool_dim]) + out_dims
        return rval


    def perform(self, node, inp, out):
        x, = inp
        z, = out
        if len(x.shape) != 5:
            raise ValueError('Pool requires 5D input for now')

        z_shape = self.out_shape(x.shape, self.ds, 3,
                                 self.ignore_border, self.st,
                                 self.padding)

        if not self.ignore_border:
            assert z_shape[1] > 0
            assert z_shape[2] > 0
            assert z_shape[3] > 0

        if (z[0] is None) or (z[0].shape != z_shape):
            z[0] = numpy.empty(z_shape, dtype=x.dtype)

        zz = z[0]
        # number of pooling output rows
        pooling_tensor_shape = x.shape[-self.n_pool_dim:]
        # number of pooling output cols
        ds0, ds1, ds2 = self.ds
        st0, st1, st2 = self.st

        pad_h = self.padding[0]
        pad_w = self.padding[1]
        pad_d = self.padding[2]

        img_rows = x.shape[-3] + 3 * pad_h
        img_cols = x.shape[-2] + 3 * pad_w
        img_depth = x.shape[-1] + 3 * pad_w

        inc_pad = self.mode == 'average_inc_pad'

        if self.padding != (0, 0, 0):
            y = numpy.zeros(
                (x.shape[0], x.shape[1], img_rows, img_cols, img_depth),
                dtype=x.dtype)
            y[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w),
              pad_d:(img_cols - pad_d)] = x
        else:
            y = x

        func = numpy.max
        if self.mode == 'sum':
            func = numpy.sum
        elif self.mode != 'max':
            func = numpy.average

        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(pr):
                    row_st = r * st0
                    row_end = builtins.min(row_st + ds0, img_rows)

                    if not inc_pad:
                        row_st = builtins.max(row_st, self.padding[0])
                        row_end = builtins.min(row_end, x.shape[-3] + pad_h)

                    for c in xrange(pc):
                        col_st = c * st1
                        col_end = builtins.min(col_st + ds1, img_cols)
                        if not inc_pad:
                            col_st = builtins.max(col_st, self.padding[1])
                            col_end = builtins.min(col_end, x.shape[-2] + pad_w)

                        for d in xrange(pd):
                            dep_st = d * st2
                            dep_end = builtins.min(dep_st + ds2, img_dep)

                            if not inc_pad:
                                dep_st = builtins.max(dep_st, self.padding[2])
                                dep_end = builtins.min(dep_end,
                                                       x.shape[-3] + pad_d)

                            zz[n, k, r, c, d] = func(y[
                                n, k, row_st:row_end, col_st:col_end,
                                dep_st:dep_end])


    def c_code(self, node, name, inp, out, sub):
        if self.mode not in ('max', 'sum', 'average_exc_pad',
                             'average_inc_pad'):
            raise theano.gof.utils.MethodNotDefined()
        x, = inp
        z, = out
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pd0, pd1 = self.padding

        if self.openmp:
            omp_parallel = '#pragma omp parallel for private(r_st, r_end, c_st, c_end, collector) schedule(static)'
        else:
            omp_parallel = ''
        ccode = """
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int z_r, z_c; // shape of the output
        int r, c; // shape of the padded_input
        if(PyArray_NDIM(%(x)s)!=5)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 5d ndarray");
            %(fail)s;
        }
        r = PyArray_DIMS(%(x)s)[2];
        c = PyArray_DIMS(%(x)s)[3];
        d = PyArray_DIMS(%(x)s)[4];
        r += %(pd0)s * 3;
        c += %(pd1)s * 3;
        d += %(pd2)s * 3;
        if (%(pd0)s != 0 && %(pd1)s != 0 && %(pd2)s && !%(ignore_border)s)
            {
              PyErr_SetString(PyExc_ValueError,
                "padding must be (0, 0, 0) when ignore border is False");
              %(fail)s;
            }
        if (%(ignore_border)s)
        {
            // '/' in C is different from '/' in python
            if (r - %(ds0)s < 0)
            {
              z_r = 0;
            }
            else
            {
              z_r = (r - %(ds0)s) / %(st0)s + 1;
            }

            if (c - %(ds1)s < 0)
            {
              z_c = 0;
            }
            else
            {
              z_c = (c - %(ds1)s) / %(st1)s + 1;
            }

            if (d - %(ds2)s < 0)
            {
              z_d = 0;
            }
            else
            {
              z_d = (c - %(ds2)s) / %(st2)s + 1;
            }

        }
        else
        {
            // decide how many rows the output has
            if (%(st0)s >= %(ds0)s)
            {
                z_r = (r - 1) / %(st0)s + 1;
            }
            else
            {
                z_r = std::max(0, (r - 1 - %(ds0)s + %(st0)s) / %(st0)s) + 1;
            }
            // decide how many columns the output has
            if (%(st1)s >= %(ds1)s)
            {
                z_c = (c - 1) / %(st1)s + 1;
            }
            else
            {
                z_c = std::max(0, (c - 1 - %(ds1)s + %(st0)s) / %(st1)s) + 1;
            }
            assert(z_r > 0);
            assert(z_c > 0);
        }
        // memory allocation of z if necessary
        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x)s)[1])
          ||(PyArray_DIMS(%(z)s)[2] != z_r)
          ||(PyArray_DIMS(%(z)s)[3] != z_c)
          )
        {
          if (%(z)s) Py_XDECREF(%(z)s);
          npy_intp dims[4] = {0,0,0,0};
          dims[0]=PyArray_DIMS(%(x)s)[0];
          dims[1]=PyArray_DIMS(%(x)s)[1];
          dims[2]=z_r;
          dims[3]=z_c;
          //TODO: zeros not necessary
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
        }
        // used for indexing a pool region inside the input
        dtype_%(x)s collector; // temp var for the value in a region
        if (z_r && z_c)
        {
            int r_st, r_end, c_st, c_end;
            %(omp_parallel)s
            for(int t = 0; t < PyArray_DIMS(%(x)s)[0] * PyArray_DIMS(%(x)s)[1]; t++){
                int b = t %% PyArray_DIMS(%(x)s)[0];
                int k = t / PyArray_DIMS(%(x)s)[0];
                for(int i=0; i < z_r; i++){
                  r_st = i * %(st0)s;
                  r_end = r_st + %(ds0)s;
                  // skip the padding
                  r_st = r_st < %(pd0)s ? %(pd0)s : r_st;
                  r_end = r_end > (r - %(pd0)s) ? r - %(pd0)s : r_end;
                  // from padded_img space to img space
                  r_st -= %(pd0)s;
                  r_end -= %(pd0)s;
                  // handle the case where no padding, ignore border is True
                  if (%(ignore_border)s)
                  {
                    r_end = r_end > r ? r : r_end;
                  }
                  for(int j=0; j<z_c; j++){
                    c_st = j * %(st1)s;
                    c_end = c_st + %(ds1)s;
                    // skip the padding
                    c_st = c_st < %(pd1)s ? %(pd1)s : c_st;
                    c_end = c_end > (c - %(pd1)s) ? c - %(pd1)s : c_end;
                    dtype_%(z)s * z = (
                          (dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, b, k, i, j)));
                    // change coordinates from padding_img space into img space
                    c_st -= %(pd1)s;
                    c_end -= %(pd1)s;
                    // handle the case where no padding, ignore border is True
                    if (%(ignore_border)s)
                    {
                      c_end = c_end > c ? c : c_end;
                    }
        """
        if self.mode == 'max':
            ccode += """
                    // use the first element as the initial value of collector
                    collector = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,r_st,c_st)))[0];
                    // go through the pooled region in the unpadded input
                    for(int m=r_st; m<r_end; m++)
                    {
                      for(int n=c_st; n<c_end; n++)
                      {
                        dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,m,n)))[0];
                        collector = (a > collector) ? a : collector;
                      }
                    }
                    z[0] = collector;
            """
        elif self.mode in ('sum', 'average_exc_pad', 'average_inc_pad'):
            ccode += """
                    // initialize the sum at zero
                    collector = ((dtype_%(x)s)(0));
                    // go through the pooled region in the unpadded input
                    for(int m=r_st; m<r_end; m++)
                    {
                      for(int n=c_st; n<c_end; n++)
                      {
                        dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,m,n)))[0];
                        collector += a;
                      }
                    }
            """
            if self.mode == "sum":
                ccode += """
                    z[0] = collector;
                """
            elif self.mode == 'average_inc_pad' and self.ignore_border:
                ccode += """
                    z[0] = collector / (%(ds0)s * %(ds1)s);
                """
            else:
                ccode += """
                    z[0] = collector / ((r_end-r_st)*(c_end-c_st));
                """
        ccode += """
                  }
                }
              }
            }
        """
        pass

    def c_code_cache_version(self):
        return (0, 6, 8, 4, self.openmp)



if __name__ == '__main__':
    np = numpy
    from theano.tensor.signal import pool
    x = tensor.tensor4('x')
    pool_ = PoolTensor()
    pool_expect = pool.Pool
    print pool_expect.out_shape((12, 12, 12, 12, 12, 12), (2, 2))

    print pool_.out_shape((12, 12, 12, 12, 12, 12), (2, 2, 2),
                          padding=(2, 2, 2), n_pool_dim=3)
    #z = pool(x)
    #print z.eval({x: np.random.normal((12, 4, 10, 12))})
    tensor_shape = (100, 12, 23) # loop index
    current_idx = [0] * 3

    print nested_loop([12])
