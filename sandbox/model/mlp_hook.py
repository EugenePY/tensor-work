"""
Letting Layers support ContextSpace input
"""
from pylearn2.utils.track_version import MetaLibVersion
from pylearn2.utils import wraps


class AttentionWrapper(MetaLibVersion):
    pass

if __name__ == "__main__":
    # simple test it seems ok....
    from pylearn2.models.mlp import Softmax
    from pylearn2.models.mlp import MLP
    from space import ContextSpace
    a = Softmax(layer_name='asd', n_classes=10, irange=.2)
    mlp = MLP(input_space=ContextSpace(dim=10, num_annotation=102),
              layers=[a], seed=123)
    print mlp.layers[0].needs_reformat
