from model.attention import Attention
from model.highway import LinearAttention
from util.tests import Im2LatexTest
from space import ContextSpace


@Im2LatexTest.call_test
def test_attention():
    input_space = ContextSpace(dim=14, num_annotation=2)
    state_below = input_space.make_theano_batch()

    atten = Attention(input_space=input_space,
        layers=[LinearAttention(dim=12, layer_name='proj',
                                irange=0)])
    atten.set_input_space(input_space)
    for i in range(len(atten.layers)):
        print atten.layers[i], atten.layers[i].get_input_space()
    pctx = atten.project(state_below)
    atten.alpha_dis(pctx)

