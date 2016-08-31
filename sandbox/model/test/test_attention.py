from model.attention import Attention
from model.highway import LinearAttention
from util.tests import Im2LatexTest
from space import ContextSpace
from pylearn2.space import CompositeSpace, VectorSpace

@Im2LatexTest.call_test
def test_attention():
    input_space = CompositeSpace([ContextSpace(dim=14, num_annotation=2),
                                  VectorSpace(dim=12)])
    state_below = input_space.make_theano_batch()

    atten = ConditionLSTM(
        layer_name='cond_lstm',
        attention=Attention(layer_name='attention',
                            layers=[LinearAttention(layer_name='proj',
                                                    irange=2.)]))
    atten.set_input_space(input_space)
    # for i in range(len(atten.layers)):
    #    print atten.layers[i], atten.layers[i].get_input_space()
    # pctx = atten.project(state_below)
    # atten.alpha_dis(pctx)



def attention():
    sensor_layer = GlimpsSensor('sensor', dim=3, sensor_type='retina',
                                state_type='continue')
    glimps_network = MLP('glimps', layers=[])
    location_network = MLP('location', layers=[])


