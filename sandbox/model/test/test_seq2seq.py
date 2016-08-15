from model.im2latex import Seq2Seq
from util.tests import Im2LatexTest
from models.lstm import LSTMwithAttention
from model.highway import HighWay
from pylearn2.sandbox.rnn.models.rnn import LSTM


@Im2LatexTest.test_Seq2Seq
def test_Seq2Seq():
    encoder_layers = [LSTM]
    bridge_layers = [HighWay]
    decoder_layers = [LSTMwithAttention]
    model = Seq2Seq(encoder_layers=encoder_layers, bridge_layers=bridge_layers,
                    decoder_layers=decoder_layers)
    return model
