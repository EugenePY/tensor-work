
#####################################################
#    Design Pattern for Decorator for the project   #
#####################################################

import functools
import abc

from model import LayerMeta



class DropOut(object):

    def dropout_lstm(self):
        pass


class LSTM(object):

    @abc.abstractmethod
    def partial_fprop(self, state_below):



class LSTMDropOut(LSTM, DropOut):
    pass
