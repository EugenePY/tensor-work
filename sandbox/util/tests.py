from dataset.im2latex import Im2LatexData
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.cost import Cost


class Im2LatexTest:

    @staticmethod
    def test_layers(func):
        try:
            dataset = Im2LatexData(which_set='debug')
            func(dataset)
            print str(func) + " Status: PASS"
        except:
            dataset = Im2LatexData(which_set='debug')
            func(dataset)

    def test_structure(self, test):
        pass

    @staticmethod
    def call_test(func):
        try:
            func()
            print str(func) + " Status: PASS"
        except:
            func()

    @staticmethod
    def test_Seq2Seq(func):
        model = func()
        try:
            cost = model.cost_from_X(dataset)
            print str(func) + " Status: Pass"
        except:
            cost = model.cost_from_X(dataset)

if __name__ == '__main__':
    pass


