from abc import ABCMeta
from abc import abstractmethod

# simple Abastract Base Class demo (Not exactly what I wnat....
# anyway......


class Foo(object):

    def oop_trash(self):
        return "agreement from Foo"


class LayerMeta(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self):
        raise NotImplementedError

    def getitem(self, index):
        return self.__getitem__(index)

    @classmethod
    def __subclasshook__(cls, C):
        if cls is LayerMeta:
            if any("oop_trash" in base.__dict__ for base in C.__mro__):
                return True
        return NotImplementedError


LayerMeta.register(Foo)  # Now the LayerMeta has attribute


class LayerFoo(LayerMeta):
    def __init__(self):
        print "OOP sucks"

    def __getitem__(self, index):
        return "Foo is Here {0}".format(index)

    def oop_trash(self):
        return False


if __name__ == "__main__":
    layer, n_vis, n_hidden = [None] * 3
    print dir(LayerMeta)
    assert issubclass(Foo, LayerMeta)
    m = LayerFoo()
    print m.oop_trash()
    # m = LayerMeta([19, 12])
    # print LayerMeta()(10, 10)
    # print dir(LayerMeta())
    # print tuple.__mro__
    # print dir(iter(range(10)))
    # print dir(tuple)
