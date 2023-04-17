import math
import numpy as np
from decimal import Decimal


class Sigmoid:
    def __init__(self):
        pass

    def __repr__(self):
        return "Sigmoid"

    def apply_func(self, net):
        return 1.0 / (1.0 + np.exp(-net))

    def apply_derived(self, output):
        return output * (1 - output)


class SoftMax:
    def __init__(self):
        pass

    def __repr__(self):
        return "SoftMax"

    def apply_func(self, net):
        e_net = np.exp(net - np.max(net))
        e_denom0 = e_net.sum(axis=0, keepdims=True)
        result = e_net / e_denom0
        return result

    def apply_derived(self, output):
        return output * (1 - output)


class OutputProcessor:
    def __init__(self):
        pass

    def make_one_hot(input_array):
        output_array = np.zeros((input_array.size, input_array.max() + 1))
        output_array[np.arange(input_array.size), input_array] = 1
        return output_array


class Exp2:
    def __init__(self):
        pass

    def apply_derived(self, output):
        if output >= 0:
            if output > 1:
                return np.NAN
            return - np.log(1 - output / 2)

        elif output <= -0:
            if output < -1:
                return np.NAN
            return np.log(1 + output / 2)

        return output  # NaN


class Tanh:
    def __init__(self):
        pass

    def __repr__(self):
        return "Tanh"

    def apply_func(self, net):
        return np.tanh(net)

    def apply_derived(self, x):
        return (1 + x) * (1 - x)


class QuasiPow:
    def __init__(self):
        pass

    def __repr__(self):
        return "QuasiPow"

    def apply_func(self, base, exp):
        return 1 - exp * (1 - base)

    def apply_derived(self, x):
        return
