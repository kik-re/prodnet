import numpy as np
from base.net_util import Sigmoid, Tanh, QuasiPow


class MLP:
    def __init__(self, params):
        # read hyperparameters
        self.layers = params["layers"]  # [ input dimension, h_1, ... , h_n, output dimension ]
        self.activation_functions = params["activation_functions"]
        self.learning_rate = params["learning_rate"]
        # initialize Weights
        self.W = []
        for i in range(len(self.layers) - 1):
            if type(self.activation_functions[i]) == QuasiPow:
                self.W.append(np.random.normal(params["weight_mean"], params["weight_variance"],
                                               (params["layers"][i + 1], params["layers"][i])))
            else:  # add bias if not Quasi
                self.W.append(np.random.normal(params["weight_mean"], params["weight_variance"],
                                               (params["layers"][i + 1], params["layers"][i] + 1)))
        self.sigmoid = Sigmoid()

    def activation(self, x):
        act = [x]
        for i in range(len(self.activation_functions)):
            if type(self.activation_functions[i]) == QuasiPow:
                rows = self.W[i].shape[0]
                columns = act[-1].shape[1]
                act_i = np.ones((rows, columns))
                for x in range(rows):
                    for y in range(columns):
                        for z in range(self.W[i].shape[1]):
                            act_i[x][y] *= self.activation_functions[i].apply_func(act[-1][z][y], self.sigmoid.apply_func(self.W[i][x][z]))
            else:
                biased_act_i = np.vstack([act[-1], np.ones(len(act[-1][0]))])
                act_i_dot = np.dot(self.W[i], biased_act_i)
                act_i = self.activation_functions[i].apply_func(act_i_dot)
            act.append(act_i)
        return act

    def learning(self, act, d):
        y = act[-1]
        error = d - y
        w_changes = []
        for i in range(len(self.activation_functions) - 1, -1, -1):
            if type(self.activation_functions[i]) == QuasiPow:
                error_new, w_change_new = self.delta_quasi(act, i, error)
            else:
                error_new, w_change_new = self.delta(act, i, error)
            error = error_new
            w_changes.insert(0, w_change_new)
        for i in range(len(w_changes)):
            self.W[i] += self.learning_rate * w_changes[i]

    def delta_quasi(self, act, i, error):
        error_new = act[i].copy()
        w_change_new = self.W[i].copy()
        delta = error * act[i + 1]
        for j in range(self.layers[i]):
            tmp = 0
            for k in range(self.layers[i + 1]):
                quasi_exp = self.sigmoid.apply_func(self.W[i][k][j])
                common_term_j_k = self.activation_functions[i].apply_func(act[i][j][0], quasi_exp)
                if common_term_j_k < 0.0000000001:
                    common_term_j_k = 1.0
                    for j2 in range(self.layers[i]):
                        if j2 != j:
                            common_term_j_k *= self.activation_functions[i].apply_func(act[i][j2][0],
                                   self.sigmoid.apply_func(self.W[i][k][j2]))
                    common_term_j_k *= error[k][0]
                else:
                    common_term_j_k = delta[k] / common_term_j_k
                w_change_new[k][j] = common_term_j_k * (act[i][j] - 1) * quasi_exp * (1 - quasi_exp)
                tmp += common_term_j_k * quasi_exp
            error_new[j] = tmp
        return error_new, w_change_new

    def delta(self, act, i, error):
        delta = error * self.activation_functions[i].apply_derived(act[i + 1])
        biased_act = np.vstack([act[i], np.ones(len(act[i][0]))])
        w_change_new = np.dot(biased_act, delta.transpose()).transpose()
        error_new = np.dot(self.W[i][:, :self.layers[i]].transpose(), delta)
        return error_new, w_change_new