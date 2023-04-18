import random
import sys
import time
import numpy as np

from base.mlp import MLP


class ExpConvergence:
    def __init__(self, params):
        self.repetitions = params["repetitions"]
        self.net_hyperparams = params["net_hyperparams"]
        self.max_epoch = params["max_epoch"]
        self.success_window = params["success_window"]

    def get_threshold(self, inp_labels):
        threshold = 0.5
        label = 0
        for item in inp_labels:
            if item[0] == -1:
                threshold = 0
                label = -1
            if item[0] == 0:
                pass
        return threshold, label

    def convergence(self, inputs, labels, show=False):
        threshold, label = self.get_threshold(labels)
        results_epochs = []
        results_runtime = []
        results_converge = []
        x_dim = len(inputs[0])
        start_time = time.time()
        for n in range(self.repetitions):
            runtime = {"start": time.time(), "end": time.time()}
            network = MLP(self.net_hyperparams)
            epoch = 0
            success = []
            indexer = list(range(len(inputs)))
            while epoch < self.max_epoch and (sum(success[-self.success_window:]) != self.success_window):
                random.shuffle(indexer)
                good_outputs = 0
                for i in indexer:
                    x = np.reshape(inputs[i], (x_dim, 1))
                    act = network.activation(x)
                    network.learning(act, labels[i])
                    y = act[-1]
                    if y[0][0] >= threshold and labels[i][0] == 1 or y[0][0] < threshold and labels[i][0] == label:
                        good_outputs += 1
                success.append(good_outputs/len(inputs))
                epoch += 1
                runtime["end"] = time.time()
            runtime_total = runtime["end"] - runtime["start"]
            converged = (sum(success[-self.success_window:]) == self.success_window)
            results_converge.append(int(converged))
            results_epochs.append(epoch)
            results_runtime.append(runtime_total)
            if show:
                print(f"Parity repetition {n} converged {converged}. Epochs reached: {epoch}. Last succ: {success[:-1]}. Runtime: {runtime_total:.1f}")
            else:
                sys.stdout.write(".")
        end_time = time.time()
        return results_converge,results_epochs,results_runtime

