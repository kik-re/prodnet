import random
import sys
import time
import numpy as np
from base.util import humanreadible_runtime
from base.mlp import MLP
from base.net_util import SquareError

class MLPExperiment:
    def __init__(self, params):
        self.repetitions = params["repetitions"]
        self.net_hyperparams = params["net_hyperparams"]
        self.max_epoch = params["max_epoch"]
        self.success_window = params["success_window"]

    def get_threshold(self, inp_labels):
        threshold = 0.5
        label = 0
        if inp_labels[0] == [-1] or inp_labels[0] == [1]:
            threshold = 0
            label = -1
        return threshold, label

    def convergence(self, inputs, labels, show=False):
        threshold, label = self.get_threshold(labels)
        samples = len(inputs)
        results_epochs = []
        results_runtime = []
        results_converge = []
        for n in range(self.repetitions):
            runtime = {"start": time.time(), "end": time.time()}
            network = MLP(self.net_hyperparams)
            epoch = 0
            success = []
            indexer = list(range(samples))
            while epoch < self.max_epoch and (sum(success[-self.success_window:]) != self.success_window):
                start_time = time.time()
                random.shuffle(indexer)
                good_outputs = 0
                for i in indexer:
                    x = np.column_stack([inputs[i]])
                    act = network.activation(x)
                    network.learning(act, labels[i])
                    y = act[-1]
                    if y[0][0] >= threshold and labels[i][0] == 1 or y[0][0] < threshold and labels[i][0] == label:
                        good_outputs += 1
                success.append(good_outputs/samples)
                epoch += 1
                runtime["end"] = time.time()
                runtime_epoch =  runtime["end"] - start_time
                # print(f"Epoch {epoch} succ {good_outputs} of {samples} in {humanreadible_runtime(runtime_epoch)}")
            runtime_total = runtime["end"] - runtime["start"]
            converged = (sum(success[-self.success_window:]) == self.success_window)
            results_converge.append(int(converged))
            results_epochs.append(epoch)
            results_runtime.append(runtime_total)
            if show:
                print(f"Net repetition {n} converged {converged}. Epochs reached: {epoch}. Last succ: {success[-1]}. Runtime: {runtime_total:.1f}")
            else:
                sys.stdout.write(".")
        return results_converge,results_epochs,results_runtime

    def generalization(self, inputs_train, labels_train, inputs_val, labels_val, show=False):
        threshold, label = self.get_threshold(labels_train)
        results_runtime = []
        results_acc_train = []
        results_error_train = []
        results_acc_val = []
        results_error_val = []
        for n in range(self.repetitions):
            print(f"Repetition {n+1}")
            runtime = {"start": time.time(), "end": time.time()}
            acc_train_epoch = []
            error_train_epoch = []
            acc_val_epoch = []
            error_val_epoch = []
            network = MLP(self.net_hyperparams)
            epoch = 0
            indexer = list(range(len(inputs_train)))
            while epoch < self.max_epoch:
                start_time = time.time()
                succ_train = 0
                se_train = 0
                random.shuffle(indexer)
                for i in indexer:
                    x = np.column_stack([inputs_train[i]])
                    act = network.activation(x)
                    network.learning(act, labels_train[i])
                    y = act[-1]
                    if y[0][0] >= threshold and labels_train[i][0] == 1 or y[0][0] < threshold and labels_train[i][0] == label:
                        succ_train += 1
                    se_train += SquareError().apply_func(labels_train[i][0], y)
                acc_train_epoch.append(succ_train / len(inputs_train))
                error_train_epoch.append(np.sqrt(se_train) / len(inputs_train))
                succ_val = 0
                se_val = 0
                for i in range(len(inputs_val)):
                    x = np.column_stack([inputs_val[i]])
                    act = network.activation(x)
                    y = act[-1]
                    if y[0][0] >= threshold and labels_val[i][0] == 1 or y[0][0] < threshold and labels_val[i][0] == label:
                        succ_val += 1
                    se_val += SquareError().apply_func(labels_val[i][0], y)
                acc_val_epoch.append(succ_val / len(labels_val))
                error_val_epoch.append(np.sqrt(se_val) / len(inputs_val))
                epoch += 1
                runtime["end"] = time.time()
                runtime_epoch = runtime["end"] - start_time
                if (epoch+1) % 100 == 0:
                    print(f"Epoch {epoch} T/V ACC: {acc_train_epoch[-1]} / {acc_val_epoch[-1]} RMS: {error_train_epoch[-1]}"
                      f" / {error_val_epoch[-1]} in {humanreadible_runtime(runtime_epoch)}")
                # else:
                #     sys.stdout.write(".")
            runtime_total = runtime["end"] - runtime["start"]
            results_acc_train.append(acc_train_epoch)
            results_error_train.append(error_train_epoch)
            results_acc_val.append(acc_val_epoch)
            results_error_val.append(error_val_epoch)
            results_runtime.append(runtime_total)
            if show:
                print(f"Repetition {n}. Training acc:{acc_train_epoch[-1]} mse:{error_train_epoch[-1]} "
                      f"validation acc:{acc_val_epoch[-1]} Runtime: {runtime_total:.1f}")
            else:
                sys.stdout.write(".")
        return results_acc_train,results_error_train,results_acc_val,results_runtime

