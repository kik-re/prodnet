from statistics import mean, stdev
from base.data_util import parity_minus
from base.net_util import Tanh, QuasiPow
from base.experiment import MLPExperiment
from base.util import humanreadible_runtime, save_results

p = 10
if p == 2:
    expname = 'xor'
inputs, labels = parity_minus(p)
expname = 'parity_{}_hidden'.format(p)

print(inputs[0])
exit()

hidden_size = [3,4,5]
exp_params = {
    "repetitions": 10,
    "net_hyperparams": {
    # "activation_functions": [Tanh(), QuasiPow()],
    "activation_functions": [Tanh(), Tanh()],
        "learning_rate": 0.9,
        "weight_mean": 0.0,
        "weight_variance": 1.0
    },
    "max_epoch":5000,
    "success_window": 10,
}

plot_expnet_nets = []
plot_expnet_epcs = []

for h in hidden_size:
    print("Testing hidden size: {}".format(h))
    exp_params["net_hyperparams"]["layers"] = [p, h, 1]
    exp_runner = MLPExperiment(exp_params)
    results_converge,results_epochs,results_runtime = exp_runner.convergence(inputs, labels, False)
    print(results_converge)
    print(results_epochs)
    print("Total runtime",humanreadible_runtime(sum(results_runtime)))
    print("Nets converged",sum(results_converge))
    plot_expnet_nets.append("{} {}\n".format(h, sum(results_converge)))
    plot_expnet_epcs.append("{} {} {}\n".format(h, mean(results_epochs), stdev(results_epochs)))

save_results("prodnet", expname, plot_expnet_nets, plot_expnet_epcs)
