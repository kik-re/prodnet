import time
import os


def notetime(starttime):
    exp_end = time.time()
    runtime = exp_end - starttime
    m, s = divmod(runtime, 60)
    h, m = divmod(m, 60)
    # print(s)
    print('\nExperiment finished in {:d}:{:02d}:{:02d}'.format(int(h), int(m), round(s)))


def humanreadible_runtime(runtime):
    m, s = divmod(runtime, 60)
    h, m = divmod(m, 60)
    return '{:d}:{:02d}:{:02d}'.format(int(h), int(m), round(s))\


def save_results( net_name, exp_name, nets, epc):
    if not os.path.exists("results"):
        os.makedirs("results")
    with open(f'results/{net_name}_{exp_name}_nets.txt', 'a') as f:
        f.write('x y\n')
        f.writelines(nets)
    with open(f'results/{net_name}_{exp_name}_epcs.txt', 'a') as f:
        f.write('x y err\n')
        f.writelines(epc)


def check_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)
