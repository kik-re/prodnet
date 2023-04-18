import time


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
    return '{:d}:{:02d}:{:02d}'.format(int(h), int(m), round(s))