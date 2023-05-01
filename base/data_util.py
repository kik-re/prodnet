import random

import numpy as np

def parity(n):
    inputs, labels = [], []
    inputs = [list(bin(x)[2:].rjust(n, '0')) for x in range(2 ** n)]
    for i in range(len(inputs)):
        inputs[i] = list(map(int, inputs[i]))
        if inputs[i].count(1) % 2 == 0:
            labels.append([0])
        else:
            labels.append([1])
    return inputs, labels


def parity_minus(n):
    inputs, labels = [], []
    inputs = [list(bin(x)[2:].rjust(n, '0')) for x in range(2 ** n)]
    for i in range(len(inputs)):
        inputs[i] = list(map(int, inputs[i]))
        if inputs[i].count(1) % 2 == 0:
            labels.append([-1])
        else:
            labels.append([1])
        for j in range(len(inputs[i])):
            if inputs[i][j] == 0:
                inputs[i][j] = -1
    return inputs, labels

def banana():
    inputs, labels = [], []
    with open("../data/banana_dataset.arff") as file:
        for riadok in file:
            riadok = riadok.split(",")
            inputs.append([float(item) for item in riadok[:2]])
            if int(riadok[2]) == 2:
                labels.append([-1])
            else:
                labels.append([1])
    return inputs, labels


def banana_transformed():
    inputs, labels = [], []
    with open("../data/banana_dataset.arff") as file:
        for riadok in file:
            riadok = riadok.split(",")
            inputs.append([float(item) for item in riadok[:2]])
            if int(riadok[2]) == 2:
                labels.append([-1])
            else:
                labels.append([1])
    m = np.max(inputs)
    inputs = inputs / m
    return inputs, labels


# 2 spirals dataset from https://arxiv.org/abs/0911.1210
def spirals_load_transform(data_step=2, test_ratio=0.1):
    inputs = []
    labels = []
    with open("../data/spirals-10x780.arff") as file:
        for riadok in file:
            riadok = riadok.split(",")
            inputs.append([float(item)/10 for item in riadok[:2]])
            labels.append([-1] if int(riadok[2]) == 0 else [1])
    indexer = list(range(len(inputs)))
    dsize = int(len(indexer)/data_step)
    tsize = int(test_ratio*dsize)
    random.shuffle(indexer)
    idx_train = indexer[0:dsize-tsize]
    idx_test = indexer[dsize-tsize:dsize]
    return [inputs[i] for i in idx_train], [labels[i] for i in idx_train], \
        [inputs[i] for i in idx_test], [labels[i] for i in idx_test]

def twospirals(n_points, step_test=5):
    n_train = list(range(n_points))
    n_test = n_train[::step_test]
    del n_train[::step_test]

    x_train = []
    y_train = []
    for n in n_train:
        r = 0.4 * (105 - n) / 104
        a = np.pi * (n - 1) / 16
        x1 = r * np.sin(a) + 0.5
        x2 = r * np.cos(a) + 0.5
        x_train.append([2*x1-1, 2*x2-1])
        x_train.append([2*(1-x1)-1, 2*(1-x2)-1])
        y_train.append([1])
        y_train.append([0])

    x_test = []
    y_test = []
    for n in n_test:
        r = 0.4 * (105 - n) / 104
        a = np.pi * (n - 1) / 16
        x1 = r * np.sin(a) + 0.5
        x2 = r * np.cos(a) + 0.5
        x_test.append([2*x1-1, 2*x2-1])
        x_test.append([2*(1-x1)-1, 2*(1-x2)-1])
        y_test.append([1])
        y_test.append([0])

    train = [x_train, y_train]
    test = [x_test, y_test]

    return train, test

def twospirals_minus(n_points, step_test=11):
    train, test = twospirals(n_points, step_test)

    x_train, y_train = train
    y_train_new = []
    for label in y_train:
        if label[0] == 0:
            y_train_new.append([-1])
        else:
            y_train_new.append([1])

    x_test, y_test = test
    y_test_new = []
    for label in y_test:
        if label[0] == 0:
            y_test_new.append([-1])
        else:
            y_test_new.append([1])

    train = [x_train, y_train_new]
    test = [x_test, y_test_new]

    return train, test

# 2 spirals dataset from https://glowingpython.blogspot.com/2017/04/solving-two-spirals-problem-with-keras.html?m=1

def twospirals_raw(n_points, noise=0.5):
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
            np.hstack((np.zeros(n_points), np.ones(n_points))))


def spirals(points, test_batch_size=0.2):
    x, y = twospirals_raw(points)
    y = np.reshape(y, (len(y), 1))
    return x, y


def spirals_minus(points, test_batch_size=0.2):
    x, y = twospirals_raw(points)
    y = np.where(y == 0, -1, y)
    y = np.reshape(y, (len(y), 1))
    return x, y


def spirals_minus_transformed(points):
    x, y = twospirals_raw(points)
    m = np.max(x)
    x = x / m
    y = np.where(y == 0, -1, y)
    y = np.reshape(y, (len(y), 1))
    return x, y