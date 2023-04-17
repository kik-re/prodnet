import numpy as np
from sklearn.model_selection import KFold


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


def banana():
    inputs, labels = [], []
    with open("banana_dataset.arff") as file:
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
    with open("banana_dataset.arff") as file:
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

