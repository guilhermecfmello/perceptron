import numpy as np
import numpy.random as rd
import matplotlib as plt
import csv


def getDatabase(dbName):
    with open(dbName) as db:
        csvInput = csv.reader(db, delimiter=',', quotechar='|')
        nRows = sum(1 for line in csvInput)
        db.seek(0)
        inputs = np.zeros(shape=(nRows, 2), dtype=float)
        outputs = np.array([], dtype=float)
        i = 0
        for c in csvInput:
            inputs[i][0] = c[0].replace('.','')
            inputs[i][1] = c[1].replace('.','')
            outputs = np.append(outputs, float(c[2]))
            i = i + 1
        return inputs, outputs
    return []

def linear_combination(inputs, weights, bias):
    return inputs[0] * weights[0] + inputs[1] * weights[1] + bias


# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calcWeights(weight, bias, lr, n, error):
    return weight - 1/n*lr * error


# quadractic error
def calcError(y, d, xn):
    return (2*y-2*d)*(y*(1-y))*xn

def calcError(y, d, xn):
    return (((2*y) - (2*d)) * (y*(1-y)))
# def derivative(y, d):
    # return (((2y) - (2d)) * (y*(1-y)))

def trainning(inputs, outputs, ages = 0, lr = 0.1):
    # Defining random weights
    weights = rd.uniform(-1,1,2)
    bias = rd.uniform(-1,1)
    n = len(inputs)
    for a in range(ages):
        for i in range(len(outputs)):
            # calculating sum with bias
            comb = linear_combination(inputs[i], weights, bias)
            y = sigmoid(comb)
            for j in range(len(weights)):
                error = calcError(y, outputs[i], inputs[i][j])
                weights[j] = calcWeights(weights[j], bias, lr, n, error)
            # print(weights, bias)
    return weights, bias

def compare(inputs,outputs,weights, bias):
    rights = 0
    wrong = 0
    for i in range(len(outputs)):
        y = linear_combination(inputs[i],weights, bias)
        y = 1 if y > 0.5 else 0
        error = y - outputs[i]
        if error == 0:
            rights = rights + 1
        else: wrong = wrong + 1
        print("finalError: " + str(error))
    return rights, wrong


inputs, outputs = getDatabase('perceptron-database.csv')
ages = 1000
learningRate = 0.01
weights, bias = trainning(inputs, outputs, ages, learningRate)
rights, wrongs = compare(inputs, outputs, weights, bias)
print(weights, bias)

print("Certos: " + str(rights) + " Errados: " + str(wrongs))

