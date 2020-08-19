import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import csv


def getDatabase(dbName):
    with open(dbName) as db:
        csvInput = csv.reader(db, delimiter=',', quotechar='|')
        nRows = sum(1 for line in csvInput)
        db.seek(0)
        inputs = np.zeros(shape=(nRows, 2), dtype=np.double)
        outputs = np.array([], dtype=np.double)
        i = 0
        for c in csvInput:
            inputs[i][0] = c[0]
            inputs[i][1] = c[1]
            outputs = np.append(outputs, float(c[2]))
            i = i + 1
        return inputs, outputs
    return []

def linear_combination(inputs, weights, bias):
    return np.dot(inputs, weights) + bias

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calcWeights(weight, lr, n, error):
    temp = weight - (1/n)*lr * error
    return temp 

def derivative(y, d):
    return (((2*y) - (2*d)) * (y*(1-y)))

def trainning(inputs, outputs, ages = 0, lr = 0.1):
    # Defining random weights
    weights = rd.uniform(-1,1,2)
    bias = rd.uniform(-1,1)
    n = len(inputs)
    for a in range(ages):
        for i in range(n):
            # calculating sum with bias
            comb = linear_combination(inputs[i], weights, bias)
            y = sigmoid(comb)
            for j in range(len(weights)):
                error = derivative(y, outputs[i]) * inputs[i][j]
                weights[j] = calcWeights(weights[j], lr, n, error)
            error = derivative(y, outputs[i])
            bias = calcWeights(bias, lr, n, error)

    return weights, bias

def compare(inputs,outputs,weights, bias):
    # positives rights
    pr = 0 
    # positives wrongs
    pw = 0
    # negatives rights
    nr = 0
    # negatives wrong
    nw = 0
    for i in range(len(outputs)):
        y = sigmoid(linear_combination(inputs[i],weights, bias))
        y = 1. if y > 0.5 else 0.
        print(str(outputs[i]) + " == " + str(y))
        if(outputs[i] == 1):
            if y == outputs[i]:
                pr = pr + 1
            else:
                pw = pw + 1
        else:
            if y == outputs[i]:
                nr = nr + 1
            else:
                nw = nw + 1
    rights = pr+nr
    wrongs = pw+nw
    print("Total: " + str(rights + wrongs))
    print("Rights: " + str(rights) + " Wrongs: " + str(wrongs))
    print("Percent rights: " + str((rights/len(inputs))*100) + "%")
    print("Positives rights: " + str(pr))
    print("Positives wrongs: " + str(pw))
    print("Negatives rights: " + str(nr))
    print("Negatives wrongs: " + str(nw))
    return rights, wrongs

def plot_all(X, Y, w):
    pos_X = np.take(X[:, 0], np.where(Y == 1))
    pos_Y = np.take(X[:, 1], np.where(Y == 1))
    neg_X = np.take(X[:, 0], np.where(Y == 0))
    neg_Y = np.take(X[:, 1], np.where(Y == 0))
    plt.plot(pos_X, pos_Y, "+r")
    plt.plot(neg_X, neg_Y, "+b")
    xx = np.linspace(-3, 4)  # hyperplane? '-''
    # print(xx)
    plt.plot(xx, (w[0] * xx + w[1]), "green")  # dúvida
    plt.show()

inputs, outputs = getDatabase('data.csv')
ages = 20
learningRate = 0.2
weights, bias = trainning(inputs, outputs, ages, learningRate)
rights, wrongs = compare(inputs, outputs, weights, bias)

plot_all(inputs, outputs, weights)


