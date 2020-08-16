import numpy as np
import numpy.random as rd
import matplotlib as mt
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

def calcOutput(inputs, weights, bias):
    sum = inputs[0] * weights[0] + inputs[1] * weights[1] + bias
    return np.tanh(sum)

def calcWeights(inputs, weights, bias, lr, error):
    for i in range(len(weights)):
        # weights[i] = weights[i] + (lr * inputs[i] * error) # old
        weights[i] = weights[i] - 1/len(inputs)*lr
    bias = bias - 1/len(inputs)*lr 
    return weights, bias


# quadractic error
def calcError(y, yCalc):
    error = (yCalc - y)**2


def gradient():
    
def trainning(inputs, outputs, ages = 0, lr = 0.1):
    # Defining random weights
    weights = rd.uniform(-1,1,2)
    bias = rd.uniform(-1,1)

    for a in range(ages):
        for i in range(len(outputs)):
            # print(inputs[i][1])
            # exit()
            # calculating sum with bias
            y = calcOutput(inputs[i], weights, bias)
            
            error = calcError(outputs[i], y)
        
            
            # print(y)
            weights, bias = calcWeights(inputs[i], weights, bias, lr, error)
            # print(weights, bias)
            




inputs, outputs = getDatabase('perceptron-database.csv')
ages = 1
learningRate = 0.1
trainning(inputs, outputs, ages, learningRate)

