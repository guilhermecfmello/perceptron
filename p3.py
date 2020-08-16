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

def calcOutput(e):
    sum = e.dot(weights)
    return stepFunction(sum)


def trainning(ages = 0, outputs):
    for a in range(ages):
        for i in range(len(outputs)):
                
            outputCalc = calcOutput(np.asarray(inputs[i]))
            error = outputs[i] - outputCalc
            totalError += error
            for j in range(len(weights)):
                weights[j] = weights[j] + (learningRate * inputs[i][j] * error)
                print("weights refreshed: " + str(weights[j]))


inputs, outputs = getDatabase('perceptron-database.csv')
# Defining random weights
weights = rd.uniform(-1,1,[1,3])
learningRate = 0.1




trainning(1)

