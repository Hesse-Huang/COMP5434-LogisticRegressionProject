#!/usr/bin/env python3

import csv
import math

dataset = './Dataset/diabetes_dataset.csv'
testDataset = './Dataset/test_samples.csv'

def sigmoid(x):
    return 1.0 / (1.0 + math.e ** (-float(x)))

def hythesis(thetas, xs):
    s = sum([theta * x for theta, x in zip(thetas, xs)])
    return sigmoid(s)

def normalize(data):
    count = len(data[0])
    # [(minValue, maxValue)]
    minMaxList = []
    for c in range(count):
        column = [row[c] for row in data]
        minMaxList.append((min(column), max(column)))

    data = [[(value - minMaxList[c][0]) / (minMaxList[c][1] - minMaxList[c][0]) for c, value in enumerate(row)] for row in data]
    return data


def logisticRegression(data, iteration, alpha=0.001, intercept=0.0):

    m = len(data)
    ys = []

    for xs in data:
        ys.append(float(xs.pop()))

    # data = normalize(data)

    for xs in data:
        xs.insert(0, 1.0)

    thetas = ([0] * (len(data[0]) - 1)) # initial thetas are all zero
    thetas = [intercept] + thetas
    # print(thetas)
    previousCost = 0

    # print(data)

    # print('m =', m)
    # print('thetas =', thetas)
    # print('ys =', ys)

    def costFunction():
        summing = sum([
            y * math.log(hythesis(thetas, xs)) + \
            (1 - y) * math.log(1 - hythesis(thetas, xs)) \
            for y, xs in zip(ys, data)
        ])
        return summing / (-m)

    for k in range(iteration):
        newThetas = []
        for j, theta in enumerate(thetas):
            summing = sum([(hythesis(thetas, xs) - y) * xs[j] for xs, y in zip(data, ys)])
            newTheta = theta - (alpha / m) * summing
            newThetas.append(newTheta)
        thetas = newThetas
        cost = costFunction()

        print('iteration #{} :'.format(k + 1))
        str = ''
        for i, theta in enumerate(newThetas):
            str += 'θ{} = {}\n'.format(i, theta)
        str += 'J = {}, Δ = {}\n'.format(cost, previousCost - cost)
        print(str)

        previousCost = cost

    return (thetas, cost)

# Run the example given in the slide
def runExample():
    data = [
        [55, 130, 0],
        [58, 160, 1],
        [62, 148, 0],
        [67, 186, 1]
    ]
    logisticRegression(data, iteration=10, alpha=0.002)


def runDiabetesDataset():
    with open(dataset, 'r', newline='') as file:
        reader = csv.reader(file)
        data = [[float(v) for v in row] for row in reader]
        return logisticRegression(data, iteration=500, alpha=0.000255, intercept=-6)

def predictResult(data, thetas):

    results = []
    ys = []

    for i, xs in enumerate(data):
        xs.insert(0, 1)
        y = hythesis(thetas, xs)
        result = 1 if y > 0.5 else 0
        results.append(result)
        ys.append(y)
        str = 'For test item #{}:\ny = {}\nprediction: {}\n'.format(i + 1, y, result)
        print(str)
    return (results, ys)

def predictWithGivenSampleData(thetas):
    with open(testDataset, 'r', newline='') as file:
        reader = csv.reader(file)
        data = [[float(v) for v in row] for row in reader]
        return predictResult(data, thetas)


from sklearn import linear_model

def validateWithSklearn():
    with open(dataset, 'r', newline='') as file:
        reader = csv.reader(file)
        Xs = []
        ys = []
        for row in reader:
            ys.append(row.pop())
            Xs.append(row)
        logreg = linear_model.LogisticRegression()
        logreg.fit(Xs, ys)

        with open(testDataset, 'r', newline='') as test:
            t_reader = csv.reader(test)
            t_xs = [[float(v) for v in r] for r in t_reader]
            result = [int(r) for r in logreg.predict(t_xs)]
            print('sklearn predicted result = \n{}'.format(result))



def crossValidate():
    with open(dataset, 'r', newline='') as file:
        reader = csv.reader(file)
        data = [[float(v) for v in row] for row in reader]
        testData = data[-20:]
        data = data[:-20]
        (thetas, cost) = logisticRegression(data, iteration=1000, alpha=0.000255, intercept=-6)
        results = predictResult(testData, thetas)

        ys = [int(row[-1]) for row in testData]
        print('rs: {}\nys: {}\n'.format(results, ys))


if __name__ == '__main__':
    # runExample()

    # (thetas, cost) = runDiabetesDataset()
    # print(thetas)
    # print(cost)
    thetas = [-6.0001912691823875, 0.02165496180764577, 0.030991577379505527, -0.01311780141592515, 0.006417413200072731,
     -0.0008618117026806096, 0.04681317942738537, 0.0019464356359718935, 0.023055837604493076]
    # 0.491719453107581
    (result, ys) = predictWithGivenSampleData(thetas)

    # My final result
    print('my predicted result =')
    print(result)
    # The sklearn result
    validateWithSklearn()