import numpy as np
import pandas as pd

# Normalization
def normalize(dataFrame):
    return (dataFrame - dataFrame.min()) / (dataFrame.max() - dataFrame.min())


def logistic(theta, x):
    # print('\nnp.dot(x, theta) = {}\n'.format(np.dot(x, theta)))
    # print('\nlogistic result = {}\n'.format(1 / (1 + np.exp(-np.dot(x, theta)))))
    return 1 / (1 + np.exp(-np.dot(x, theta)))

def cost(h, y):
    return -np.mean((y * np.log(h)) + ((1 - y) * np.log(1 - h)))


def runLogisticRegression(x, y, theta, iteration, alpha):

    previous_cost = cost(logistic(theta, x), y)

    for k in range(iteration):

        new_theta = []
        for j in range(len(theta)):
            nt = theta[j] - alpha * np.mean((logistic(theta, x) - y) * x[j])
            new_theta.append(nt)
            # print('\n\ntheta[{}] = {}\nalpha = {}\nlogistic(theta, x) = {}\nlogistic(theta, x) - y = \n{}\nx[{}] = \n{}\nnp.mean((logistic(theta, x) - y) * x[j]) = \n{}\n\n'.format(
            #     j,
            #     theta[j],
            #     alpha,
            #     logistic(theta, x),
            #     logistic(theta, x) - y,
            #     j,
            #     x[j],
            #     np.mean((logistic(theta, x) - y) * x[j])
            # ))

        theta = new_theta
        print('Iteration #{} :'.format(k + 1))
        print('new theta = \n{}'.format(new_theta))

        new_cost = cost(logistic(new_theta, x), y)
        print('J = {}, Δ = {}\n'.format(new_cost, new_cost - previous_cost))

    return theta


def runDiabetesDataSet():
    dataset = './Dataset/diabetes_dataset.csv'

    columns = [
        'Number of times pregnant',
        'Plasma glucose concentration',
        'Diastolic blood pressure',
        'Triceps skin fold thickness',
        'Serum insulin',
        'BMI',
        'Diabetes pedigree function',
        'Age',
        'Class variable'
    ]
    dataFrame = pd.read_csv(dataset, header=None, names=np.arange(1, 10))
    dataFrame.insert(0, 0, 1)
    # data.columns = columns

    # x = normalize(dataFrame.ix[:, :8])
    # after normalization, column 0 becomes NaN
    # x[0] = 1

    x = dataFrame.ix[:, :8]
    y = dataFrame[9]  # 768 y values
    theta = np.array([-6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # len = 9

    theta = runLogisticRegression(x, y, theta, iteration=500, alpha=0.000255)

    return theta

def runTestSample(theta):
    testDataset = './Dataset/test_samples.csv'
    dataFrame = pd.read_csv(testDataset, header=None, names=np.arange(1, 9))
    dataFrame.insert(0, 0, 1)
    print(dataFrame)
    # x = normalize(dataFrame.ix[:, :8])
    # x[0] = 1

    x = dataFrame.ix[:, :8]
    predicted_y = logistic(theta, x)
    print(predicted_y)
    print('My predicted result =')
    print([int(r) for r in np.where(predicted_y > 0.5, 1, 0)])

def runExample():
    data = [
        [55, 130, 0],
        [58, 160, 1],
        [62, 148, 0],
        [67, 186, 1]
    ]
    dataFrame = pd.DataFrame(data, columns=[1, 2, 3], dtype=np.float64)
    dataFrame.insert(0, 0, 1)
    print(dataFrame)
    # x = normalize(dataFrame.ix[:, :2])
    # x[0] = 1
    x = dataFrame.ix[:, :2]
    y = dataFrame[3]
    print(x)
    print(y)

    theta = np.array([0.0, 0.0, 0.0])
    # logisticRegression(data, iteration=500, alpha=0.002)
    runLogisticRegression(x, y, theta, 500, 0.002)

from logreg import validateWithSklearn

if __name__ == '__main__':
    # runExample()
    theta = runDiabetesDataSet()
    # theta = [-6.0001912691823875, 0.02165496180764577, 0.030991577379505527, -0.013117801415925143, 0.006417413200072739, -0.00086181170268060856, 0.046813179427385346, 0.0019464356359718935, 0.023055837604493094]
    # J = 0.49171945310758075, Δ = -1.6045062320301497
    runTestSample(theta)

    validateWithSklearn()
    # [0.20409552  0.06024204  0.22735258  0.6666928   0.4319579   0.05999181
    #  0.13590673  0.20094398  0.87030158  0.02350539  0.5862682   0.22218152
    #  0.31523186  0.14277819  0.23562528  0.56214336  0.1051052   0.73242782
    #  0.43983493  0.1756029   0.57929172  0.44655204  0.12644887  0.23855408
    #  0.22201285  0.47160416  0.37924086  0.20009919  0.65198341  0.11446384]
    # [0 0 0 1 0 0 0 0 1 0 1 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 1 0]
