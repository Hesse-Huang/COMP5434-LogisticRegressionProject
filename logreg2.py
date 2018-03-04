import numpy as np
import pandas as pd
from sklearn import linear_model


class LogisticRegressionModel:
    """The model to handle logistic regression"""

    def __init__(self):
        pass


    def fit(self, X, y, theta, iteration, alpha):

        previous_cost = self.__cost(self.__logistic(theta, X), y)

        for k in range(iteration):

            new_theta = []

            for j in range(len(theta)):
                nt = theta[j] - alpha * np.mean((self.__logistic(theta, X) - y) * X[j])
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
            # print('new theta = \n{}'.format(new_theta))
            for (i, t) in enumerate(new_theta):
                print('θ{} = {}'.format(i + 1, t))

            new_cost = self.__cost(self.__logistic(new_theta, X), y)
            print('J = {}, Δ = {}\n'.format(new_cost, new_cost - previous_cost))

        self.theta = theta
        return theta

    def predict(self, X):
        predicted_y = self.__logistic(self.theta, X)
        print('my predicted y =')
        print(predicted_y)
        return [int(v) for v in np.where(predicted_y > 0.5, 1, 0)]

    # Normalization
    def __normalize(self, dataFrame):
        return (dataFrame - dataFrame.min()) / (dataFrame.max() - dataFrame.min())

    # The hypothesis function
    def __logistic(self, theta, x):
        return 1 / (1 + np.exp(-np.dot(x, theta)))

    # The cost function, aka J function
    def __cost(self, h, y):
        return -np.mean((y * np.log(h)) + ((1 - y) * np.log(1 - h)))


def validate_with_sklearn(X, y, predictingX):
    logreg = linear_model.LogisticRegression()
    logreg.fit(X, y)
    logreg.predict(predictingX)
    result = [int(r) for r in logreg.predict(predictingX)]
    print('sklearn predicted result = \n{}'.format(result))


def diabetes_dataset():
    dataset = './Dataset/diabetes_dataset.csv'

    df = pd.read_csv(dataset, header=None, names=np.arange(1, 10))
    df.insert(0, 0, 1)

    # df.columns = [
    #     'Number of times pregnant',
    #     'Plasma glucose concentration',
    #     'Diastolic blood pressure',
    #     'Triceps skin fold thickness',
    #     'Serum insulin',
    #     'BMI',
    #     'Diabetes pedigree function',
    #     'Age',
    #     'Class variable'
    # ]

    # x = normalize(dataFrame.ix[:, :8])
    # x[0] = 1 # after normalization, column 0 becomes NaN

    X = df.ix[:, :8]
    y = df[9]  # 768 y values
    theta = np.array([-6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # len = 9

    return (X, y, theta)

def split(X, y, portion):
    count = int(len(X) * portion)
    training_X = X.iloc[: -count, :]
    training_y = y[: -count]
    testing_X = X.iloc[-count: , :]
    testing_y = y[-count:]
    return (training_X, training_y, testing_X, testing_y)

def error_rate(predicted_y, ground_truth_y):
    y1 = np.array(predicted_y)
    y2 = np.array(ground_truth_y)
    return np.mean(np.abs(y1 - y2))


def test_sample():
    test_dataset = './Dataset/test_samples.csv'

    df = pd.read_csv(test_dataset, header=None, names=np.arange(1, 9))
    df.insert(0, 0, 1)
    # print(dataFrame)

    # x = normalize(dataFrame.ix[:, :8])
    # x[0] = 1

    X = df.ix[:, :8]
    return X


if __name__ == '__main__':
    # load the diabetes dataset
    (X, y, theta) = diabetes_dataset()
    # split the trainingg dataset by 20%
    (training_X, training_y, cv_X, cv_y) = split(X, y, 0.2)

    model = LogisticRegressionModel()
    model.fit(training_X, training_y, theta, iteration=500, alpha=0.000255)

    tested_y = model.predict(cv_X)
    print('testing y = \n', list(cv_y))
    print('tested y = \n', tested_y)
    print('error rate = \n', error_rate(tested_y, cv_y))

    predicting_X = test_sample()
    predicted_y = model.predict(predicting_X)
    print('my predicted result =')
    print(predicted_y)

    validate_with_sklearn(X, y, predicting_X)

# testing y =
#  [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]
# tested y =
#  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
# error rate = 0.222222222222
# my predicted y =
# [ 0.19540834  0.06221878  0.24437646  0.68044723  0.42790872  0.06253981
#   0.13143393  0.21184273  0.83409151  0.02709623  0.57719967  0.23323624
#   0.32820875  0.15915501  0.231035    0.56594949  0.11080122  0.74435404
#   0.4260598   0.17781125  0.61334309  0.45731944  0.12762545  0.17411757
#   0.2100293   0.50010686  0.38886434  0.20626956  0.60440581  0.11264872]
# my predicted result =
# [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]
# sklearn predicted result =
# [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]

