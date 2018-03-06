import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


class LogisticRegressionModel:
    """The model to handle logistic regression"""

    def __init__(self):
        self.cost_records = []

    def fit(self, X, y, theta, iteration, alpha):

        cost_records = []
        previous_cost = self.__cost(self.__logistic(theta, X), y)

        for k in range(iteration):
            new_theta = []
            for j in range(len(theta)):
                nt = theta[j] - alpha * np.mean((self.__logistic(theta, X) - y) * X[j])
                new_theta.append(nt)
            theta = new_theta
            print('Iteration #{} :'.format(k + 1))

            for (i, t) in enumerate(new_theta):
                print('θ{} = {}'.format(i, t))

            new_cost = self.__cost(self.__logistic(new_theta, X), y)
            print('J = {}, Δ = {}\n'.format(new_cost, new_cost - previous_cost))
            previous_cost = new_cost
            cost_records.append(new_cost)

        self.theta = theta
        self.cost_records.append((alpha, cost_records))

        return theta

    def predict(self, X):
        predicted_y = self.__logistic(self.theta, X)
        return [int(v) for v in np.where(predicted_y > 0.5, 1, 0)]

    def plot_cost_function_variation(self):
        plots = []
        alphas = []
        for (alpha, cost_records) in self.cost_records:
            y = cost_records
            x = range(len(y))
            p = plt.scatter(x, y, s=2)
            plots.append(p)
            alphas.append(alpha)

        plt.title('Evolvement of Cost Function Value')
        plt.xlabel('Number of Iteration')
        plt.ylabel('J(θ)')

        labels = ['α = {}'.format(a) for a in alphas]
        plt.legend(handles=plots, labels=labels)

    # The hypothesis function
    def __logistic(self, theta, x):
        return 1 / (1 + np.exp(-np.dot(x, theta)))

    # The cost function, aka J function
    def __cost(self, h, y):
        return -np.mean((y * np.log(h)) + ((1 - y) * np.log(1 - h)))


def validate_with_sklearn(X, y, predicting_X):
    logreg = linear_model.LogisticRegression()
    logreg.fit(X, y)
    logreg.predict(predicting_X)
    result = [int(r) for r in logreg.predict(predicting_X)]
    return result


def diabetes_dataset():
    dataset = './Dataset/diabetes_dataset.csv'
    df = pd.read_csv(dataset, header=None, names=np.arange(1, 10))
    df.insert(0, 0, 1)

    X = df.iloc[:, :9]
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
    X = df.iloc[:, :9]
    return X


if __name__ == '__main__':
    # load the diabetes dataset
    (X, y, theta) = diabetes_dataset()
    # split the training dataset by 20%
    (training_X, training_y, cv_X, cv_y) = split(X, y, 0.2)

    model = LogisticRegressionModel()

    # uncomment these three lines if you want to compare different alpha on a graph
    # model.fit(training_X, training_y, theta, iteration=500, alpha=0.001000)
    # model.fit(training_X, training_y, theta, iteration=500, alpha=0.000030)
    # model.fit(training_X, training_y, theta, iteration=500, alpha=0.000005)
    model.fit(training_X, training_y, theta, iteration=500, alpha=0.000288)

    # uncomment the following line of code if you want to plot cost function variation
    # model.plot_cost_function_variation()

    cv_predicted_y = model.predict(cv_X)
    print('[20% cross validation] ground truth y = \n', list(cv_y))
    print('[20% cross validation] predicted y = \n', cv_predicted_y)
    print('[20% cross validation] error rate = \n', error_rate(cv_predicted_y, cv_y))

    predicting_X = test_sample()
    predicted_y = model.predict(predicting_X)
    print('[test sample] my predicted result =\n{}'.format(predicted_y))

    result = validate_with_sklearn(X, y, predicting_X)
    print('[test sample] sklearn predicted result = \n{}'.format(result))


# Iteration #500 :
# θ0 = -6.000196781930353
# θ1 = 0.023242005168141424
# θ2 = 0.028818544308311874
# θ3 = -0.01192406877290299
# θ4 = 0.0032677609382930394
# θ5 = -0.0003714984835183602
# θ6 = 0.057551866141246764
# θ7 = 0.0024377987090487634
# θ8 = 0.018707523363176565
# J = 0.4907776746149357, Δ = -1.2655252140614248e-05
#
# [20% cross validation] ground truth y =
#  [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]
# [20% cross validation] predicted y =
#  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
# [20% cross validation] error rate =
#  0.222222222222
# [test sample] my predicted result =
# [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]
# [test sample] sklearn predicted result =
# [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
