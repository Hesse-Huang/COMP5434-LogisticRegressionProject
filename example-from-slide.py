import numpy as np
import pandas as pd
from logreg2 import LogisticRegressionModel

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
    X = dataFrame.ix[:, :2]
    y = dataFrame[3]
    print(X)
    print(y)

    theta = np.array([0.0, 0.0, 0.0])
    # logisticRegression(data, iteration=500, alpha=0.002)
    # runLogisticRegression(x, y, theta, 500, 0.002)

    model = LogisticRegressionModel()
    model.fit(X, y, theta, 500, 0.002)

if __name__ == '__main__':
    runExample()