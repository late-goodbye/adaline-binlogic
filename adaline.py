class Adaline(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.weights = [0 for _ in range(len(X[0]) + 1)]
        self.costs = []

        for _ in range(self.n_iter):
            output = [self.predict(x) for x in X]
            errors = [y - o for y, o in zip(y, output)]
            self.weights[0] += self.eta * sum(errors)
            for i in range(len(X[0])):
                self.weights[i+1] += self.eta * sum(
                        [X[j][i] * e for j, e in zip(
                            range(len(X)), errors)])
            self.costs.append(sum(map(lambda x: x * x, errors)) / 2.0)
        return self
    
    def net_input(self, x):
        return (sum([xi * wi for xi, wi in zip(x, self.weights[1:])])
            + self.weights[0])

    def activation(self, x):
        return self.net_input(x)

    def predict(self, x):
        return 1 if self.activation(x) >= 0 else 0


if __name__ == '__main__':

    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 0, 0, 1]

    ada = Adaline().fit(X, y)
    
    print('Weights:', ada.weights)
    print('Costs:', ada.costs)
    
    assert ada.predict([0, 0]) == 0
    assert ada.predict([0, 1]) == 0
    assert ada.predict([1, 0]) == 0
    assert ada.predict([1, 1]) == 1

