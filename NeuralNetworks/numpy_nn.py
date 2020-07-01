import numpy as np
import matplotlib.pyplot as plt

# Data Gen
Nclass = 500
D = 2  # dimensionality of input
X1 = np.random.randn(Nclass, D) + np.array([0, -2])
X2 = np.random.randn(Nclass, D) + np.array([2, 2])
X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])                                 # numpy 2D array N * D
Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)    # numpy 1D array N

plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.title('Data')
plt.show(block=Flase)

class myNN(object):
    def __init__(self, X, Y, M1, M2):
        # Data
        self.X = X
        self.Y = Y

        # Number of units in two hidden layers
        self.M1 = M1
        self.M2 = M2

        self.N = len(Y)             # number of observations
        self.K = len(np.unique(Y))  # number of classes
        self.D = X.shape[1]         # dimensionality of input

        # randomly initialize weights
        self.W1 = np.random.randn(self.D, self.M1)
        self.b1 = np.random.randn(self.M1)

        self.W2 = np.random.randn(self.M1, self.M2)
        self.b2 = np.random.randn(self.M2)

        self.W3 = np.random.randn(self.M2, self.K)
        self.b3 = np.random.randn(self.K)


    def preproc(self):
        T = np.zeros((self.N, self.K))
        for i in range(self.N):
            T[i, self.Y[i]] = 1
        self.T = T
        return T

    def predict_proba(self, X_new):
        Z1 = 1 / (1 + np.exp(-X_new.dot(self.W1) - self.b1))
        Z2 = 1 / (1 + np.exp(-Z1.dot(self.W2) - self.b2))
        A = Z2.dot(self.W3) + self.b3
        expA = np.exp(A)
        P = expA / expA.sum(axis=1, keepdims=True)
        self.Z1 = Z1
        self.Z2 = Z2
        self.P = P
        return P

    def clf_accuracy(self):
        n_correct = 0
        n_total = 0
        P = np.argmax(self.P, axis=1)
        for i in range(len(self.Y)):
            n_total += 1
            if Y[i] == P[i]:
                n_correct += 1
        return float(n_correct) / n_total

    @staticmethod
    def cost(T, Y):
        tot = T * np.log(Y)
        return (-1) * tot.sum()

    def derivative_w3(self):
        P, T, Z1, Z2, W2, W3 = self.P, self.T, self.Z1, self.Z2, self.W2, self.W3
        res = Z2.T.dot(T - P)
        return res

    def derivative_b3(self):
        P, T, Z1, Z2, W2, W3 = self.P, self.T, self.Z1, self.Z2, self.W2, self.W3
        return (T - P).sum(axis=0)

    def derivative_w2(self):
        P, T, Z1, Z2, W2, W3 = self.P, self.T, self.Z1, self.Z2, self.W2, self.W3
        dZ2 = (T - P).dot(W3.T) * Z2 * (1 - Z2)
        res = Z1.T.dot(dZ2)
        return res

    def derivative_b2(self):
        P, T, Z1, Z2, W2, W3 = self.P, self.T, self.Z1, self.Z2, self.W2, self.W3
        res = (T - P).dot(W3.T) * Z2 * (1 - Z2)
        return res.sum(axis=0)

    def derivative_w1(self):
        P, T, Z1, Z2, W2, W3, X = self.P, self.T, self.Z1, self.Z2, self.W2, self.W3, self.X
        dZ1 = ((T - P).dot(W3.T) * Z2 * (1 - Z2)).dot(W2.T) * Z1 * (1 - Z1)
        res = X.T.dot(dZ1)
        return res

    def derivative_b1(self):
        P, T, Z1, Z2, W2, W3 = self.P, self.T, self.Z1, self.Z2, self.W2, self.W3
        res = ((T - P).dot(W3.T) * Z2 * (1 - Z2)).dot(W2.T) * Z1 * (1 - Z1)
        return res.sum(axis=0)

    def fit(self, nb_epoch, learning_rate=0.001, plot_cost=True):

        costs = []
        T = self.preproc()

        for epoch in range(nb_epoch):
            output = self.predict_proba(self.X)

            if epoch % 10 == 0:
                c = self.cost(T, output)
                accuracy = self.clf_accuracy()
                print("epoch:", epoch, "cost:", c, "classification_rate:", accuracy)
                costs.append(c)

            W1 = self.W1
            b1 = self.b1
            W2 = self.W2
            b2 = self.b2
            W3 = self.W3
            b3 = self.b3

            W3 += learning_rate * self.derivative_w3()
            b3 += learning_rate * self.derivative_b3()
            W2 += learning_rate * self.derivative_w2()
            b2 += learning_rate * self.derivative_b2()
            W1 += learning_rate * self.derivative_w1()
            b1 += learning_rate * self.derivative_b1()

            self.W1 = W1
            self.b1 = b1
            self.W2 = W2
            self.b2 = b2
            self.W3 = W3
            self.b3 = b3

        if plot_cost == True:
            plt.plot(costs)
            plt.title('Cost function')
            plt.xlabel('epoch')
            plt.ylabel('J')
            plt.show(block=False)


model = myNN(X, Y, 5, 5)

model.fit(1000)

pred = model.predict_proba(X)

model.clf_accuracy() # accuracy 0.972











