import numpy as np
import matplotlib.pyplot as plt


class NeuralNet:
    """
    A two layer neural network
    """

    # initializing constructor for 'layers', 'learning_rate' and iterations
    def __init__(self, layers=[13, 8, 1], learning_rate=0.01, iterations=100):
        self.params = {}
        self.layers = layers
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.X = None
        self.y = None

    def init_weights(self):
        """
        Initialize the weights and biases from a random normal distribution
        """
        np.random.seed(1)  # generate the same number in each execution
        self.params["W1"] = np.random.rand(self.layers[0], self.layers[1])
        self.params["b1"] = np.random.randn(self.layers[1],)
        self.params["W2"] = np.random.randn(self.layers[1], self.layers[2])
        self.params["b2"] = np.random.randn(self.layers[2])

    # defining activation function
    def relu(self, Z):
        """
        function to get the absolute values by removing the signed number to zero
        [Rectified Linear Units]
        """
        return np.maximum(0, Z)

    def sigmoid(self, Z):
        """
        squashes the any real value integer between zero and one
        """
        return 1 / (1 + np.exp(-Z))

    def dsigmoid(self, Z):
        return Z * (1 - Z)

    # defining loss function
    # categorical cross-entropy loss

    def entropy_loss(self, y, yhat):
        nsample = len(y)
        loss = (
            -1
            / nsample
            * (
                np.sum(
                    np.multiply(np.log(yhat), y)
                    + np.multiply((1 - y), np.log(1 - yhat))
                )
            )
        )
        return loss

    def forward_propagation(self):
        """
        performs the forward propagation
        """
        Z1 = self.X.dot(
            self.params["W1"] + self.params["b1"]
        )  # dot product of input features 'X' and first layer weights 'W1'
        A1 = self.relu(
            Z1
        )  # the output of first layer is passed through the activation function
        Z2 = (
            A1.dot(self.params["W2"]) + self.params["b2"]
        )  # dot product between the o/p of activation function 'A1' and with weight of 2nd layer 'W2' and bias is added
        yhat = self.sigmoid(Z2)  # the final o/p is obtained from sigmoid function

        # calculating the loss
        loss = self.entropy_loss(self.y, yhat)

        # save the calculation parameters
        self.params["Z1"] = Z1
        self.params["Z2"] = Z2
        self.params["A1"] = A1

        return yhat, loss

    def back_propagation(self, yhat):
        """
        computes the derivatives and update the weights and biases
        """

        # defining the Relu derivative
        def dRelu(x):
            x[x <= 0] = 0
            x[x > 0] = 1

            return x

        """
        derivative of categorical cross entropy loss is
        -(y / y_prime) - (1 - y) / (1 - y_prime)
        """
        dl_wrt_yhat = -(np.divide(self.y, yhat)) - np.divide((1 - self.y), (1 - yhat))
        dl_wrt_sig = yhat * (1 - yhat)
        #dl_wrt_sig = dsigmoid(yhat)
        dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_A1 = dl_wrt_z2.dot(self.params["W2"].T)
        dl_wrt_w2 = self.params["A1"].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0)

        dl_wrt_z1 = dl_wrt_A1 * dRelu(self.params["Z1"])
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0)

        # update the weights and bias
        self.params["W1"] = self.params["W1"] - self.learning_rate * dl_wrt_w1
        self.params["W2"] = self.params["W2"] - self.learning_rate * dl_wrt_w2
        self.params["b1"] = self.params["b1"] - self.learning_rate * dl_wrt_b1
        self.params["b2"] = self.params["b2"] - self.learning_rate * dl_wrt_b2

    def fit(self, X, y):
        """
        trains the neural network using the specified data and labels
        """
        self.X = X
        self.y = y

        self.init_weights()  # initialize the weights and biases

        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)
        print(loss)

    def predict(self, X):
        """
        predicts the test data
        """
        Z1 = X.dot(self.params["W1"]) + self.params["b1"]
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params["W2"]) + self.params["b2"]

        pred = self.sigmoid(Z2)

        return np.round(pred)

    def acc(self, y, yhat):
        """
        calculates the accurany between the predicted and the truth labels
        """
        acc = int(sum(y == yhat) / len(y) * 100)
        return acc

    def plot_loss(self):
        """
        plots the loss curve
        """
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("logloss")
        plt.title("Loss curve for training")
        plt.show()
