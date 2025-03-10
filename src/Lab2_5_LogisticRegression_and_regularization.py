import numpy as np

class LogisticRegressor:
    def __init__(self):
        """
        Initializes the Logistic Regressor model.

        Attributes:
        - weights (np.ndarray): A placeholder for the weights of the model.
        - bias (float): A placeholder for the bias of the model.
        """
        self.weights = None
        self.bias = None

    def fit(
        self,
        X,
        y,
        learning_rate=0.01,
        num_iterations=1000,
        penalty=None,
        l1_ratio=0.5,
        C=1.0,
        verbose=False,
        print_every=100,
    ):
        """
        Fits the logistic regression model to the data using gradient descent.
        """
        m, n = X.shape  # m: número de muestras, n: número de features

        # 1) Inicializamos pesos y bias en cero
        self.weights = np.zeros(n)
        self.bias = 0.0

        # 2) Iteramos en gradient descent
        for i in range(num_iterations):
            # a) Cálculo del logit y de la probabilidad
            z = np.dot(X, self.weights) + self.bias
            y_hat = self.sigmoid(z)

            # b) Cálculo de la función de pérdida (neg log-likelihood)
            loss = self.log_likelihood(y, y_hat)

            # Verbose
            if verbose and i % print_every == 0:
                print(f"Iteration {i}: Loss {loss}")

            # c) Gradientes sin regularización
            error = y_hat - y
            dw = (1/m) * np.dot(X.T, error)
            db = (1/m) * np.sum(error)

            # d) Añadimos la contribución de la regularización (solo a dw)
            if penalty == "lasso":
                dw = self.lasso_regularization(dw, m, C)
            elif penalty == "ridge":
                dw = self.ridge_regularization(dw, m, C)
            elif penalty == "elasticnet":
                dw = self.elasticnet_regularization(dw, m, C, l1_ratio)

            # e) Actualización de parámetros
            self.weights -= learning_rate * dw
            self.bias   -= learning_rate * db

    def predict_proba(self, X):
        """
        Predicts probability estimates for the positive class for each sample in X.
        """
        # Forma 1: Cálculo directo
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)


    def predict(self, X, threshold=0.5):
        """
        Predicts class labels for samples in X given a threshold.
        """
        proba = self.predict_proba(X)
        return np.array([1 if p >= threshold else 0 for p in proba])

    def lasso_regularization(self, dw, m, C):
        """
        L1 (Lasso) regularization update for dw.
        """
        return dw + (C / m) * np.sign(self.weights)

    def ridge_regularization(self, dw, m, C):
        """
        L2 (Ridge) regularization update for dw.
        """
        return dw + (C / m) * self.weights

    def elasticnet_regularization(self, dw, m, C, l1_ratio):
        """
        Elastic Net regularization update for dw.
        """
        dw_l1 = (C / m) * np.sign(self.weights)
        dw_l2 = (C / m) * self.weights
        return dw + l1_ratio * dw_l1 + (1 - l1_ratio) * dw_l2

    @staticmethod
    def log_likelihood(y, y_hat):
        """
        Computes the negative log-likelihood (cross-entropy) loss.
        """
        m = y.shape[0]
        # Evitar log(0) con un pequeño epsilon
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return -(1/m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    @staticmethod
    def sigmoid(z):
        """
        Computes the sigmoid of z.
        """
        return 1 / (1 + np.exp(-z))
