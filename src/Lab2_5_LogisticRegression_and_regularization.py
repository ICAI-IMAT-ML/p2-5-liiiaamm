import numpy as np

class LogisticRegressor:
    def __init__(self):
        """
        Initializes the Logistic Regressor model.

        Attributes:
        - weights (np.ndarray): A placeholder for the weights of the model.
                                These will be initialized in the training phase.
        - bias (float): A placeholder for the bias of the model.
                        This will also be initialized in the training phase.
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

        This method initializes the model's weights and bias, then iteratively updates these parameters by
        moving in the direction of the negative gradient of the loss function (computed using the
        log_likelihood method).

        The regularization terms are added to the gradient of the loss function as follows:

        - No regularization: The standard gradient descent updates are applied without any modification.

        - L1 (Lasso) regularization: Adds a term to the gradient that penalizes the absolute value of
            the weights, encouraging sparsity.

        - L2 (Ridge) regularization: Adds a term to the gradient that penalizes the square of the weights,
            discouraging large weights.

        - ElasticNet regularization: Combines L1 and L2 penalties.

        Parameters:
        - X (np.ndarray): The input features, with shape (m, n), where m is the number of examples and n is
                            the number of features.
        - y (np.ndarray): The true labels of the data, with shape (m,).
        - learning_rate (float): The step size at each iteration while moving toward a minimum of the
                            loss function.
        - num_iterations (int): The number of iterations for which the optimization algorithm should run.
        - penalty (str): Type of regularization (None, 'lasso', 'ridge', 'elasticnet'). Default is None.
        - l1_ratio (float): The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        - C (float): Inverse of regularization strength; must be a positive float.
        - verbose (bool): Print loss every print_every iterations.
        - print_every (int): Period of number of iterations to show the loss.
        """
        m, n = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n)
        self.bias = 0

        for i in range(num_iterations):
            # Compute logits and probabilities
            z = np.dot(X, self.weights) + self.bias
            y_hat = self.sigmoid(z)

            # Compute loss
            loss = self.log_likelihood(y, y_hat)
            
            if verbose and i % print_every == 0:
                print(f"Iteration {i}: Loss {loss}")

            # Compute gradients
            error = y_hat - y
            dw = (1/m) * np.dot(X.T, error)
            db = (1/m) * np.sum(error)

            # Apply regularization
            if penalty == "lasso":
                dw = self.lasso_regularization(dw, C, m)
            elif penalty == "ridge":
                dw = self.ridge_regularization(dw, C, m)
            elif penalty == "elasticnet":
                dw = self.elasticnet_regularization(dw, C, m, l1_ratio)

            # Update weights and bias
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict_proba(self, X):
        """
        Predicts probability estimates for the positive class for each sample in X.
        """
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
