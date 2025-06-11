import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ------------------------- Distribution Classes -------------------------

class PoissonGLM:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1)
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.epochs):
            linear_pred = np.clip(X @ self.weights, -10, 10)
            y_hat = np.exp(linear_pred)
            gradient = X.T @ (y_hat - y)
            self.weights -= self.lr * gradient

    def predict(self, X):
        return np.exp(np.clip(X @ self.weights, -10, 10))


class NormalGLM:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        return X @ self.weights


class GammaGLM:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.epochs):
            eta = np.clip(X @ self.weights, -10, 10)
            mu = np.exp(eta)
            gradient = X.T @ (1 - y / mu)
            self.weights -= self.lr * gradient

    def predict(self, X):
        return np.exp(np.clip(X @ self.weights, -10, 10))


# ---------------------- Data Generation -------------------------

def generate_data(n_samples, dist, true_weights):
    X = np.random.randn(n_samples, len(true_weights) - 1)
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # add intercept term
    eta = X @ true_weights

    if dist == "Normal":
        y = eta + np.random.normal(0, 1, n_samples)
    elif dist == "Poisson":
        mu = np.exp(eta)
        y = np.random.poisson(mu)
    elif dist == "Gamma":
        mu = np.exp(eta)
        shape = 2.0
        scale = mu / shape
        y = np.random.gamma(shape, scale)
    else:
        raise ValueError("Unsupported distribution")

    return X, y


# ---------------------- Streamlit App -------------------------

st.title("GLM Parameter Estimation with MLE")

n_samples = st.slider("Select number of samples", min_value=10, max_value=1000, value=100)
lr = st.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=1e-3, step=1e-4, format="%f")
epochs = st.slider("Epochs", min_value=100, max_value=5000, value=1000)
dist_choice = st.selectbox("Choose distribution to sample from", ["Normal", "Poisson", "Gamma"])
model_choice = st.selectbox("Choose model to estimate parameters", ["Normal", "Poisson", "Gamma"])

if st.button("Run Simulation"):
    st.subheader("Running...")

    true_weights = np.array([2.0, 1.5, -1.0, 0.5, 0.0, 0.7])  # includes intercept
    X, y = generate_data(n_samples, dist_choice, true_weights)

    # Display generated data (excluding intercept)
    df = pd.DataFrame(X[:, 1:], columns=[f"x{i}" for i in range(1, X.shape[1])])
    df["y"] = y
    st.write("### Generated Dataset")
    st.dataframe(df)

    # Choose model
    if model_choice == "Normal":
        model = NormalGLM()
    elif model_choice == "Poisson":
        model = PoissonGLM(lr=lr, epochs=epochs)
    elif model_choice == "Gamma":
        model = GammaGLM(lr=lr, epochs=epochs)
    else:
        raise ValueError("Unsupported model")

    model.fit(X, y)
    y_pred = model.predict(X)
    estimated_weights = model.weights

    # Display weights in vector format
    st.write("### True Weights")
    st.code("[" + ", ".join([f"{w:.4f}" for w in true_weights]) + "]")

    st.write("### Estimated Weights")
    st.code("[" + ", ".join([f"{w:.4f}" for w in estimated_weights]) + "]")

    # Compute Euclidean distance
    distance = np.linalg.norm(true_weights - estimated_weights)
    st.metric("Euclidean Distance Between Weights", round(distance, 4))

    # R^2 Score
    r2 = r2_score(y, y_pred)
    st.metric("R^2 Score", round(r2, 4))

# ---------------------- Distance vs Sample Size Plot -------------------------

st.subheader("Effect of Sample Size on Estimation Accuracy")
n_min = st.number_input("Start of sample size range", min_value=10, max_value=900, value=50)
n_max = st.number_input("End of sample size range", min_value=20, max_value=1000, value=500)
step = st.number_input("Step size", min_value=5, max_value=200, value=50)

if st.button("Plot Distance vs Sample Size"):
    true_weights = np.array([2.0, 1.5, -1.0, 0.5, 0.0, 0.7])
    sample_sizes = list(range(n_min, n_max + 1, step))
    distances = []

    for n in sample_sizes:
        X, y = generate_data(n, dist_choice, true_weights)

        if model_choice == "Normal":
            model = NormalGLM()
        elif model_choice == "Poisson":
            model = PoissonGLM(lr=lr, epochs=epochs)
        elif model_choice == "Gamma":
            model = GammaGLM(lr=lr, epochs=epochs)

        model.fit(X, y)
        est_weights = model.weights
        dist = np.linalg.norm(est_weights - true_weights)
        distances.append(dist)

    fig, ax = plt.subplots()
    ax.plot(sample_sizes, distances, marker='o')
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Distance Between Weights")
    ax.set_title("Convergence of Estimated Weights with Sample Size")
    st.pyplot(fig)
