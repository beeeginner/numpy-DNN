# example1: fit an function
import matplotlib.pyplot as plt
import numpy as np
from DeepNueralNetwork import CosineAnnealing,DNN

def function(x):
    return (x ** 2) + 2 * x - 3


def generate_points(interval=[-10, 10], num_points=30000, shuffle=True):
    X = np.linspace(interval[0], interval[1], num_points)
    y = function(X)
    if shuffle:
        indices = np.arange(num_points)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        return X_shuffled, y_shuffled
    else:
        return X, y


activation = ['relu', 'leakyrelu', 'tanh', 'sigmoid']
for act in activation:
    # Initiate
    dnn = DNN(layers=[1, 5, 7, 8, 1], act=act, alpha=0.01, reg=True)
    X, y = generate_points(shuffle=True)
    scheduler = CosineAnnealing(eta_min=1e-5, eta_max=1e-5, T=50)
    epochs, train_loss = dnn.fit(X, y, max_iter=100, scheduler=scheduler, early_stopping_patience=10)
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # Plot loss curve
    X, y = generate_points(shuffle=False)
    axs[0].plot(range(epochs), train_loss)
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title(f'Training Loss Curve ({act})')

    # Plot fitted curve and Original Curve
    y_pred = np.array([dnn.predict(xi) for xi in X])
    axs[1].plot(X, y, label='Original Data')
    axs[1].plot(X, y_pred, label='Fitted Curve', color='red')
    axs[1].legend()
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('y')
    axs[1].set_title(f'Fitted Curve and Original Curve ({act})')

    plt.tight_layout()
    plt.show()