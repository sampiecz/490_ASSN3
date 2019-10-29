# Imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split

# Configuration options
num_observations = 10000
x1 = np.random.multivariate_normal([0,0], [[0.1, 0.2], [0.2, 0.1]], num_observations)
x2 = np.random.multivariate_normal([3,3], [[0.1, 0.2], [0.2, 0.1]], num_observations)
x3 = np.random.multivariate_normal([3,0], [[0.1, 0.2], [0.2, 0.1]], num_observations)
x4 = np.random.multivariate_normal([0,3], [[0.1, 0.2], [0.2, 0.1]], num_observations)
features = np.vstack((x1, x2, x3, x4)).astype(np.float32)
labels = np.hstack((np.full(num_observations, 0), np.full(num_observations,1), np.full(2*num_observations, 2)))
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5)
model.evaluate(X_test, y_test, verbose=2)

# Plot decision boundary
plot_decision_regions(X_train, y_train, clf=model, legend=2)
plt.show()
