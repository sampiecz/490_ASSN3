from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.cm import get_cmap
import tensorflow as tf
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions

def plot_decision_boundary(prediction_model, X, Y):
    # Plot the decision boundary
    # Determine grid range in x and y directions
    x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
    y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1

    # Set grid spacing parameter
    spacing = min(x_max - x_min, y_max - y_min) / 100

    # Create grid
    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                   np.arange(y_min, y_max, spacing))
    
    # Concatenate data to match input
    data = np.hstack((XX.ravel().reshape(-1,1), 
                      YY.ravel().reshape(-1,1)))
    # Get decision boundary probabilities
    db_prob = prediction_model.predict(data)
    
    # Convert probabilities to classes
    clf = np.where(db_prob<0.5,0,1)
    
    Z = clf.reshape(XX.shape[0], XX.shape[1])

    plt.figure(figsize=(10,8))
    plt.contourf(XX, YY, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=Y, 
                cmap=plt.cm.Spectral)
    plt.show()

# D
num_observations = 1000
x1 = np.random.multivariate_normal([0,0], [[0.1, 0.2], [0.2, 0.1]], num_observations)
x2 = np.random.multivariate_normal([3,3], [[0.1, 0.2], [0.2, 0.1]], num_observations)
x3 = np.random.multivariate_normal([3,0], [[0.1, 0.2], [0.2, 0.1]], num_observations)
x4 = np.random.multivariate_normal([0,3], [[0.1, 0.2], [0.2, 0.1]], num_observations)
features = np.vstack((x1, x2, x3, x4)).astype(np.float32)
labels = np.hstack((np.full(num_observations, 0), np.full(num_observations,1), np.full(2*num_observations, 2)))
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)


# XOR data vis.  
#fig, ax = plt.subplots()
#ax.set_ylabel("Features")
#ax.set_xlabel("Labels")
#scatter = ax.scatter(features[5000:, 0], features[5000:, 1], c=labels[5000:])
#plt.title("XOR - Features & Labels")
#legend = ax.legend(*scatter.legend_elements(), loc="lower right", title="Classes")
#ax.add_artist(legend)

# Training data.
#fig, ax = plt.subplots()
#ax.set_ylabel("Features")
#ax.set_xlabel("Labels")
#scatter = ax.scatter(X_train[:,0], y_train, c=y_train)
#plt.title("Training Data")
#legend = ax.legend(*scatter.legend_elements(), loc="lower right", title="Classes")
#ax.add_artist(legend)
#plt.show()

print("""\n###########################################
# Model One                               #
###########################################""")

# Build model
feature_vector_shape = len(X_train[0])
input_shape = (feature_vector_shape,)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile, fit, and train the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5)
model.evaluate(X_test, y_test, verbose=2)
plot_decision_boundary(model, features, labels)
#
#print("""\n###########################################
## Model Two                               #
############################################""")
## Build model
#model = tf.keras.models.Sequential([
#    tf.keras.layers.Dense(20, activation="relu"),
#    tf.keras.layers.Dropout(0.1),
#    tf.keras.layers.Dense(30, activation="softmax")
#])
#
## Compile, fit, and train the model
#model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#model.fit(X_train, y_train, epochs=5)
#model.evaluate(X_test, y_test, verbose=2)
#plot_decision_regions(X_test[:,0], y_test, clf=model, legend=2)
#plt.show()
#
#
#print("""\n###########################################
## Model Three                             #
############################################""")
## Build model
#model = tf.keras.models.Sequential([
#    tf.keras.layers.Dense(50, activation="relu"),
#    tf.keras.layers.Dropout(0.8),
#    tf.keras.layers.Dense(50, activation="softmax")
#])
#
## Compile, fit, and train the model
#model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#model.fit(X_train, y_train, epochs=5)
#model.evaluate(X_test, y_test, verbose=2)
#plot_decision_regions(X_test[:,0], y_test, clf=model, legend=2)
#plt.show()
#
#print("""\n###########################################
## Model Four                              #
############################################""")
## Build model
#model = tf.keras.models.Sequential([
#    tf.keras.layers.Dense(50, activation="relu"),
#    tf.keras.layers.Dropout(0.1),
#    tf.keras.layers.Dense(50, activation="softmax")
#])
#
## Compile, fit, and train the model
#model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#model.fit(X_train, y_train, epochs=5)
#model.evaluate(X_test, y_test, verbose=2)
#plot_decision_regions(X_test[:,0], y_test, clf=model, legend=2)
#plt.show()
#
#print("""\n###########################################
## Model Five                              #
############################################""")
## Build model
#model = tf.keras.models.Sequential([
#    tf.keras.layers.Dense(100, activation="relu"),
#    tf.keras.layers.Dropout(0.5),
#    tf.keras.layers.Dense(100, activation="softmax")
#])
#
## Compile, fit, and train the model
#model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#model.fit(X_train, y_train, epochs=5)
#model.evaluate(X_test, y_test, verbose=2)
#plot_decision_regions(X_test[:,0], y_test, clf=model, legend=2)
#plt.show()

