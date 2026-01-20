# --- imports ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

import time

# overall flow: train using training data and validation data
# -------------------------------
# TensorFlow / Keras workflow
# -------------------------------

# variables
numEpochs = 30
learningRate = 0.009

print(f"Running tensorflow version: {tf.__version__}\nRunning keras version: {keras.__version__}\n")

# Load dataset
fashion_mnist_set = keras.datasets.fashion_mnist
(trainingData_full, trainingLabels_full), (test_data, test_labels) = fashion_mnist_set.load_data()

numTrainingData, rowsPerData, colsPerData = trainingData_full.shape
print(f"There are {numTrainingData} data artifacts, each {rowsPerData}x{colsPerData}.")
print(f"Type of training data X: {trainingData_full.dtype}\n")

# Split validation / training / test (~80-20 ratio seems ballpark okay)
splitRatio = 0.1
splitPoint = int(splitRatio * numTrainingData)
validationData, trainingData = trainingData_full[:splitPoint]/255, trainingData_full[splitPoint:]/255
validationLabels, trainingLabels = trainingLabels_full[:splitPoint], trainingLabels_full[splitPoint:]
test_data = test_data / 255

class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal",
               "Shirt","Sneaker","Bag","Ankle boot"]

# Build  the Model
myModel = keras.models.Sequential([
    Flatten(input_shape=[rowsPerData, colsPerData]),
    Dense(150, activation="relu"),
    Dense(120, activation="relu"),
    Dense(75, activation="relu"),
    Dense(50, activation="relu"),
    Dense(10, activation="softmax")
])

print("Summary of myModel:")
myModel.summary()

myModel.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=learningRate),
    metrics=["accuracy"]
)

# Train myModel
history = myModel.fit(trainingData, trainingLabels, epochs=numEpochs, validation_data=(validationData, validationLabels))

# use the myModel to evaluate on the test set of input data and labels that has not been seen yet,
#and see how the performance is
loss, accuracy = myModel.evaluate(test_data, test_labels)
print(f"-Evaluation Results- \n Accuracy: {accuracy}\n Loss: {loss}\n")

# Convert training history to DataFrame
history_df = pd.DataFrame(history.history)

# Plot accuracy and loss
history_df.plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()  # Now appears directly inside Jupyter


# now use the model to predict the entire test set (hasnt seen it yet)
predictedProbs = myModel.predict(test_data) # has shape of 10000, 10, since 10000 of them, and 10 class labels total

# now for the labels
predictedLabels = np.argmax(predictedProbs, axis=1)

conf_matrix = tf.math.confusion_matrix(test_labels, predictedLabels, num_classes=10).numpy()
print(conf_matrix)

for i, name in enumerate(class_names):
    TP = conf_matrix[i, i]
    FP = conf_matrix[:,i].sum() - TP
    FN = conf_matrix[i,:].sum() - TP
    TN = conf_matrix.sum() - (TP + FP + FN)
    print(f"{name:12s} | TP={TP:4d}, FP={FP:4d}, FN={FN:4d}, TN={TN:5d}")

import seaborn as sns

plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix – Fashion MNIST")
plt.show()


