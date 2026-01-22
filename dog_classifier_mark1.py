# --- imports ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Input
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # this might need to be removed if running on non-WSL and non-virtual python environment...
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from tensorflow.keras.utils import to_categorical
import seaborn as sns

# -------------------------------
# --- Configuration Variables ---
# -------------------------------
imageFolder = "/home/npurd/School/trainingData1/trainingData/images"
labelsTrainVal = "/home/npurd/School/trainingData1/trainingData/annotations/trainval.txt"
labelsTest = "/home/npurd/School/trainingData1/trainingData/annotations/test.txt"

numEpochs = 20
learningRate = 0.0001
batch_size = 16
image_size = (240, 240)  # RGB image size

class_names = ["Cat", "Dog"]  # species mapping

print(f"TensorFlow version: {tf.__version__}, Keras version: {keras.__version__}")

# --- Load Labels ---
labelsTrainVal = pd.read_csv(labelsTrainVal, sep=" ")
labelsTest = pd.read_csv(labelsTest, sep=" ")

# Combine all labels into one dataframe with a marker for train/val or test
labelsTrainVal['dataPurpose'] = 'train/val'
labelsTest['dataPurpose'] = 'test'
allLabels = pd.concat([labelsTrainVal, labelsTest], ignore_index=True)


#  per batch data generator --------------------
# need this special data generator paradigm because my ram is too small to load in a bunch of image data at once
class ImageDataGenerator(keras.utils.Sequence):
    global batch_size
    global image_size

    # load a sinlge batch of images each sweep in the training process (only load what we need when wee need it so this process is not as painfull)
    def __init__(self, dataframe, imageFolder, batch_size=batch_size, image_size=image_size,
                 n_classes=2, shuffle=True, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.imageFolder = imageFolder
        self.batch_size = batch_size
        self.image_size = image_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_data = self.df.iloc[batch_indexes]
        
        X = np.zeros((len(batch_data), *self.image_size, 3), dtype=np.float32)
        y = np.zeros((len(batch_data), self.n_classes), dtype=np.float32)
        
        for i, (_, row) in enumerate(batch_data.iterrows()):
            img_path = os.path.join(self.imageFolder, f"{row['imageFile']}.jpg")
            with Image.open(img_path) as img:
                img = img.resize(self.image_size).convert("RGB")
                
                # Maybe add some image augmentation here at some point to fluff up the data
                if self.augment:
                    if np.random.rand() > 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    # more augmentation could be added here
                    
                X[i] = np.array(img) / 255.0
                
            y[i] = to_categorical(row['SpeciesID'] - 1, num_classes=self.n_classes)
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# --- Split Data using labels only, since the file is very small compared to data file
# Take half of test data and add to train/val for more training data if desired
test_df = allLabels[allLabels['dataPurpose'] == 'test'].reset_index(drop=True)
train_val_df = allLabels[allLabels['dataPurpose'] == 'train/val'].reset_index(drop=True)

stealFromTest = test_df.sample(frac=0.8, random_state=42)
test_df = test_df.drop(stealFromTest.index).reset_index(drop=True)
train_val_df = pd.concat([train_val_df, stealFromTest], ignore_index=True).reset_index(drop=True)

# Further split train/val into validation (~20% of train_val)
val_size = int(0.2 * len(train_val_df))
validation_df = train_val_df.iloc[:val_size]
training_df = train_val_df.iloc[val_size:]


# --- Create Generators for data
train_gen = ImageDataGenerator(training_df, imageFolder, batch_size=batch_size, image_size=image_size, augment=True)
val_gen = ImageDataGenerator(validation_df, imageFolder, batch_size=batch_size, image_size=image_size, augment=False)
test_gen = ImageDataGenerator(test_df, imageFolder, batch_size=batch_size, image_size=image_size, augment=False, shuffle=False)

# -------------------------------------------------------------------------------------------------
# construct the CNN Model 
myModel = keras.models.Sequential([
    Input(shape=(*image_size, 3)),
    
    #small shapes
    Conv2D(8, (4,4), activation="relu", padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    #larger shapes
    Conv2D(16, (3,3), activation="relu", padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    # even larger shapes
    Conv2D(32, (3,3), activation="relu", padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    # Conv2D(16, (5,5), activation="relu", padding='same'),
    # BatchNormalization(),
    # MaxPooling2D(2,2),
    # Conv2D(16, (5,5), activation="relu", padding='same'),
    # BatchNormalization(),
    # MaxPooling2D(2,2),
    
    
    Flatten(),
    Dense(32, activation="relu"),
    Dropout(0.4), # not sure if i need this...
    Dense(2, activation="softmax")
])

myModel.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=learningRate),
    metrics=["accuracy"]
)

myModel.summary()

# -------------------------------
# --- Train Model ---
# -------------------------------
history = myModel.fit(
    train_gen,
    validation_data=val_gen,
    epochs=numEpochs
)

# -------------------------------
# --- Evaluate on Test Data ---
loss, accuracy = myModel.evaluate(test_gen)
print(f"Test accuracy: {accuracy:.4f}, loss: {loss:.4f}")



#  Plot Accuracy & Loss 
history_df = pd.DataFrame(history.history)
history_df.to_csv(f'dogClassifier_TEST_loss{loss}_accuracy{accuracy}_epochs{numEpochs}_learningRate{learningRate}.csv', mode='a')
history_df.plot(figsize=(8,5))
plt.grid(True)
plt.ylim(0,1)
plt.show()

# Confusion Matrix 
pred_probs = myModel.predict(test_gen)
pred_labels = np.argmax(pred_probs, axis=1)

true_labels = []
for _, row in test_df.iterrows():
    true_labels.append(row['SpeciesID'] - 1)
true_labels = np.array(true_labels)

conf_matrix = tf.math.confusion_matrix(true_labels, pred_labels, num_classes=2).numpy()
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()