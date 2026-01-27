import pickle
import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import pandas as pd


# some important high level parameters...
numEpochs = 30
learningRate = 0.001
batch_size = 16
image_size = (32, 32)   # 32x32 colour image
augmentTrainData = True
testName = "basicDeeperTest"


# model is defined here
class CNN(nn.Module):
    def __init__(self, input_shape=(3, image_size[0], image_size[1]), num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding='same'),  # padding=1 keeps size for 3x3 kernel
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),  # halves spatial dimensions

            nn.Conv2d(8, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # halves again

            nn.Conv2d(16, 32, kernel_size=2, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # halves again
        )

        # Compute flattened size properly
        C, H, W = input_shape
        numPoolingLayers = 3 # count them above
        H = int(H /(2**numPoolingLayers)) # 3 MaxPool layers, each halves H
        W = int(W / (2**numPoolingLayers)) # 3 MaxPool layers, each halves W

        flattened_size = 32 * H * W

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class DeeperCNN(nn.Module):
    def __init__(self, input_shape=(3,32,32), num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 8x8 -> 4x4
        )

        flattened_size = 128 * 4 * 4  # last block channels * spatial dims

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# data set loading by batch routine, even though the data is already all loaded in as a binary dict, this will need to be used for training on images with higher resolution
class CIFARDataset(Dataset):
    def __init__(self, data_dict, augment=False, randSeed=42):
        """
        data_dict must contain:
          - b'data': numpy array (N, 3072) or (N, 3, 32, 32)
          - b'labels': list or array (N,)
        """
        self.X = data_dict[b'data']
        self.y = np.array(data_dict[b'labels'], dtype=np.int64)
        self.augment = augment
        self.rngForAugs = np.random.RandomState(randSeed)

        # reshape once if needed
        if self.X.ndim == 2:
            self.X = self.X.reshape(-1, 3, 32, 32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx].astype(np.float32) / 255.0
        label = self.y[idx]

        # simple augmentation (horizontal flip)
        if self.augment and self.rngForAugs.rand() > 0.5:

            img = img[:, :, ::-1].copy()  # flip width axis
            # perform more complicated augmentations here if you want to get better performance

        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)

"""
The purpose of this script is to load/understand the CIFAR classification data set (60000 low res images)
and then use a common architecture (used in dog_classifier_mark1_pytorch.py) but alter it for 10 classes instead of just dog or cat.
architecture: deep CNN (but not tuned or anything fancy, just implementing basic image classification pipeline, maybe some data augmentation on the fly)
INPUT DATA: BINARY files conatining batch splits of the data (should be roughly randomized across classes per batch)
- 32x32 colour image
-  The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. 
- The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
"""

# function kto open CIFAR files (python version of data download) - this was form the website
def unpickle(file):
    """# Loaded in this way, each of the batch files contains a dictionary with the following elements:
    # data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    # labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
    """

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

################################################# main routine here... ###############################################

# cpu?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)






# where is the data located (its binary format like)
dataFolder = "/mnt/d/SCHOOL_crap/ece_523/sandbox/dataSets/CIFAR/cifar-10-batches-py/"
classDefinitions = dataFolder + 'batches.meta'
dataBatch1 = dataFolder + "data_batch_1"  # can load them in individually to have partitioned data (train on batches 1 through 5, test on test batch)
dataBatch2 = dataFolder + "data_batch_2"
dataBatch3 = dataFolder + "data_batch_3"
dataBatch4 = dataFolder + "data_batch_4"
dataBatch5 = dataFolder + "data_batch_5" # use for validation during training
testBatch = dataFolder + "test_batch"

# generate the data containing dictionaries 
batch1_dict = unpickle(dataBatch1)
batch2_dict = unpickle(dataBatch2)
batch3_dict = unpickle(dataBatch3)
batch4_dict = unpickle(dataBatch4)
batch5_dict = unpickle(dataBatch5)
testBatch_dict = unpickle(testBatch)

#keys of the dictionary thats loaded in
DATA_KEY = b'data'
LABEL_KEY = b'labels'
# combine the training data into one massive dict
combinedForTraining = {}
forTesting = {}
forValidation = {}

# concatenate image data
combinedForTraining[b'data'] = np.concatenate(
    (
        batch1_dict[b'data'],
        batch2_dict[b'data'],
        batch3_dict[b'data'],
        batch4_dict[b'data'],
    ),
    axis=0
)

# concatenate labels (lists)
combinedForTraining[b'labels'] = (
    batch1_dict[b'labels'] +
    batch2_dict[b'labels'] +
    batch3_dict[b'labels'] +
    batch4_dict[b'labels'] 
)



# for validation 
forValidation[b'labels'] = batch5_dict[b'labels']
forValidation[b'data'] = batch5_dict[b'data']

# also do the same thing for clarity with the test data
forTesting[b'labels'] = testBatch_dict[b'labels']
forTesting[b'data'] = testBatch_dict[b'data']

# print(combinedForTraining)
# print(forTesting)


# define the classes of objects that are in the images (can also be loaded in from)
meta = unpickle(classDefinitions)
meta_decoded = {
    key.decode("utf-8"): value
    for key, value in meta.items()
}

# Decode class names too
meta_decoded["label_names"] = [
    name.decode("utf-8") for name in meta_decoded["label_names"]
]
label_names = meta_decoded["label_names"]

class_id_to_name = {
    i: name for i, name in enumerate(label_names)
}

# print(class_id_to_name)

# okay now we have training data and labels.... SET UP THE MODEL!!!
model = DeeperCNN().to(device) 
criterion = nn.CrossEntropyLoss() # for binary classification (ex: dog or cat)
optimizer = optim.Adam(model.parameters(), lr=learningRate) # seems okay

# -------------------------------
# --- Training Loop setup  --- ----------------------------------------------------------------------------------
history = {
    "loss": [],
    "accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

# intitalize classes for training and testing
train_dataset = CIFARDataset(combinedForTraining, augment=True)
val_dataset   = CIFARDataset(forValidation, augment=False)
test_dataset = CIFARDataset(forTesting, augment=False)
batchSize = 16 # how many images to load at a time
# training data
train_loader = DataLoader(train_dataset,batch_size=batchSize, shuffle=True, num_workers=2)
# validation data
val_loader = DataLoader(val_dataset,batch_size=batchSize,shuffle=False,num_workers=2)
# testing data
test_loader = DataLoader(test_dataset,batch_size=batchSize,shuffle=False,num_workers=2)


########### DO TRAINING HERE ##############
for epoch in range(numEpochs):
    #  Training 
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # loading bar returns data (isnt that nice)
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{numEpochs}", leave=False)

    # go through the batches of data
    for images, labels in train_bar:
        # load in batch of images and labels, save to computer mem
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() # zero out gradient for each pass
        outputs = model(images) # pass a batch of images through model
        loss = criterion(outputs, labels) #compute loss given loss functiion
        loss.backward() # backward pass
        optimizer.step() # adjust model parameters using computed gradients
        running_loss += loss.item() * labels.size(0) # compute contribution to epoch loss from this single batch
        preds = outputs.argmax(dim=1) # determine predicted classes
        correct += (preds == labels).sum().item() # how many of them are correct
        total += labels.size(0) # running total of samples passed

        # update training bar after batch is sent through
        train_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct/total:.4f}"
        })

    train_loss = running_loss / total
    train_acc = correct / total

    # Execute the validation phase of the training process
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad(): # not updating model using validation data...
        for images, labels in val_loader:
            # laod in batch of validation images for validation
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    history["loss"].append(train_loss)
    history["accuracy"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)

    # update training progress like keras does
    print(
        f"Epoch {epoch+1}/{numEpochs} "
        f"- loss: {train_loss:.4f} "
        f"- acc: {train_acc:.4f} "
        f"- val_loss: {val_loss:.4f} "
        f"- val_acc: {val_acc:.4f}"
    )

# -------------------------------
# --- Test Evaluation ---
# -------------------------------
model.eval()
preds_all, labels_all = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        outputs = model(x)
        preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_all.extend(y.numpy())

preds_all = np.array(preds_all)
labels_all = np.array(labels_all)

test_accuracy = (preds_all == labels_all).mean()
print(f"Test accuracy: {test_accuracy:.4f}")

# -------------------------------
# --- Plot History ---
# -------------------------------
history_df = pd.DataFrame(history)
history_df.plot(figsize=(8,5))
plt.grid(True)
plt.ylim(None)
plt.show()
history_df.to_csv(f"CIFAR_classifierMark1_{testName}_{numEpochs}epochs_{np.round(test_accuracy,2)}acc")

# -------------------------------
# --- Confusion Matrix ---
# -------------------------------
conf_matrix = np.zeros((num_classes,num_classes), dtype=int)
for t, p in zip(labels_all, preds_all):
    conf_matrix[t, p] += 1

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


