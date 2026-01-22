# --- imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import seaborn as sns
from tqdm import tqdm

# --------------------------------------------------------------------------------------------------------
# Where is the trainingVal and Test: data and labels
imageFolder = "/home/npurd/School/trainingData1/trainingData/images"
labelsTrainVal = "/home/npurd/School/trainingData1/trainingData/annotations/trainval.txt"
labelsTest = "/home/npurd/School/trainingData1/trainingData/annotations/test.txt"

# set up some of the NN parameters
numEpochs = 30
learningRate = 0.0001
batch_size = 16
image_size = (240, 240)

# what are we classfiying?
class_names = ["Cat", "Dog"]
# cpu?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# -------------------------------
# Load the labels so we can organize what data we are loading in where and when 
labelsTrainVal = pd.read_csv(labelsTrainVal, sep=" ")
labelsTest = pd.read_csv(labelsTest, sep=" ")
# Combine the training and testing labels
labelsTrainVal["dataPurpose"] = "train/val"
labelsTest["dataPurpose"] = "test"
allLabels = pd.concat([labelsTrainVal, labelsTest], ignore_index=True)

# -------------------------------
# made a dataset generator here because my RAM is limited 
class ImageDataset(Dataset):
    # used to load in the data from a folder on the fly during training,
    # as to avoid loading in thousands of imaiges at once (slow!!)
    def __init__(self, dataframe, imageFolder, image_size, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.imageFolder = imageFolder
        self.image_size = image_size # nxm size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # use the image file name to correlate the images to class labels from a text file of row data 
        row = self.df.iloc[idx]
        img_path = os.path.join(self.imageFolder, f"{row['imageFile']}.jpg")

        with Image.open(img_path) as img:
            img = img.resize(self.image_size).convert("RGB")

            if self.augment and np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            img = np.array(img, dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # PIL (H,W,C) → convert to PYTORCH format (C,H,W) --> [column height widht]

        label = row["SpeciesID"] - 1
        return torch.tensor(img), torch.tensor(label, dtype=torch.long)

# -------------------------------
# --- Data Splits ---
# -------------------------------
test_df = allLabels[allLabels["dataPurpose"] == "test"].reset_index(drop=True)
train_val_df = allLabels[allLabels["dataPurpose"] == "train/val"].reset_index(drop=True)

stealFromTest = test_df.sample(frac=0.6, random_state=42)
test_df = test_df.drop(stealFromTest.index).reset_index(drop=True)
train_val_df = pd.concat([train_val_df, stealFromTest], ignore_index=True)

val_size = int(0.2 * len(train_val_df))
validation_df = train_val_df.iloc[:val_size]
training_df = train_val_df.iloc[val_size:]

# -------------------------------
# --- DataLoaders ---
# -------------------------------
train_loader = DataLoader(
    ImageDataset(training_df, imageFolder, image_size, augment=False),
    batch_size=batch_size, shuffle=True
)

val_loader = DataLoader(
    ImageDataset(validation_df, imageFolder, image_size),
    batch_size=batch_size, shuffle=False
)

test_loader = DataLoader(
    ImageDataset(test_df, imageFolder, image_size),
    batch_size=batch_size, shuffle=False
)

# -------------------------------
# --- CNN Model ---
# -------------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 30 * 30, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learningRate)

# -------------------------------
# --- Training Loop (Keras-like) ---
# -------------------------------
history = {
    "loss": [],
    "accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

for epoch in range(numEpochs):
    # ---- Training ----
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{numEpochs}", leave=False)

    for x, y in train_bar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        train_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct/total:.4f}"
        })

    train_loss = running_loss / total
    train_acc = correct / total

    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            val_loss += loss.item() * y.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    history["loss"].append(train_loss)
    history["accuracy"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)

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
plt.ylim(0,1)
plt.show()

# -------------------------------
# --- Confusion Matrix ---
# -------------------------------
conf_matrix = np.zeros((2,2), dtype=int)
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
