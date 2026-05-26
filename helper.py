import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import io
import mlflow
import mlflow.pytorch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

def plot_to_tensorboard(fig, writer, tag, step):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    image = np.array(image)
    image = torch.tensor(image).permute(2, 0, 1) / 255.0
    writer.add_image(tag, image, global_step=step)
    plt.close(fig)

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MLPClassifier(nn.Module):
    def __init__(self, input_size=64*64*3, hidden1=512, hidden2=128, dropout=0.0, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),

            # --- CAPA 1 ---
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            # --- CAPA 2 ---
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),

            # --- SALIDA ---
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class CNNClassifier(nn.Module):
    def __init__(self, input_size, dropout=0.0, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, padding_mode="reflect"),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1, padding_mode="reflect"),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear((input_size//4)**2*32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)