#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset, random_split
from tqdm import tqdm

DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = r'splited/train'

BATCH_SIZE = 40
N_EPOCH = 50
TRAIN_RATIO = 0.9
PATIENCE = 4
LR = 0.0002
WEIGHT_DECAY = 1.0e-5

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    # Ignore .DS_STORE
    if len(os.listdir(save_dir)) <= 1:
        train_split(data_folder, save_dir)

    batch_size = BATCH_SIZE
    n_epoch = N_EPOCH
    train_ratio = TRAIN_RATIO
    patience = PATIENCE

    net = NeuralNetwork()
    device = DEVICE_NAME
    print(device)

    net = net.to(device)

    train_data = ICareDataset(save_dir)
    train_amount = int(len(train_data) * train_ratio)
    val_amount = len(train_data) - train_amount
    train_set, valid_set = torch.utils.data.random_split(train_data, [train_amount, val_amount])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6)
    valid_loader = DataLoader(valid_set, shuffle=True, num_workers=6)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_epoch * len(train_loader))

    train_losses = []
    val_losses = []
    counter = 0
    best_loss = None

    for epoch in range(n_epoch):
        net.train()
        loss_record = []
        avg_loss = 0

        train_pbar = tqdm(train_loader, position=0, leave=True)
        train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epoch}]')
        train_pbar.set_postfix({'loss': "?", "train_loss": "?"})

        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = net(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach().item())
            mean_train_loss = sum(loss_record) / len(loss_record)
            train_pbar.set_postfix({'loss': loss.detach().item(), "train_loss": mean_train_loss})
            avg_loss = mean_train_loss
            # print(f'Epoch [{epoch + 1}/{n_epoch}]: Train loss: {mean_train_loss:.4f}')
        scheduler.step()
        train_losses.append(avg_loss)

        net.eval()
        loss_record = []
        valid_pbar = tqdm(valid_loader, position=0, leave=True)
        valid_pbar.set_description("Validation:")
        valid_pbar.set_postfix({"valid_loss": "?"})
        for x, y in valid_pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = net(x)
                loss = criterion(pred, y)
            loss_record.append(loss.item())
            mean_valid_loss = sum(loss_record) / len(loss_record)
            valid_pbar.set_postfix({"valid_loss": mean_valid_loss})

        val_losses.append(mean_valid_loss)

        if best_loss is None:
            best_loss = mean_valid_loss
            create_dir_not_exists(model_folder)
            torch.save(net.state_dict(), os.path.join(model_folder, "model.pth"))
            print("Saving model to {0}...".format(model_folder))
        elif mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(net.state_dict(), os.path.join(model_folder, "model.pth"))
            print("Saving model to {0}...".format(model_folder))
            counter = 0
        else:
            counter += 1
        if counter > patience:
            print('\nModel is not improving, so we halt the training session.')
            break

    return train_losses, val_losses


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    model = NeuralNetwork()
    model.load_state_dict(torch.load(os.path.join(model_folder, "model.pth")))
    device = DEVICE_NAME
    model.to(device)
    return model.eval()


# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    signals = test_split(data_folder, patient_id=patient_id)

    if len(signals) == 0:
        outcome, outcome_probability, cpc = 1, 0.667, 4
        return outcome, outcome_probability, cpc

    signals = np.array(signals)
    signals = torch.FloatTensor(signals)
    test_dataset = Dataset_test(signals)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    pred = []
    device = DEVICE_NAME

    for item in test_loader:
        with torch.no_grad():
            pred.append(models(item.to(device))[0])

    pred_mean = sum(pred) / len(pred)
    pred_mean = [float(pred_mean[0]), float(pred_mean[1])]
    outcome = 0 if pred_mean[0] > pred_mean[1] else 1
    outcome_probability = pred_mean[1]
    cpc = 1 + 1 * (1 - pred_mean[0]) if outcome == 0 else 2 + 3 * pred_mean[1]

    return outcome, outcome_probability, cpc


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def create_dir_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


# Data preprocessing.
def train_split(path, save_dir=None):
    count = 0
    patients = find_data_folders(path)

    for item in patients:
        patient_metadata, recording_metadata, recording_data = load_challenge_data(path, item)
        channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                    'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
        current_cpc = get_cpc(patient_metadata)

        for i in range(72):
            signal_data, sampling_frequency, signal_channels = recording_data[i]
            if signal_data is not None:
                signal_data = reorder_recording_channels(signal_data, signal_channels,
                                                         channels)
                signal_data = [signal for signal in signal_data]
                signal_data = np.array(signal_data).transpose((1, 0))
                signal_data = np.array_split(signal_data, 30)
                for j in range(len(signal_data)):
                    np.save(os.path.join(save_dir, f"{count}_{current_cpc}_signal.npy"), signal_data[j])
                    count += 1
        print("Dataset {0} is done!".format(item))
    print("All is done!")


def test_split(path, patient_id=None):
    count = 0
    patients = [patient_id]
    signals = []
    for item in patients:
        patient_metadata, recording_metadata, recording_data = load_challenge_data(path, item)
        channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                    'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
        for record in recording_data:
            signal_data, sampling_frequency, signal_channels = record
            if signal_data is not None:
                signal_data = reorder_recording_channels(signal_data, signal_channels,
                                                         channels)
                signal_data = [signal for signal in signal_data]
                signal_data = np.array(signal_data).transpose((1, 0))
                signal_data = np.array_split(signal_data, 30)
                for j in range(len(signal_data)):
                    signals.append([signal_data[j].transpose(1, 0)])
                    count += 1
        print("Dataset {0} is done!".format(item))

    return signals


# Dataset
class ICareDataset(Dataset):
    def __init__(self, path):
        super(ICareDataset, self).__init__()
        self.path = path
        # self.fileList = os.listdir(path)
        self.fileList = [f for f in os.listdir(path) if not f.startswith('.')]

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, idx):
        fname = self.fileList[idx]
        fullpath = os.path.join(self.path, fname)
        signal = np.load(fullpath)
        cpc = float(fname.split('_')[1])

        outcome = [1, 0] if cpc <= 2 else [0, 1]

        cpc = torch.FloatTensor([cpc])
        outcome = torch.FloatTensor(outcome)
        signal = signal.transpose((1, 0))
        signal = torch.FloatTensor(signal)
        signal = signal.reshape((1, signal.shape[0], -1))
        return signal, outcome


class Dataset_test(Dataset):
    def __init__(self, signals):
        super(Dataset_test, self).__init__()
        self.signals = signals

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, item):
        return self.signals[item]


# Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 64, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.permute(0, 1, 3, 2)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    pass
