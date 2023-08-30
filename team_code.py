#!/usr/bin/env python
import time

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import mne
import numpy as np
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset, random_split
from tqdm import tqdm
from threading import Thread

DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = r'split/train'
# SAVE_DIR = r"D:\Data\Dataset\split\train"

# split eeg length
EEG_LENGTH = 1920

BATCH_SIZE = 60

N_EPOCH = 50
TRAIN_RATIO = 0.9
PATIENCE = 4
LR = 0.0002
WEIGHT_DECAY = 1.0e-5

# 115200 = 32 hrs
TIME_LIMIT = 115200

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'eeg'), exist_ok=True)
    # Ignore .DS_STORE
    if len(os.listdir(os.path.join(save_dir, 'eeg'))) <= 1:
        train_split(data_folder, save_dir)

    train_losses = 1.0
    val_losses = 1.0

    os.makedirs(r"model", exist_ok=True)

    train_losses, val_losses = train_eeg(model_folder)

    return train_losses, val_losses


def train_eeg(model_folder):
    work_dir = os.path.join(SAVE_DIR, 'eeg')

    batch_size = BATCH_SIZE
    n_epoch = N_EPOCH
    train_ratio = TRAIN_RATIO
    patience = PATIENCE

    net = NeuralNetwork()
    device = DEVICE_NAME
    print(device)

    net = net.to(device)

    train_data = EEGDatasetTrain(work_dir)
    train_amount = int(len(train_data) * train_ratio)
    val_amount = len(train_data) - train_amount
    train_set, valid_set = torch.utils.data.random_split(train_data, [train_amount, val_amount])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=7)
    valid_loader = DataLoader(valid_set, shuffle=True, num_workers=6)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_epoch * len(train_loader))

    train_losses = []
    val_losses = []
    counter = 0
    best_loss = None

    train_eeg_timer = time.time()

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
            torch.save(net.state_dict(), os.path.join(model_folder, "eeg_model.pth"))
            print("Saving model to {0}...".format(model_folder))
        elif mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(net.state_dict(), os.path.join(model_folder, "eeg_model.pth"))
            print("Saving model to {0}...".format(model_folder))
            counter = 0
        else:
            counter += 1
        total_time = time.time() - train_eeg_timer
        # 115200 = 32 hrs
        if total_time >= TIME_LIMIT:
            print("Time limit is exceeded {0} > {1}".format(total_time, TIME_LIMIT))
            break
        if counter > patience:
            print('\nModel is not improving, so we halt the training session.')
            break

    return train_losses, val_losses


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    model_dict = {}
    device = DEVICE_NAME

    eeg_model = NeuralNetwork()
    eeg_model.load_state_dict(torch.load(os.path.join(model_folder, "eeg_model.pth")))
    eeg_model.to(device)

    model_dict["eeg"] = eeg_model.eval()
    return model_dict


# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    # EEG part
    eeg_signals = test_split_eeg(data_folder, patient_id=patient_id)
    outcome, outcome_probability, cpc = run_eeg_model(models['eeg'], eeg_signals)

    return outcome, outcome_probability, cpc


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def create_dir_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = 128
    else:
        resampling_frequency = 125
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)

    # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data

    return data, resampling_frequency


def train_split_eeg(path, patients, save_dir=None, thread_index=0):
    eeg_length = EEG_LENGTH
    count = 0

    os.makedirs(save_dir, exist_ok=True)

    for item in patients:
        t2_start = time.time()
        # patient_metadata, recording_metadata, recording_data = load_challenge_data(path, item)
        patient_metadata = load_challenge_data(path, item)
        recording_ids = find_recording_files(path, item)
        num_recordings = len(recording_ids)

        # channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
        #             'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
        current_cpc = get_cpc(patient_metadata)

        # Extract EEG features.
        # eeg_channels = ['F3', 'P3', 'F4', 'P4']
        eeg_channels = ['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'T3', 'T4', 'C3',
                        'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']
        # eeg_channels = ['Fp1', 'Fp2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2', 'F3', 'P3', 'F4', 'P4']
        group = 'EEG'

        if num_recordings > 0:
            for recording_id in recording_ids:
                recording_location = os.path.join(path, item, '{}_{}'.format(recording_id, group))
                if os.path.exists(recording_location + '.hea'):
                    data, channels, sampling_frequency = load_recording_data(recording_location)
                    utility_frequency = get_utility_frequency(recording_location + '.hea')
                    if all(channel in channels for channel in eeg_channels):
                        data, channels = reduce_channels(data, channels, eeg_channels)
                        data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                        # data = np.array([data[0, :] - data[2, :],
                        #                  data[2, :] - data[4, :],
                        #                  data[4, :] - data[6, :],
                        #                  data[6, :] - data[8, :],
                        #                  data[10, :] - data[11, :],
                        #                  data[12, :] - data[13, :],
                        #                  data[1, :] - data[3, :],
                        #                  data[3, :] - data[5, :],
                        #                  data[5, :] - data[7, :],
                        #                  data[7, :] - data[9, :]
                        #                  ])
                        # 2 aEEG channels
                        # data = np.array([data[0, :] - data[1, :],
                        #                  data[2, :] - data[3, :]])  # Convert to bipolar montage: F3-P3 and F4-P4
                        # 18 channels
                        data = np.array([data[0, :] - data[2, :],
                                         data[2, :] - data[6, :],
                                         data[6, :] - data[10, :],
                                         data[10, :] - data[14, :],
                                         data[1, :] - data[3, :],
                                         data[3, :] - data[7, :],
                                         data[7, :] - data[11, :],
                                         data[11, :] - data[15, :],
                                         data[0, :] - data[4, :],
                                         data[4, :] - data[8, :],
                                         data[8, :] - data[12, :],
                                         data[12, :] - data[14, :],
                                         data[1, :] - data[5, :],
                                         data[5, :] - data[9, :],
                                         data[9, :] - data[13, :],
                                         data[13, :] - data[15, :],
                                         data[16, :] - data[17, :],
                                         data[17, :] - data[18, :]])

                        # eeg_features = get_eeg_features(data, sampling_frequency).flatten()
                        if data is not None:
                            # print("sampling freq: " + str(sampling_frequency))
                            # Sampling Freq = 128Hz
                            num_split = data.shape[1] // eeg_length
                            if num_split <= 0:
                                continue
                            data = np.hsplit(data, [num_split * eeg_length])[0]
                            split_array = np.asarray(np.hsplit(data, num_split))
                            # slice length = 10s, step = 17 ---> 1 slice per 180s
                            rewindowed_array = split_array[::17, :, :]
                            for slice in rewindowed_array:
                                np.save(os.path.join(save_dir, f"{item}_{count}_{current_cpc}_signal.npy"), slice)
                                count += 1
                        else:
                            print("Dataset {0}_{1} EEG skipped - none data.".format(item, recording_id))
                    else:
                        print("Dataset {0}_{1} EEG skipped - missing channel.".format(item, recording_id))
                else:
                    print("Dataset {0}_{1} EEG skipped - missing header file.".format(item, recording_id))
        else:
            print("Dataset {0} EEG skipped - none recording.".format(item))
            continue
        # for i in range(72):
        #     signal_data, sampling_frequency, signal_channels = recording_data[i]
        #     if signal_data is not None:
        #         signal_data = reorder_recording_channels(signal_data, signal_channels,
        #                                                  channels)
        #         signal_data = [signal for signal in signal_data]
        #         signal_data = np.array(signal_data).transpose((1, 0))
        #         signal_data = np.array_split(signal_data, 30)
        #         for j in range(len(signal_data)):
        #             np.save(os.path.join(save_dir, f"{count}_{current_cpc}_signal.npy"), signal_data[j])
        #             count += 1
        print("Dataset {0} is done!".format(item))
        t2_end = time.time()
        print('Patient time cost: %s s' % ((t2_end - t2_start) * 1))

    print("Thread {0}: All is done!".format(thread_index))


# Data preprocessing.
def train_split(path, save_dir=None, thread_count=5):
    patients = find_data_folders(path)

    # EEG Split
    eeg_dir = os.path.join(save_dir, "eeg")
    if len(patients) < thread_count:
        train_split_eeg(path, patients, eeg_dir, 0)
    elif len(os.listdir(eeg_dir)) <= 1:
        print("No EEG slice, start splitting...")
        t1_start = time.time()
        thread_list = []
        count = 0
        split_length = len(patients) // thread_count
        for i in range(thread_count):
            _patients = patients[count: min(len(patients) - 1, count + split_length)]
            thread = Thread(target=train_split_eeg, args=(path, _patients, eeg_dir, i))
            thread_list.append(thread)
            thread.start()
            count += split_length

        for _item in thread_list:
            _item.join()
        # train_split_eeg(path, patients, os.path.join(save_dir, "eeg"))
        t1_end = time.time()
        print('EEG Total time cost: %s s' % ((t1_end - t1_start) * 1))
    else:
        print("EEG slices exist, abort splitting.")

    pass


def test_split_eeg(path, patient_id=None):
    eeg_length = EEG_LENGTH
    count = 0
    patients = [patient_id]
    signals = []
    for item in patients:
        t2_start = time.time()
        patient_metadata = load_challenge_data(path, item)
        recording_ids = find_recording_files(path, item)
        num_recordings = len(recording_ids)

        # Extract EEG features.
        # eeg_channels = ['F3', 'P3', 'F4', 'P4']
        eeg_channels = ['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'T3', 'T4', 'C3',
                        'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']
        group = 'EEG'

        if num_recordings > 0:
            for recording_id in recording_ids:
                recording_location = os.path.join(path, item, '{}_{}'.format(recording_id, group))
                if os.path.exists(recording_location + '.hea'):
                    data, channels, sampling_frequency = load_recording_data(recording_location)
                    utility_frequency = get_utility_frequency(recording_location + '.hea')
                    if all(channel in channels for channel in eeg_channels):
                        data, channels = reduce_channels(data, channels, eeg_channels)
                        data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                        # data = np.array([data[0, :] - data[1, :],
                        #                  data[2, :] - data[3, :]])  # Convert to bipolar montage: F3-P3 and F4-P4
                        data = np.array([data[0, :] - data[2, :],
                                         data[2, :] - data[6, :],
                                         data[6, :] - data[10, :],
                                         data[10, :] - data[14, :],
                                         data[1, :] - data[3, :],
                                         data[3, :] - data[7, :],
                                         data[7, :] - data[11, :],
                                         data[11, :] - data[15, :],
                                         data[0, :] - data[4, :],
                                         data[4, :] - data[8, :],
                                         data[8, :] - data[12, :],
                                         data[12, :] - data[14, :],
                                         data[1, :] - data[5, :],
                                         data[5, :] - data[9, :],
                                         data[9, :] - data[13, :],
                                         data[13, :] - data[15, :],
                                         data[16, :] - data[17, :],
                                         data[17, :] - data[18, :]])
                        # eeg_features = get_eeg_features(data, sampling_frequency).flatten()
                        if data is not None:
                            # print("sampling freq: " + str(sampling_frequency))
                            # Sampling Freq = 128Hz
                            num_split = data.shape[1] // eeg_length
                            if num_split <= 0:
                                continue
                            data = np.hsplit(data, [num_split * eeg_length])[0]
                            split_array = np.asarray(np.hsplit(data, num_split))
                            # slice length = 10s, step = 17 ---> 1 slice per 180s
                            rewindowed_array = split_array[::17, :, :]
                            for slice in rewindowed_array:
                                signals.append(slice)
                                count += 1
                        else:
                            print("Test Dataset {0}_{1} EEG skipped - none data.".format(item, recording_id))
                    else:
                        print("Test Dataset {0}_{1} EEG skipped - missing channel.".format(item, recording_id))
                else:
                    print("Test Dataset {0}_{1} EEG skipped - missing header file.".format(item, recording_id))
        else:
            print("Test Dataset {0} EEG skipped - none recording.".format(item))
            continue
        print("Test Dataset {0} is done!".format(item))
        t2_end = time.time()
        print('Test split time cost: %s s' % ((t2_end - t2_start) * 1))

    return signals


def run_eeg_model(model, signals):
    if len(signals) == 0:
        outcome, outcome_probability, cpc = 1, 0.667, 4
        return outcome, outcome_probability, cpc

    signals = np.array(signals)
    signals = torch.FloatTensor(signals)
    test_dataset = EEGDatasetTest(signals)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    pred = []
    device = DEVICE_NAME

    for item in test_loader:
        with torch.no_grad():
            if test_loader.batch_size == 1:
                item = torch.unsqueeze(item, 0)
            pred.append(model(item.to(device))[0])

    pred_mean = sum(pred) / len(pred)
    pred_mean = [float(pred_mean[0]), float(pred_mean[1])]
    outcome = 0 if pred_mean[0] > pred_mean[1] else 1
    outcome_probability = pred_mean[1]
    cpc = 1 + 1 * (1 - pred_mean[0]) if outcome == 0 else 2 + 3 * pred_mean[1]

    return outcome, outcome_probability, cpc


# Dataset
class EEGDatasetTrain(Dataset):
    def __init__(self, path):
        super(EEGDatasetTrain, self).__init__()
        self.path = path
        # self.fileList = os.listdir(path)
        self.fileList = [f for f in os.listdir(path) if not f.startswith('.')]

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, idx):
        fname = self.fileList[idx]
        fullpath = os.path.join(self.path, fname)
        signal = np.load(fullpath)
        cpc = float(fname.split('_')[2])

        outcome = [1, 0] if cpc <= 2 else [0, 1]

        cpc = torch.FloatTensor([cpc])
        outcome = torch.FloatTensor(outcome)
        signal = signal.transpose((1, 0))
        signal = torch.FloatTensor(signal)
        signal = signal.reshape((1, signal.shape[0], -1))
        return signal, outcome


class EEGDatasetTest(Dataset):
    def __init__(self, signals):
        super(EEGDatasetTest, self).__init__()
        self.signals = signals

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, item):
        out_signal = self.signals[item].transpose(1, 0)
        return out_signal


# Modified for 2 channels aEEG Neural Network
class EEGNeuralNetwork(nn.Module):
    def __init__(self):
        super(EEGNeuralNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 2), padding=(0, 1)),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, padding=(0, 1)),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, padding=(0, 1)),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, padding=(0, 1)),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 1 * 81, 4096),
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
        # print(x.shape)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    pass


# EEGNet
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 120

        # Layer 1
        self.conv1 = nn.Conv1d(18, 16, 64, padding=0)
        self.batchnorm1 = nn.BatchNorm1d(16, False)
        # self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=0)
        # self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # 全连接层
        # 此维度将取决于数据中每个样本的时间戳数。
        # I have 120 timepoints.
        # self.fc1 = nn.Linear(4*2*7, 1)
        self.fc1 = nn.Sequential(
            nn.Linear(608, 2),
            nn.Sigmoid()
        )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(608, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(4096, 2),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        # Layer 1
        x = x.permute(0, 1, 3, 2)
        x = torch.squeeze(x, 1)
        # print(x.shape)
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = torch.unsqueeze(x, 1)
        # print(x.shape)
        # x = x.permute(0, 3, 1, 2)

        # print(x.shape)
        # Layer 2
        x = self.padding1(x)
        # print(x.shape)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # 全连接层
        # x = x.view(-1, 4*2*7)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x


# Unofficial Phase - Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cnn3 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 121, 4096),
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
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.cnn1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=2, padding=1),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.cnn2 = nn.Sequential(
#             nn.Conv2d(64, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=1, padding=1),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.cnn3 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, kernel_size=1, padding=1),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.cnn4 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, kernel_size=1, padding=1),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(512 * 121 * 2, 4096),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(4096, 2),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         # x = x.permute(0, 1, 3, 2)
#         x = self.cnn1(x)
#         x = self.cnn2(x)
#         x = self.cnn3(x)
#         x = self.cnn4(x)
#         x = x.view(x.shape[0], -1)
#         x = self.fc(x)
#         return x


if __name__ == '__main__':
    # train_challenge_model(r"E:\Downloads\physionet\i-care-2.0.physionet.org\training", r"model", 3)
    # train_challenge_model(r"D:\Data\Dataset\pc2023\tiny", r"model", 3)
    pass
