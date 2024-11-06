import numpy as np
import pandas as pd

from config import TRAIN_CSV, TRAIN_EEGS, eeg_duration, TRAIN_SPECTROGRAMS, spectrogram_duration, eeg_sample_rate
from fourier import fourier_transform_eeg


def setup_data():
    pass


def sample_data():
    train = pd.read_csv(TRAIN_CSV)

    eeg_id = train['eeg_id'][0]
    spectrogram_id = train['spectrogram_id'][0]

    eeg_offset = train['eeg_label_offset_seconds'][0]
    spectrogram_offset = train['spectrogram_label_offset_seconds'][0]

    eeg = pd.read_parquet(f'{TRAIN_EEGS}{eeg_id}.parquet').loc[eeg_offset: eeg_offset + eeg_duration - 1, ]
    eeg = eeg.T.to_numpy()

    spectrogram = pd.read_parquet(f'{TRAIN_SPECTROGRAMS}{spectrogram_id}.parquet').loc[
                  spectrogram_offset: spectrogram_offset + spectrogram_duration - 1, ]
    spectrogram = spectrogram.T.to_numpy()

    return eeg, spectrogram


def get_input_data(length=1000):
    # Reading from the train csv
    train_data = pd.read_csv(TRAIN_CSV)

    list_of_x_train = np.empty([length, 2, 20, 5001])
    list_of_y_train = np.empty([length, 6])

    for i in range(1, length + 1):
        # Getting the i^th eeg and spectrogram data id respectively
        eeg_id = train_data['eeg_id'][i]

        # Getting the first label offset seconds for eeg and spectrogram respectively
        eeg_label_offset_seconds = train_data['eeg_label_offset_seconds'][i] * eeg_sample_rate

        # Reading the eeg and spectrogram data with the aforementioned ids
        eeg = pd.read_parquet(TRAIN_EEGS + f'{eeg_id}.parquet')

        # Takes the 50 second sample of the eeg data
        eeg = eeg.loc[eeg_label_offset_seconds : eeg_label_offset_seconds + eeg_duration - 1,]
        eeg = eeg.to_numpy().T
        # Each row represents a time point with all the node.
        # Each column represents a node at all time points one by one.

        # Votes for each class
        seizure_vote = train_data['seizure_vote'][i]
        lpd_vote = train_data['lpd_vote'][i]
        gpd_vote = train_data['gpd_vote'][i]
        lrda_vote = train_data['lrda_vote'][i]
        grda_vote = train_data['grda_vote'][i]
        other_vote = train_data['other_vote'][i]

        # Defining the new X_train and Y_train
        y_train = np.array([seizure_vote,
                            lpd_vote,
                            gpd_vote,
                            lrda_vote,
                            grda_vote,
                            other_vote])
        x_train = fourier_transform_eeg(eeg)
        np.append(list_of_x_train, x_train)
        np.append(list_of_y_train, y_train)

    return list_of_x_train, list_of_y_train

