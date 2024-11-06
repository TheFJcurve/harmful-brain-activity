import numpy as np
import pandas as pd
from pandas import DataFrame

from config import TRAIN_CSV, TRAIN_EEGS, eeg_duration, TRAIN_SPECTROGRAMS, spectrogram_duration, eeg_sample_rate
from fourier import fourier_transform_eeg


def setup_data():
    pass


def sample_data() -> tuple[np.ndarray, np.ndarray]:
    train: DataFrame = pd.read_csv(TRAIN_CSV)

    eeg_id: int = train['eeg_id'][0]
    spectrogram_id: int = train['spectrogram_id'][0]

    eeg_offset: int = train['eeg_label_offset_seconds'][0]
    spectrogram_offset: int = train['spectrogram_label_offset_seconds'][0]

    eeg: DataFrame = pd.read_parquet(f'{TRAIN_EEGS}{eeg_id}.parquet').loc[eeg_offset: eeg_offset + eeg_duration - 1, ]
    transformed_eeg: np.ndarray = eeg.T.to_numpy()

    spectrogram: DataFrame = pd.read_parquet(f'{TRAIN_SPECTROGRAMS}{spectrogram_id}.parquet').loc[
                  spectrogram_offset: spectrogram_offset + spectrogram_duration - 1, ]
    transformed_spectrogram: np.ndarray = spectrogram.T.to_numpy()

    return transformed_eeg, transformed_spectrogram


def get_input_data(length: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    # Reading from the train csv
    train_data: DataFrame = pd.read_csv(TRAIN_CSV)

    list_of_x_train: np.ndarray = np.empty([length, 2, 20, 5001], dtype=np.complex128)
    list_of_y_train: np.ndarray = np.empty([length, 6], dtype=np.int64)

    for i in range(1, length + 1):
        # Getting the i^th eeg and spectrogram data id respectively
        eeg_id: int = train_data['eeg_id'][i]

        # Getting the first label offset seconds for eeg and spectrogram respectively
        eeg_label_offset_seconds: int = train_data['eeg_label_offset_seconds'][i] * eeg_sample_rate

        # Reading the eeg and spectrogram data with the aforementioned ids
        eeg: DataFrame = pd.read_parquet(TRAIN_EEGS + f'{eeg_id}.parquet')

        # Takes the 50-second sample of the eeg data
        eeg = eeg.loc[eeg_label_offset_seconds: eeg_label_offset_seconds + eeg_duration - 1, ]
        transformed_eeg: np.ndarray = eeg.to_numpy().T
        # Each row represents a time point with all the node.
        # Each column represents a node at all-time points one by one.

        # Votes for each class
        seizure_vote: int = train_data['seizure_vote'][i]
        lpd_vote: int = train_data['lpd_vote'][i]
        gpd_vote: int = train_data['gpd_vote'][i]
        lrda_vote: int = train_data['lrda_vote'][i]
        grda_vote: int = train_data['grda_vote'][i]
        other_vote: int = train_data['other_vote'][i]

        # Defining the new X_train and Y_train
        x_train: np.ndarray = fourier_transform_eeg(transformed_eeg)
        y_train: np.ndarray = np.array([seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote])

        np.append(list_of_x_train, x_train)
        np.append(list_of_y_train, y_train)

    return list_of_x_train, list_of_y_train
