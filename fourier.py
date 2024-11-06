from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.fft import rfft, rfftfreq, irfft

from config import eeg_sample_rate, eeg_time


def fourier_transform_eeg(eeg_data: np.array, sample_rate: int = eeg_sample_rate, duration: int = eeg_time) -> np.ndarray:
    """
    :param eeg_data: 1D np.array
    :param sample_rate: int
    :param duration: int
    :return: np.array(2, 20, duration / 2 + 1)

    Function to Take an EEG and Fourier Transform it

    Assumption:
                - The eeg_data is a 1D numpy array
                - The eeg_data is a time series data
                - The eeg_data comprises sums of sine or cosine waves (with a little bit of noise)

    This function takes in an EEG data and plots the original signal, calculates the fourier transform and plots the
    filtered signal.
    """

    # Creating a list to store the filtered signals
    filtered_signals: Union[list, np.ndarray] = [[], []]

    for i in range(eeg_data.shape[0]):
        # Getting the amplitude and frequency of the signal
        amplitude: np.ndarray = rfft(eeg_data[i])
        frequency: np.ndarray = rfftfreq(sample_rate * duration, 1 / sample_rate)

        filtered_signals[0].append(amplitude)
        filtered_signals[1].append(frequency)

    filtered_signals = np.array(filtered_signals)
    return filtered_signals


def visualize_inverse_fourier(transformed_eeg_data: np.ndarray):
    """
    :param transformed_eeg_data: 1D np.array
    :return: None

    Function to visualize a Fourier Transformed EEG
    """
    axis: Axes
    figure: Figure
    amplitude: int
    frequency: int

    nx = 5
    ny = 4

    figure, axis = plt.subplots(nx, ny, figsize=(20, 20))
    figure.supxlabel("Time(s)")
    figure.supylabel("Voltage (micro volts)")

    for i in range(transformed_eeg_data.shape[1]):
        # Reconstructing the filtered signal using the inverse Fourier transform
        amplitude, frequency = transformed_eeg_data[:, i]
        reconstructed_signal: np.ndarray = irfft(amplitude)
        axis[i // ny, i % ny].plot(reconstructed_signal, label="Reconstructed Signal", color="blue")

    plt.show()
