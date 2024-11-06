import os
import time

import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, History
from keras.src.layers import Flatten, Dense, Dropout
from keras.src.models.sequential import Sequential
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame
from tensorflow import keras

from config import BATCH_SIZE, TEST_EEGS, TEST_CSV
from data_setup import setup_data, get_input_data
from fourier import fourier_transform_eeg, visualize_inverse_fourier


def visualize_eeg(eeg_data: np.ndarray) -> None:
    """
    Function to Visualize an EEG
    """
    figure: Figure
    axis: Axes

    nx: int = 5
    ny: int = 4

    figure, axis = plt.subplots(nx, ny, figsize=(20, 20))
    figure.supxlabel("Time(s)")
    figure.supylabel("Voltage (micro volts)")

    for i in range(eeg_data.shape[0]):
        axis[i // ny, i % ny].plot(eeg_data[i], label="Filtered Signal", color="red")

    plt.show()


def setup_model(x_train_list: np.ndarray, y_train_list: np.ndarray) -> tuple[Sequential, History]:
    # Callbacks
    early_stopping_cb = EarlyStopping(patience=20, restore_best_weights=True)

    reduce_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, )

    root_logdir = os.path.join(os.curdir, "logs")

    def get_run_logdir():
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)

    run_logdir = get_run_logdir()
    tensorboard_cb = TensorBoard(run_logdir)

    # Model Architecture:
    novice_model: Sequential = keras.models.Sequential([Flatten(),
                                                        Dense(units=64, activation='relu'),
                                                        Dropout(0.5),
                                                        Dense(units=128, activation='relu'),
                                                        Dropout(0.4),
                                                        Dense(units=128, activation='relu'),
                                                        Dropout(0.6),
                                                        Dense(units=32, activation='relu'),
                                                        Dropout(0.5),
                                                        Dense(units=6, activation='sigmoid'),
                                                        ])

    # Model Compilation
    novice_model.compile(optimizer="rmsprop", loss="mse", loss_weights=None, metrics=['acc'], weighted_metrics=None,
                         run_eagerly=False, steps_per_execution=1, jit_compile="auto", auto_scale_loss=True, )

    # Model Training
    history: History = novice_model.fit(x_train_list, y_train_list, epochs=100, batch_size=32,
                                        callbacks=[tensorboard_cb, early_stopping_cb, reduce_on_plateau])

    return novice_model, history


def get_prediction(trained_model: Sequential) -> None:
    test: DataFrame = pd.read_csv(TEST_CSV)

    test_eeg_id: int = test['eeg_id'][0]

    eeg: DataFrame = pd.read_parquet(f'{TEST_EEGS}{test_eeg_id}.parquet')
    numpy_eeg: np.ndarray = eeg.T.to_numpy()

    visualize_eeg(numpy_eeg)
    transformed_test_eeg: np.ndarray = fourier_transform_eeg(numpy_eeg).real
    visualize_inverse_fourier(transformed_test_eeg)

    print("Shape of the fourier transformed signal: ", transformed_test_eeg.shape)
    print("Data points for an entire EEG signal: ", transformed_test_eeg.size)

    answer: np.ndarray = trained_model.predict(np.array([transformed_test_eeg]))
    total_answer: int = np.sum(answer)

    print("Output Probability: ", answer / total_answer)


if __name__ == "__main__":
    setup_data()
    X_train_list, Y_train_list = get_input_data(BATCH_SIZE)
    model, model_history = setup_model(X_train_list, Y_train_list)
    get_prediction(model)
