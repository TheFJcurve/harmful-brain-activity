import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow import keras

from config import BATCH_SIZE, TEST_EEGS, TEST_CSV
from data_setup import setup_data, get_input_data
from fourier import fourier_transform_eeg, visualize_inverse_fourier


def visualize_eeg(eeg_data):
    """
    Function to Visualize an EEG
    """

    nx = 5
    ny = 4

    figure, axis = plt.subplots(nx, ny, figsize=(20, 20))
    figure.supxlabel("Time(s)")
    figure.supylabel("Voltage (micro volts)")

    for i in range(eeg_data.shape[0]):
        axis[i // ny, i % ny].plot(eeg_data[i],
                                   label="Filtered Signal",
                                   color="red")

    plt.show()


def setup_model(x_train_list, y_train_list):
    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=20,
                                                      restore_best_weights=True)

    reduce_on_plateau = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=10,
    )

    root_logdir = os.path.join(os.curdir, "logs")

    def get_run_logdir():
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)

    run_logdir = get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    # Model Architecture:
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dense(units=6, activation='sigmoid'),
    ])

    # Model Compilation
    model.compile(
        optimizer="rmsprop",
        loss="mse",
        loss_weights=None,
        metrics=[keras.metrics.BinaryAccuracy()],
        weighted_metrics=None,
        run_eagerly=False,
        steps_per_execution=1,
        jit_compile="auto",
        auto_scale_loss=True,
    )

    # Model Training
    history = model.fit(x_train_list,
                        y_train_list,
                        epochs=100,
                        batch_size=32,
                        callbacks=[tensorboard_cb, early_stopping_cb, reduce_on_plateau])

    return model


def get_prediction(model):
    test = pd.read_csv(TEST_CSV)

    test_eeg_id = test['eeg_id'][0]

    test_eeg = pd.read_parquet(f'{TEST_EEGS}{test_eeg_id}.parquet')
    test_eeg = test_eeg.T.to_numpy()

    visualize_eeg(test_eeg)
    transformed_test_eeg = fourier_transform_eeg(test_eeg).real
    visualize_inverse_fourier(transformed_test_eeg)

    print("Shape of the fourier transformed signal: ", transformed_test_eeg.shape)
    print("Data points for an entire EEG signal: ", transformed_test_eeg.size)

    answer = model.predict(np.array([transformed_test_eeg]))
    total_answer = np.sum(answer)

    print("Output Probability: ", answer / total_answer)


if __name__ == "__main__":
    setup_data()
    X_train_list, Y_train_list = get_input_data(BATCH_SIZE)
    model = setup_model(X_train_list, Y_train_list)
    get_prediction(model)
