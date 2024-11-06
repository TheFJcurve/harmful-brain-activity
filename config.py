import os.path

PATH = os.path.abspath('data')
TRAIN_CSV = os.path.join(PATH, 'train.csv')
TEST_CSV = os.path.join(PATH, 'test.csv')
SUBMISSION_CSV = os.path.join(PATH, 'submission.csv')
TEST_EEGS = os.path.join(PATH, 'test_eegs/')
TRAIN_EEGS = os.path.join(PATH, 'train_eegs/')
TEST_SPECTROGRAMS = os.path.join(PATH, 'test_spectrograms/')
TRAIN_SPECTROGRAMS = os.path.join(PATH, 'train_spectrograms/')

eeg_duration = 10000
eeg_time = 50
eeg_sample_rate = 200
spectrogram_duration = 300
spectrogram_time = 10 * 60

BATCH_SIZE = 100
