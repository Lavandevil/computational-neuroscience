import mne
import numpy as np

face = mne.read_epochs_eeglab(r"D:\8thSeme\neuroscience\EEG\EEG data\data\face.set")
nonface = mne.read_epochs_eeglab(r"D:\8thSeme\neuroscience\EEG\EEG data\data\New folder\nonface.set")

# label گذاری ساده: face=1, nonface=0
face_data = face.get_data()      # shape: (166, n_ch, n_times)
nonface_data = nonface.get_data() # shape: (821, n_ch, n_times)

X = np.concatenate([face_data, nonface_data], axis=0)
y = np.array([1]*166 + [0]*821)

print(X.shape, y.shape)
