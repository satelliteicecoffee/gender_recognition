# Read wav file and extract feature
# Export feature to csv
# https://www.kaggle.com/primaryobjects/voicegender
# http://librosa.org/doc/latest/index.html

from scipy.io import wavfile
import numpy as np
import pandas as pd
from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import MidTermFeatures as mF
from pyAudioAnalysis import audioBasicIO as aIO

# Read wav file
filelist = ["beep"]  # filenames to read, could be mutiple items
label = 1  # corresponding label of this file set

win = 0.05  # window length, unit second
step = 0.025  # step length, unit second
fs = 44100  # sample rate
bitdepth = 32  # file bit depth, control normalizization

fl = []  # short feature list to be appended
fl_mid = []  # mid feature list to be appended
for i, filename in enumerate(filelist):
    fs, x = wavfile.read(f'test_set/{filename}.wav')  # sound file array, sample rate
    x = aIO.stereo_to_mono(x)      # transfer to mono

    # Normalization to [-1, 1]
    # x = x / (2**(bitdepth-1))    # transfer sample range to [-1, 1] float, if already then skip
    print(max(x),x[0].dtype)

    # Feature extraction
    duration = len(x) / float(fs)
    print(f'duration = {duration} seconds')

    f, fn = aF.feature_extraction(x, fs, int(fs*win), int(fs*step))  # short feature, feature names
    fmid, fshort, fnmid = mF.mid_feature_extraction(x, fs, int(fs*8), int(fs*8), int(fs*win), int(fs*step))  # short and mid feature, feature names

    # Control of output features, total 68, changable
    fl.append(f.T[:,:68])  # short feature list
    fl_mid.append(fmid.T)  # mid feature list

# Merge features and add label
fcsv = np.concatenate(fl)
flbl = np.full((np.size(fcsv, 0), 1), label).reshape(np.size(fcsv, 0), 1)  # append label
fcsv = np.concatenate((fcsv, flbl), axis=1)  # generate csv file
fncsv = fn[:68]
fncsv.append('label')

fcsv_mid = np.concatenate(fl_mid)
fncsv_mid = fnmid.append('label')

# Save feature file
np.savetxt(f"test_set/{filelist[0]}_short.csv", fcsv, comments='', header=",".join(fncsv), delimiter=",")  # short feature save
# np.savetxt(f"test_set/{filelist[0]}_mid.csv", fmid.T, comments='', header=",".join(fnmid), delimiter=",")  # mid feature save

pass
