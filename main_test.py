## Real time record and audio ml process
## Modified from Real-time text-mode Spectrogram
## https://python-sounddevice.readthedocs.io/en/0.3.7/examples.html

import argparse
import os
import numpy as np
from scipy.io import wavfile
import pandas as pd
import joblib
import sounddevice as sd
from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO

usage_line = ' press <enter> to quit '

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

# Create cmd api arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-1', '--list-devices', action='store_true', help='list audo devices and exit')
parser.add_argument('-b', '--block-duration', type=float, metavar='DURATION', default=500, help='block size (default %(default)s milliseconds)')
parser.add_argument('-d', '--device', type=int_or_str, help='input device (numeric ID orsubstring)')
args = parser.parse_args()

scaler1 = joblib.load('model_set/scaler1.joblib')  # load feature normalizer
nmodel = joblib.load('model_set/rf.joblib')  # load prediction model

# Record and process
# Note: train set sample data type must match record type (float32, within [-1, 1])
# Note: extracted features needs the same normalization as the train sets
try:
    print(sd.query_devices())
    samplerate = sd.query_devices(args.device, 'input')['default_samplerate']  # recording device samplerate
    win = int(samplerate*args.block_duration/1000)  # record block size, window length

    def callback(indata, frames, time, status):  # call back function, constantly running while recording
        if status:
            print(f'{status=}')
        if any(indata):
            feature, fn = aF.feature_extraction(indata[:, 0], samplerate, window=win, step=win)  # extract feature
            # print(max(indata), indata[0].dtype)  # check data range and type, should be within [-1, 1] and corresponding float
            x = scaler1.transform(feature.T[:,:68])  # Normalization is necessary for model prediction
            print(nmodel.predict_proba(x))
        else:
            print('no input')

    with sd.InputStream(device=args.device, channels=1, callback=callback, blocksize=win, samplerate=samplerate):  # Input stream recording
        while True:
            response = input()
            if response in ('', 'q', 'Q'):
                break

except KeyboardInterrupt:
    parser.exit('Interrupted by user')
except Exception as e:
    parser.exit(type(e).__name__+': '+str(e))


pass
