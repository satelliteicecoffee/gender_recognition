# Real time record and audio ml process

import os
import sys
import time
import click
import numpy as np
import pandas as pd
import configparser
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import soundfile as sf
import joblib
import sounddevice as sd
from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


# Load ini configuration file
abs_path = os.getcwd()
config = configparser.ConfigParser()
try:
    config.read(f'{abs_path}\\config.ini')
    blocksize = config['Prediction']['blocksize']  # used to generate an error if no file read
except KeyError:  # Default settings, create if no config.ini found
    config['Prediction'] = {'blocksize': '500', 'nmodel': 'knn'}
    config['Train'] = {'retrain': '0', 'window': '50', 'step': '25', 'fs': '44100'}
    with open(f'{abs_path}\\config.ini', 'w') as configfile:
        config.write(configfile)
    config.read(f'{abs_path}\\config.ini')
finally:  # Load configuration
    blocksize = int_or_str(config['Prediction']['blocksize'])  # prediction block size
    model = int_or_str(config['Prediction']['nmodel'])  # prediction model name
    retrain = int_or_str(config['Train']['retrain'])  # if retrain model
    window = int_or_str(config['Train']['window'])  # train window size
    step = int_or_str(config['Train']['step'])  # train step size
    train_samplerate = int_or_str(config['Train']['fs'])  # train material samplerate

# Ask for choice
choice = click.prompt("选择：1.开始测试 2.校准模型", type=click.IntRange(1,2))

# Train and dump model
if choice == 2:
    print('开始校准，请等待')
    # extract feature to csv
    label_0 = [0, 1]
    featurelist = []

    for label in label_0:
        path = abs_path + f'\\test_set\\set_{label}\\'  # label integrated into folder name
        filelist = next(os.walk(path), (None, None, []))[2]  # get filename list in folder
        path_filelist = []
        for item in filelist:
            path_filelist.append((path, item))

        # read wav file, multiprocessing feature extraction
        def extract_feature(path, filename):
            ob = sf.SoundFile(f'{path}{filename}')  # used to determine PCM number
            x, fs = sf.read(f'{path}{filename}')  # already normalized in sf.read
            x = aIO.stereo_to_mono(x)
            print(f'File: {filename}, {len(x)/fs} seconds, format {ob.subtype}, max float {max(x)}, label {label}.')

            # extract feature
            f, fn = aF.feature_extraction(x, fs, int(fs*window/1000), int(fs*step/1000))
            f = f.T[:, :68]
            flbl = np.full((np.size(f, 0), 1), label).reshape(np.size(f, 0), 1)  # add label
            f = np.concatenate((f, flbl), axis=1)  # merge label column
            return f, fn

        for pathname, filename in path_filelist:
            results = extract_feature(pathname, filename)
            featurelist.append(results[0])
        # num_cores = multiprocessing.cpu_count()  # count cpu number
        # results = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(extract_feature)(path, filename) for path, filename in path_filelist)  # parallel pool substituting for(parfor)
        # for item in results:
        #     featurelist.append(item[0])

    fcsv = np.concatenate(featurelist, axis=0)  # merge feature array
    np.random.shuffle(fcsv)  # randomize row order
    fncsv = results[1]
    fncsv.append('label')
    np.savetxt(f'{abs_path}\\test_set\\train_short.csv', fcsv, comments='', header=",".join(fncsv), delimiter=",")

    # train model
    # import data
    voice_data = pd.read_csv(f'{abs_path}\\test_set\\train_short.csv')
    tx = voice_data.iloc[:, :-1]
    ty = voice_data.iloc[:, -1]

    # transform label, compensate 0 value
    le = LabelEncoder()
    ty = le.fit_transform(ty)
    imp = SimpleImputer(missing_values=0, strategy='mean')
    tx = imp.fit_transform(tx)

    # split train and test data, normalization and dump
    x_train, x_test, y_train, y_test = train_test_split(tx, ty, test_size=0.15)
    scaler1 = StandardScaler()
    scaler1.fit(x_train)
    joblib.dump(scaler1, f'{abs_path}\\model_set\\scaler1.joblib')
    x_train = scaler1.transform(x_train)
    x_test = scaler1.transform(x_test)

    # modelling and dump
    logistic = LogisticRegression(max_iter=10000)
    logistic.fit(x_train, y_train)
    joblib.dump(logistic, f'{abs_path}\\model_set\\logistic.joblib')
    print('logistic complete')
    nn = MLPClassifier(max_iter=100000)
    nn.fit(x_train, y_train)
    joblib.dump(nn, f'{abs_path}\\model_set\\nn.joblib')
    print('nn complete')
    rf = RandomForestClassifier(n_estimators=12, criterion="gini")
    rf.fit(x_train, y_train)
    joblib.dump(rf, f'{abs_path}\\model_set\\rf.joblib')
    print('rf complete')
    # svc = SVC(C=1, kernel='rbf', probability=True)
    # svc.fit(x_train, y_train)
    # joblib.dump(svc, f'model_set/svc.joblib')
    # print('svc complete')
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    joblib.dump(knn, f'{abs_path}\\model_set\\knn.joblib')
    print('knn complete')

    # rewrite config, turn off retrain
    config.set('Train', 'retrain', '0')
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    print(time.asctime(time.localtime(time.time())))
    print('完成')
    os.system('pause')
    sys.exit(0)

    pass

# Real time predict
scaler1 = joblib.load(f'{abs_path}\\model_set\\scaler1.joblib')
nmodel = joblib.load(f'{abs_path}\\model_set\\{model}.joblib')
device = None

try:
    print(sd.query_devices())
    print('\x1b[4;30;44m' + '  Press <enter> to stop  ' + '\x1b[0m')
    samplerate = sd.query_devices(device, 'input')['default_samplerate']
    win = int(samplerate*blocksize/1000)

    showpredict = ["不合格", "合格"]
    showcolour = ['\x1b[6;30;41m', '\x1b[6;30;42m']

    def callback(indata, frames, time, status):
        if status:
            print(f'{status=}')
        indata = aIO.stereo_to_mono(indata)  # read stereo to mono
        if any(indata):
            feature, fn = aF.feature_extraction(indata, samplerate, window=win, step=win)
            x = scaler1.transform(feature.T[:, :68])
            xp = int_or_str(nmodel.predict(x))
            print('{: >26}'.format(showcolour[xp] + showpredict[xp] + '\x1b[0m'), end='\r')
        else:
            print('no input')

    with sd.InputStream(device=device, callback=callback, blocksize=win, samplerate=samplerate):
        while True:
            response = input()
            if response in (''):  # '' is <enter>
                break

except KeyboardInterrupt:
    print('interrupted by user')


pass
