import wget
import json
import glob
import numpy as np
from pydub import AudioSegment
import librosa
import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import librosa.display
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D


def download_bird_sounds(query):
    """
        Download the bird sounds from www.xeno-canto.org

        Function can be changed to other websites if needed

        param:
          query: Name of the bird. Blank space with %20
    """
    # GET ALL FILENAMES AND PATHS OF THE QUERY BIRD
    url = 'https://www.xeno-canto.org/api/2/recordings?query=' + query

    dataFolder = "Data/" + query.replace("%20", "_") + "/"
    mp3Folder = dataFolder + "mp3/"
    arrayFolder = dataFolder + "arrays/"
    predTestFolder = dataFolder + "preTest/"

    os.mkdir(dataFolder)
    os.mkdir(mp3Folder)
    os.mkdir(arrayFolder)
    os.mkdir(predTestFolder)

    filename = wget.download(url, dataFolder + 'recordings.json')
    print(filename)

    # Get the json entries from your downloaded json
    jsonFile = open(dataFolder + 'recordings.json', 'r')
    values = json.load(jsonFile)
    jsonFile.close()

    # Create a pandas dataframe of records & convert to .csv file
    record_df = pd.DataFrame(values['recordings'])
    record_df.to_csv(dataFolder + 'xc-noca.csv', index=False)

    # Make wget input file
    url_list = []
    for file in record_df['file'].tolist():
        url_list.append('{}'.format(file))
    with open(dataFolder + 'xc-noca-urls.txt', 'w+') as f:
        for item in url_list:
            f.write("{}\n".format(item))

    # Get all soundfiles
    os.system('wget -P ' + mp3Folder + ' --trust-server-names -i' + dataFolder + 'xc-noca-urls.txt')

    files = os.listdir(mp3Folder)
    [os.replace(mp3Folder + file, mp3Folder + file.replace(" ", "_")) for file in files]


def prepare_dataset(query, nbrOfTestSoundsForPrediction):
    """
        Prepares the dataset for using in model

        :param
        query: Name of the bird. Blank space with %20
        nbrOfTestSoundsForPrediction: Number of sounds copied to a
        seperate folder for later predictions
    """
    dataFolder = "Data/" + query.replace("%20", "_") + "/"
    mp3Folder = dataFolder + "mp3/"
    arrayFolder = dataFolder + "arrays/"
    predTestFolder = dataFolder + "preTest/"

    # The following line is only needed once if ffmpeg is not part of the PATH variables
    # os.environ["PATH"] += os.pathsep + r'F:\ffmpeg\bin'

    # Reformat path string
    globlist = glob.glob(mp3Folder + "*.mp3")
    new_list = []
    for string in globlist:
        new_string = string.replace("\\", "/")
        new_list.append(new_string)

    # Copy 5 entries to /predTest
    last_elements = new_list[-nbrOfTestSoundsForPrediction:]
    print(last_elements)

    for file in last_elements:
        shutil.copy(file, predTestFolder)
        os.remove(file)

    globlist.clear()
    globlist = glob.glob(mp3Folder + "*.mp3")
    new_list.clear()
    for string in globlist:
        new_string = string.replace("\\", "/")
        new_list.append(new_string)

    # Extract frequencies and save them as np array
    for file in new_list:
        src = file
        dst = "tmp/tmp.wav"

        # convert mp3 to wav
        sound = AudioSegment.from_mp3(src)
        ten_seconds = 10 * 1000
        first_10_seconds = sound[:ten_seconds]
        first_10_seconds.export(dst, format="wav")

        y, sr = librosa.load(dst)

        # CREATE A FIXED LENGTH
        librosa.util.fix_length(y, 220500)

        # EXTRACT FEATURES
        mfccs_features = librosa.feature.mfcc(y, sr, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

        # SAVE FEATURES TO FILE
        index = new_list.index(file)
        arrayPath = arrayFolder + str(index)
        np.save(arrayPath, mfccs_scaled_features)

        # Remove temp wav file
        os.remove(dst)

    # CREATE AN ARRAY OF ALL PICTURE FEATURES AND SAVE IT
    arraylist = glob.glob(arrayFolder + "*.npy")
    extracted_features = []
    for file in arraylist:
        data = np.load(file)
        label = query.replace("%20", "_")
        extracted_features.append([data, label])

    np.save(arrayFolder + "summery_array", extracted_features)


def download_and_prepare_sound_dataset(query, nbrOfTestSoundsForPrediction):
    """
        Downloads & prepares the dataset for using in model

        :param
        query: Name of the bird. Blank space with %20
        nbrOfTestSoundsForPrediction: Number of sounds copied to a
        seperate folder for later predictions
    """
    download_bird_sounds(query)
    prepare_dataset(query, nbrOfTestSoundsForPrediction)


def show_spectogram_for_mp3(filepath):
    """
        Shows the spectogram of a given mp3 path

        :param
        filepath: The filepath of the mp3
    """
    src = filepath
    dst = "tmp/tmp.wav"

    # convert wav to mp3
    sound = AudioSegment.from_mp3(src)
    ten_seconds = 10 * 1000
    first_10_seconds = sound[:ten_seconds]
    first_10_seconds.export(dst, format="wav")

    y, sr = librosa.load(dst)
    librosa.util.fix_length(y, 220500)

    # SHOW WAVE
    fig, ax = plt.subplots(nrows=3, sharex=True)
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set(title='Wave')

    # SHOW SPEC
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr, ax=ax[1])
    ax[1].set(title='Linear-frequency power spectrogram')

    # SHOW EXTRACTED FEATURES
    mfccs_features = librosa.feature.mfcc(y, sr, n_mfcc=40)
    img = librosa.display.specshow(mfccs_features, x_axis='time', ax=ax[2])
    ax[2].set(title='mfccs_features')
    plt.show()


def perpare_mp3_for_prediction(filepath):
    """
        Prepares the given mp3 for prediction

        :param
        filepath: The filepath of the mp3
    """
    src = filepath
    dst = "tmp/tmp2.wav"

    # convert mp3 to wav
    sound = AudioSegment.from_mp3(src)
    ten_seconds = 10 * 1000
    first_10_seconds = sound[:ten_seconds]
    first_10_seconds.export(dst, format="wav")

    y, sr = librosa.load(dst)

    # FIXED LENGTH
    librosa.util.fix_length(y, 220500)

    # EXTRACT FEATURES
    mfccs_features = librosa.feature.mfcc(y, sr, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    # RESHAPE
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

    # Remove temp wav file
    os.remove(dst)

    return mfccs_scaled_features


def get_dataframe(query):
    dataFolder = "Data/" + query.replace("%20", "_") + "/"
    arrayFolder = dataFolder + "arrays/"

    numpy_data = np.load(arrayFolder + "summery_array.npy", allow_pickle=True)
    extracted_features_df = pd.DataFrame(numpy_data, columns=['feature', 'label'])

    return extracted_features_df


def get_concat_dataframe(query_list):
    result = pd.DataFrame()

    for query in query_list:
        df = get_dataframe(query)
        result = result.append(df)

    return result


def create_and_fit_dense_model(batch_size, epochs, callbacks, X_train, X_test, y_train, y_test):
    # BUILD THE MODEL
    model = Sequential()
    model.add(Dense(100, input_shape=(40,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # TRAIN THE MODEL
    history = model.fit(X_train, y_train, batch_size=batch_size, callbacks=callbacks, epochs=epochs,
                        validation_data=(X_test, y_test))

    # SAVE MODEL AND HISTORY_DATA
    np.save('Auswertung/history.npy', history.history)
    model.save('Auswertung/model')

    return history


def create_and_fit_cnn_model(batch_size, epochs, callbacks, X_train, X_test, y_train, y_test):
    # BUILD THE MODEL
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(40, 1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # TRAIN THE MODEL
    history_CNN = model.fit(X_train, y_train, batch_size=batch_size, callbacks=callbacks, epochs=epochs,
                            validation_data=(X_test, y_test))

    # SAVE MODEL AND HISTORY_DATA
    np.save('Auswertung/history_CNN.npy', history_CNN.history)
    model.save('Auswertung/model_CNN')

    return history_CNN