"""
#################################################################
#### For HAR, our novel HDAD (IGAV) dataset was constructed  ####
#### by performing 4 dynamic and 3 static activities with    ####
#### the accelerometer and gyroscope sensors of the IOS      ####
#### smart phone in two different positions for a total of   ####
#### 15 seconds. Mentioned activities were collected in      ####
#### real time by placing them on the waist of a total of    ####
#### 10 volunteers.                                          ####
####                                                         ####
#### This simple source code is provided to test our dataset ####
#### by end-users with different deep learning architectural ####
#### models.                                                 ####
#################################################################
#### License Type: MIT license                               ####
#################################################################
#### Please contact for more information:                  ######
#### ibrahimalimetin@gmail.com                             ######
#### April 9, 2021                                         ######
#################################################################
"""

from __future__ import print_function

print(__doc__)

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
import os
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
import itertools

from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D


COLUMN_NAMES = [
        'user',
        'activity',
        'timestamp',
        'x-axis',
        'y-axis',
        'z-axis'
    ]

LABELS = [
        'MERDİVEN İNME',
        'UZANMA',
        'OTURMA',
        'AYAKTA DURMA',
        'MERDİVEN ÇIKMA',
        'YÜRÜME',
        'ZIPLAMA'
    ]

lables = np.array(LABELS)
veri_yolu = 'data/IGAV_3Input_Ham_AccXYZ.txt'
RANDOM_SEED = 13
ZAMAN_ADIMI = 100
ZAMAN_BOLUTU = 180
N_CLASSES = 7
N_FEATURES = 3

CHECK_ROOT = 'checkpoint/'

if not os.path.exists(CHECK_ROOT):
    os.makedirs(CHECK_ROOT)

epochs = 35
batch_size = 32
n_hidden = 128


def _count_classes(y):
        return len(set([tuple(category) for category in y]))

def plot_history(history):
        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

        if len(loss_list) == 0:
                print('Loss is missing in history')
                return

        epochs = range(1, len(history.history[loss_list[0]]) + 1)

        plt.figure(1)
        for l in loss_list:
                plt.plot(epochs, history.history[l], 'orange',
                         label='Eğitim kaybı (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
        for l in val_loss_list:
                plt.plot(epochs, history.history[l], 'blue',
                         label='Test kaybı (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

        plt.title('Model Kaybı')
        plt.xlabel('Yineleme')
        plt.ylabel('Kayıp')
        plt.grid(True)
        plt.legend()

        plt.figure(2)
        for l in acc_list:
                plt.plot(epochs, history.history[l], 'orange',
                         label='Eğitim doğruluğu (' + str(format(history.history[l][-1], '.5f')) + ')')
        for l in val_acc_list:
                plt.plot(epochs, history.history[l], 'blue',
                         label='Test doğruluğu (' + str(format(history.history[l][-1], '.5f')) + ')')

        plt.title('Model Doğruluğu')
        plt.xlabel('Yineleme')
        plt.ylabel('Doğruluk')
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == '__main__':

        veri_cnv = []
        etiketler = []
        
        veri = pd.read_csv(veri_yolu, header=None, names=COLUMN_NAMES)
        veri['z-axis'].replace({';': ''}, regex=True, inplace=True)
        veri = veri.dropna()

        for c in range(0, len(veri) - ZAMAN_BOLUTU, ZAMAN_ADIMI):
            x = veri['x-axis'].values[c: c + ZAMAN_BOLUTU]
            y = veri['y-axis'].values[c: c + ZAMAN_BOLUTU]
            z = veri['z-axis'].values[c: c + ZAMAN_BOLUTU]
            veri_cnv.append([x, y, z])
            label = stats.mode(veri['activity'][c: c + ZAMAN_BOLUTU])[0][0]
            etiketler.append(label)
      
        etiketler = np.asarray(pd.get_dummies(etiketler), dtype=np.float32)
        veri_cnv = np.asarray(veri_cnv, dtype=np.float32).transpose(0, 2, 1)

        X_train, X_test, Y_train, Y_test = train_test_split(veri_cnv, etiketler, test_size=0.3, random_state=RANDOM_SEED)
        
        print("Data shape: ", veri_cnv.shape)
        print("Labels shape:", etiketler.shape)
        print("X train size: ", len(X_train))
        print("X test size: ", len(X_test))
        print("y train size: ", len(Y_train))
        print("y test size: ", len(Y_test))
        y_test = Y_test.argmax(1)

zamanadimi = len(X_train[0])
girdi_boyutu = len(X_train[0][0])
n_classes = _count_classes(Y_train)

input_shape = (zamanadimi * girdi_boyutu)

X_train = X_train.astype('float32')
Y_train = Y_train.astype('float32')

def create_model():
        model = Sequential()
        print("CREATE MODEL_ X train shape: ", X_train.shape)
        model.add(Conv1D(n_hidden, 2, activation='relu', input_shape=(zamanadimi, girdi_boyutu)))
        model.add(Conv1D(n_hidden, 2, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(np.int(n_hidden / 2), 2, activation='relu'))
        model.add(Conv1D(np.int(n_hidden / 2), 2, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(np.int(n_hidden / 4), 2, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.2))
        model.add(Dense(n_classes, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = '1d_cnnham3input-3.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6)
callbacks = [checkpoint, lr_reducer]


krs_scikit_model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=1)
history = krs_scikit_model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                                   validation_data=(X_test, Y_test), verbose=1, callbacks=callbacks)

plot_history(history)

