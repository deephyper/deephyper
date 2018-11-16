import pandas as pd 
import numpy as np
import os
import sys
import gzip

from keras.layers import Input, Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau 

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path)

EPOCH = 400
BATCH = 20
CLASSES = 36

PL = 60484   # 1 + 60483 these are the width of the RNAseq datasets
P     = 60483   # 60483
DR    = 0.1      # Dropout rate

def load_data():
        train_path = 'type_18_300_train.csv'
        test_path = 'type_18_300_test.csv'

        df_train = (pd.read_csv(train_path,header=None).values).astype('float32')
        df_test = (pd.read_csv(test_path,header=None).values).astype('float32')

	print('df_train shape:', df_train.shape)
	print('df_test shape:', df_test.shape)

        df_y_train = df_train[:,0].astype('int')
        df_y_test = df_test[:,0].astype('int')

        Y_train = np_utils.to_categorical(df_y_train,CLASSES)
        Y_test = np_utils.to_categorical(df_y_test,CLASSES)
              
        df_x_train = df_train[:, 1:PL].astype(np.float32)
        df_x_test = df_test[:, 1:PL].astype(np.float32)
            
#        X_train = df_x_train.as_matrix()
#        X_test = df_x_test.as_matrix()
            
        X_train = df_x_train
        X_test = df_x_test
            
        scaler = MaxAbsScaler()
        mat = np.concatenate((X_train, X_test), axis=0)
        mat = scaler.fit_transform(mat)
        
        X_train = mat[:X_train.shape[0], :]
        X_test = mat[X_train.shape[0]:, :]
        
        return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = load_data()

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

x_train_len = X_train.shape[1]

# this reshaping is critical for the Conv1D to work

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

model = Sequential()
model.add(Conv1D(filters=128, kernel_size=20, strides=1, padding='valid', input_shape=(P, 1)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=128, kernel_size=10, strides=1, padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=10))
model.add(Flatten())
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(CLASSES))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])

# set up a bunch of callbacks to do work during model training..

checkpointer = ModelCheckpoint(filepath='tc1.autosave.model.h5', verbose=0, save_weights_only=False, save_best_only=True)
csv_logger = CSVLogger('tc1.training.log')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

history = model.fit(X_train, Y_train,
                    batch_size=BATCH, 
                    epochs=EPOCH,
                    verbose=1, 
                    validation_data=(X_test, Y_test),
                    callbacks = [checkpointer, csv_logger, reduce_lr])

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

# serialize model to JSON
model_json = model.to_json()
with open("tc1.model.json", "w") as json_file:
        json_file.write(model_json)

# serialize model to YAML
model_yaml = model.to_yaml()
with open("tc1.model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)


# serialize weights to HDF5
model.save_weights("tc1.model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('tc1.model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_json = model_from_json(loaded_model_json)


# load yaml and create model
yaml_file = open('tc1.model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model_yaml = model_from_yaml(loaded_model_yaml)


# load weights into new model
loaded_model_json.load_weights("tc1.model.h5")
print("Loaded json model from disk")

# evaluate json loaded model on test data
loaded_model_json.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

print('json Test score:', score_json[0])
print('json Test accuracy:', score_json[1])

print("json %s: %.2f%%" % (loaded_model_json.metrics_names[1], score_json[1]*100))


# load weights into new model
loaded_model_yaml.load_weights("tc1.model.h5")
print("Loaded yaml model from disk")

# evaluate loaded model on test data
loaded_model_yaml.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score_yaml = loaded_model_yaml.evaluate(X_test, Y_test, verbose=0)

print('yaml Test score:', score_yaml[0])
print('yaml Test accuracy:', score_yaml[1])

print("yaml %s: %.2f%%" % (loaded_model_yaml.metrics_names[1], score_yaml[1]*100))
