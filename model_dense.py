from keras.models import Sequential
from keras import metrics
import tensorflow as tf
from keras.layers import Dense, Dropout  # Thêm import này

def create_dense_model(input_size, number_layers=0):
    print("create Dense model ")
    model = Sequential()
    ##input lay
    model.add(Dense(input_size, activation='relu', input_shape=(input_size,)))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dropout(.2))
    model.add(Dense(1))

    print('Compiling...')
    model.compile(loss='mean_absolute_error',
                  optimizer=tf.optimizers.Adam(0.001),
                  metrics=[metrics.RootMeanSquaredError(), 'mean_squared_error'])

    return model