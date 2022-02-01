import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense, Activation
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential


def build_model(input_shape):
    model = Sequential()

    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.6))

    model.add(LSTM(units=256, return_sequences=True))
    model.add(Dropout(0.6))

    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.6))

    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(units=32))
    model.add(Dropout(0.4))

    model.add(Dense(units=16))
    model.add(Activation('relu'))

    model.add(Dense(units=2))
    model.add(Activation("softmax"))
    model.compile(optimizer="adam", loss=categorical_crossentropy, metrics=["accuracy"])
    print(model.summary())
    return model


def training(y_train, y_test, x_train, x_test, epochs):
    model = build_model(input_shape=(x_train.shape[1], 1))

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=256, verbose=1, validation_data=(x_test, y_test))

    return history, model
