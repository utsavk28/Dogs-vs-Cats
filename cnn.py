from config import batch_size, img_height, img_width,  train,  test
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from utils import format_data
from config import TRAIN_NUM_EPOCHS, EARLY_STOPPING
from models import cnn_model


def cnn():
    ds_train, ds_validation, _ = format_data(
        train, test, img_height, img_width, batch_size)

    model = cnn_model()

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=EARLY_STOPPING)
    model.compile(optimizer=Adam(learning_rate=3e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(ds_train, validation_data=ds_validation, epochs=TRAIN_NUM_EPOCHS, batch_size=batch_size, verbose=1,
                        callbacks=[callback])
    return model, history
