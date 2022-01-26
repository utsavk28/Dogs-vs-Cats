from config import batch_size, img_height, img_width, channels, train, test, TF_NUM_EPOCHS
from tensorflow.keras.optimizers import Adam
from utils import format_data
from models import tf_resnet_model, tf_efficientnet_model, tf_inception_model


def tf_resnet():
    ds_train, ds_validation, _ = format_data(
        train, test, img_height, img_width, batch_size)

    model = tf_resnet_model(img_height, img_width, channels)
    model.compile(optimizer=Adam(learning_rate=3e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(ds_train,
                        validation_data=ds_validation,
                        epochs=TF_NUM_EPOCHS, batch_size=batch_size, verbose=1)

    return model, history


def tf_efficientnet_v2():
    ds_train, ds_validation, _ = format_data(
        train, test, img_height, img_width, batch_size)

    model = tf_efficientnet_model(img_height, img_width, channels)
    model.compile(optimizer=Adam(learning_rate=3e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(ds_train,
                        validation_data=ds_validation,
                        epochs=TF_NUM_EPOCHS, batch_size=batch_size, verbose=1)

    return model, history


def tf_inception():
    ds_train, ds_validation, _ = format_data(
        train, test, img_height, img_width, batch_size)

    model = tf_inception_model(img_height, img_width, channels)
    model.compile(optimizer=Adam(learning_rate=3e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(ds_train,
                        validation_data=ds_validation,
                        epochs=TF_NUM_EPOCHS, batch_size=batch_size, verbose=1)

    return model, history
