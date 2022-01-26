from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import tensorflow as tf


def augment(x, y):
    return tf.image.random_brightness(x, max_delta=0.05), y


def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def format_data(train, test, img_height, img_width, batch_size):
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    AUTOTUNE = tf.data.AUTOTUNE

    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        train,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        seed=123,
        validation_split=0.1,
        subset="training"
    )

    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        train,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        seed=123,
        validation_split=0.1,
        subset="validation"
    )

    ds_test = tf.keras.preprocessing.image_dataset_from_directory(
        test,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=False,
        seed=123,

    )

    ds_train = ds_train.map(augment).map(
        lambda x, y: (normalization_layer(x), y))
    ds_validation = ds_validation.map(lambda x, y: (normalization_layer(x), y))
    ds_test = ds_test.map(lambda x, y: (normalization_layer(x), y))

    ds_train = ds_train.cache().prefetch(buffer_size=AUTOTUNE)
    ds_validation = ds_validation.cache().prefetch(buffer_size=AUTOTUNE)
    ds_test = ds_test.cache().prefetch(buffer_size=AUTOTUNE)

    return ds_train, ds_validation, ds_test
