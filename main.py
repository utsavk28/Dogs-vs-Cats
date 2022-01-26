# 0 -> cat, 1 -> dog
import os
import tensorflow as tf
from cnn import cnn
from transfer_learning import tf_resnet
from utils import plot_history
from utils import format_data
from config import  batch_size, img_height, img_width,train, test

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def experiment1():
    print(1)
    ds_train, ds_validation, ds_test = format_data(
        train, test, img_height, img_width, batch_size)
    print(1)
    model, history = cnn()
    print(1)
    model.evaluate(ds_train)
    model.evaluate(ds_validation)
    model.evaluate(ds_test)

    plot_history(history)
    model.summary()


def experiment2():
    ds_train, ds_validation, ds_test = format_data(
        train, test, img_height, img_width, batch_size)
    model, history = tf_resnet()
    model.evaluate(ds_train)
    model.evaluate(ds_validation)
    model.evaluate(ds_test)

    plot_history(history)
    model.summary()


experiments = [experiment1, experiment2]

EXP_NO = os.environ.get('EXP_NO')

from models import cnn_model

if __name__ == "__main__":
    experiment = experiments[EXP_NO]
    experiment() 
