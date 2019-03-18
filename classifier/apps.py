import numpy as np
from django.apps import AppConfig
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

_session = tf.Session(config=config)

from keras.models import load_model
from keras import backend as K
from PIL import Image

_cnn = None
_graph = None


class ClassifierConfig(AppConfig):
    name = 'classifier'
