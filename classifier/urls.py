import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

_session = tf.Session(config=config)

from keras.models import load_model
from keras import backend as K
from PIL import Image

from django.urls import path
from . import views

import classifier.views as V

urlpatterns = [
    path('', views.index, name='index'),
    path('inference', views.inference, name='inference'),
]

K.set_session(_session)
cnn = load_model("cnn_drv_dist_keras_model.h5")
img = Image.open('test_image.jpg')
img_ready = np.array(img) / 255
pred_vec = np.squeeze(cnn.predict_on_batch(np.expand_dims(img_ready, axis=0)))
pred = np.argmax(pred_vec)
assert pred == 0, "Something wrong with the image/model."
print("CNN ready")
V.CNN = cnn

graph = tf.get_default_graph()
_cnn, _graph, = cnn, graph,
print("Graph ready.")
