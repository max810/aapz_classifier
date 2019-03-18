import cv2

import numpy as np
from PIL import Image
from django.http import HttpResponse, HttpRequest, HttpResponseBadRequest

# Create your views here.
from django.views.decorators.http import require_POST

CNN = None


def index(request):
    global CNN
    img = Image.open('test_image.jpg')
    img_ready = np.array(img) / 255
    pred_vec = np.squeeze(CNN.predict_on_batch(np.expand_dims(img_ready, axis=0)))
    pred = np.argmax(pred_vec)
    return HttpResponse(pred)


@require_POST
def inference(request: HttpRequest):
    global _cnn
    # with _graph.as_default():
    try:
        pixels = np.frombuffer(request.body)
        img = cv2.imdecode(pixels, cv2.IMREAD_COLOR)
    except ValueError:
        return HttpResponseBadRequest("Invalid image")
    pred_vec = np.squeeze(_cnn.predict_on_batch(np.expand_dims(img, axis=0)))
    pred_class_idx = np.argmax(pred_vec)

    return HttpResponse(pred_class_idx)
