import cv2

import numpy as np
from PIL import Image
from django.http import HttpResponse, HttpRequest, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt

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
@csrf_exempt
def inference(request: HttpRequest):
    global CNN
    # with _graph.as_default():
    try:
        pixels = np.frombuffer(request.body, dtype='uint8')
        img = cv2.imdecode(pixels, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (200, 150), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(img)
        img = img / 255.
    except (ValueError, cv2.error):
        return HttpResponseBadRequest("Invalid image")
    pred_vec = np.squeeze(CNN.predict_on_batch(np.expand_dims(img, axis=0)))
    pred_class_idx = np.argmax(pred_vec)

    return HttpResponse(pred_class_idx)
