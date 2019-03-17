import base64
import socketio
import eventlet
import numpy as np
import matplotlib.cm as cm
import cv2
import tensorflow as tf
from io import BytesIO
from PIL import Image
from flask import Flask
from tensorflow.python.keras.models import load_model
# from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from vis.visualization import visualize_cam, overlay

from build_model import build_model


sio = socketio.Server()
app = Flask(__name__)  # __main__

SPEED_LIMIT = 13


def img_preprocess(img):
    img = img[40:140, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


# def display_heatmap(img, original_image):
#     right_heatmap = visualize_cam(
#         model2, layer_idx=-1, filter_indices=0, seed_input=img,
#         grad_modifier=None
#     )
#     jet_right_heatmap = np.uint8(cm.jet(right_heatmap)[..., :3] * 150)
#     left_heatmap = visualize_cam(
#         model2, layer_idx=-1, filter_indices=0, seed_input=img,
#         grad_modifier='negate'
#     )
#     jet_left_heatmap = np.uint8(cm.jet(left_heatmap)[..., :3] * 150)
#     maintain_heatmap = visualize_cam(
#         model2, layer_idx=-1, filter_indices=0, seed_input=img,
#         grad_modifier='small_values'
#     )
#     jet_maintain_heatmap = np.uint8(cm.jet(maintain_heatmap)[..., :3] * 150)
#
#     original_image = cv2.resize(original_image, (200, 66))
#     cv2.imshow('Right steering', overlay(jet_right_heatmap, original_image, alpha=0.4))
#     cv2.imshow('Left steering', overlay(jet_left_heatmap, original_image, alpha=0.4))
#     cv2.imshow('Maintain steering', overlay(jet_maintain_heatmap, original_image, alpha=0.4))
#
#     k = cv2.waitKey(10)
#     if k == ord('q'):
#         cv2.destroyAllWindows()


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    # original_image = image.copy()
    image = img_preprocess(image)

    # pre_process_image = image.copy()

    image = np.array([image])
    steering_angle = float(model.predict(image))

    throttle = 1.0 - (speed / SPEED_LIMIT)
    throttle = 0.4
    send_control(steering_angle, throttle)
    # display_heatmap(pre_process_image, original_image)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    model = load_model('model.h5')
    model2 = build_model()
    model2.load_weights('./model_weights.h5')

    app = socketio.Middleware(socketio_app=sio, wsgi_app=app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
