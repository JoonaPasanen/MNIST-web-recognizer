import numpy as np
from flask import Flask, request
from flask_cors import CORS
from model import NeuralNetwork
from torch import load, no_grad, tensor, nn
import cv2
import scipy

app = Flask(__name__)
cors = CORS(app)

def predict_number(data):
    model = NeuralNetwork()
    model.load_state_dict(load("./model-data/model_weights.pth", weights_only=True))
    model.eval()
    x = tensor(data).float()
    
    with no_grad():
        output = model(x)
        probabilities = nn.functional.softmax(output, dim=1)
        return probabilities.tolist()
    
def preprocess(data):
    data = np.reshape(data, (400, 400)).astype(np.uint8)
    coords = np.argwhere(data > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    data = data[y_min:y_max+1, x_min:x_max+1]

    h, w = data.shape
    if h > w:
        new_h = 20
        new_w = int(round(20 * w / h))
    else:
        new_w = 20
        new_h = int(round(20 * h / w))

    data = cv2.resize(data, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cm_y, cm_x = scipy.ndimage.center_of_mass(data)
    shift_y = int(round(14 - cm_y))
    shift_x = int(round(14 - cm_x))
    aM = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    data = cv2.warpAffine(data, aM, (28, 28), flags=cv2.INTER_NEAREST, borderValue=0)

    return np.matrix(data).flatten("C")
    
@app.route("/", methods=["POST"])
def index():
    json = request.json
    data = json["data"]
    processed_data = preprocess(data)
    return predict_number(processed_data)