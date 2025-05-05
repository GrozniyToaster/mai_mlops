import json
import os
from torchvision.models.detection import  fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from  torchvision.transforms.functional import pil_to_tensor
from prometheus_client import Counter
from prometheus_flask_exporter import PrometheusMetrics
import requests
from io import BytesIO
from PIL import Image

from flask import Flask, request, jsonify

app = Flask(__name__, static_url_path="")

metrics = PrometheusMetrics(app)
PREDICTION_COUNT = Counter("prediction_count", "Prediction Count", ["type_of_object"])

model_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
preprocess = model_weights.transforms()
model = fasterrcnn_resnet50_fpn_v2(weights=model_weights, box_score_thresh=0.8)
model.eval()

with open(f'{os.path.dirname(os.path.abspath(__file__))}/labels.json', 'r') as f:
    labels_raw = json.loads(f.read())
    labels = {int(index): value for index, value in enumerate(labels_raw)}

def load_img_and_preprocess(img_url):
    byte_img = requests.get(img_url)
    byte_img.raise_for_status()

    img = Image.open(BytesIO(byte_img.content)).convert("RGB")
    tensor_img = pil_to_tensor(img)
    preprocessed_img = preprocess(tensor_img).unsqueeze(0)
    return preprocessed_img

@app.route("/predict", methods=['POST'])
@metrics.gauge("api_in_progress", "requests in progress")
@metrics.counter("app_http_inference_count_total", "number of invocations")
def predict():
    data = request.get_json(force=True)
    im = load_img_and_preprocess(data['url'])
    result = model(im)[0]['labels']
    objects = [labels[i.item()] for i in result]
    
    PREDICTION_COUNT.labels(type_of_object=objects).inc()

    return jsonify({
        "objects": objects
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
