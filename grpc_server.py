import json
import os
from torchvision.models.detection import  fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from  torchvision.transforms.functional import pil_to_tensor
import requests
from io import BytesIO
from PIL import Image
import time
import grpc
from concurrent import futures
import inference_pb2
import inference_pb2_grpc

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

class InstanceDetectorService(inference_pb2_grpc.InstanceDetectorServicer):
    def Predict(self, request, context):   
        image_url = request.url
        im = load_img_and_preprocess(image_url)
        result = model(im)[0]['labels']
        objects = [labels[i.item()] for i in result]

        return inference_pb2.InstanceDetectorOutput(objects=objects)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InstanceDetectorServicer_to_server(InstanceDetectorService(), server)
    server.add_insecure_port('[::]:9090')
    server.start()
    print("GRPC ALIVE...")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
