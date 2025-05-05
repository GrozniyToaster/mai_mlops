import sys
import grpc
import inference_pb2
import inference_pb2_grpc
import argparse

def run(img_url, url):
    with grpc.insecure_channel(url) as channel:
        stub = inference_pb2_grpc.InstanceDetectorStub(channel)
        response = stub.Predict(inference_pb2.InstanceDetectorInput(url=img_url))
        print("Objects:", response.objects)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("img_url", help="url of image to download")
    parser.add_argument("--url", help="url to server", default="localhost:9090")
    args = parser.parse_args()


    run(args.img_url, args.url)
