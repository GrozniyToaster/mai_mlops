import argparse
import requests
from furl import furl


def main_single(img_url, server_url):
    predict_url = str(furl(server_url) / "predict")
    print("Sending POST {}".format(predict_url))
    r = requests.post(predict_url, json={'url': img_url})
    print("It is {}".format(r.json()['objects']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img", help="path to img")
    parser.add_argument("--url", help="url to server", default="http://localhost:8080")
    args = parser.parse_args()


    main_single(args.img, args.url)
