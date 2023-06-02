from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from typing import Tuple
import cv2
import json
from kserve.protocol.grpc.grpc_predict_v2_pb2 import ModelInferResponse
from kserve import Model, ModelServer, model_server, InferInput, InferRequest, InferResponse
from kserve.model import PredictorProtocol
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tritonclient.http as httpclient
import time



def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


if __name__=="__main__":

    image = cv2.imread('images/dog.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    newh, neww = get_preprocess_shape(image.shape[0], image.shape[1], 1024)
    image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_LINEAR)
    """Normalize pixel values and pad to a square input."""
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]

    # Normalize colors
    image = (image - pixel_mean) / pixel_std

    # Pad
    h, w = image.shape[:2]
    padh = 1024 - h
    padw = 1024 - w
    print("padh :", padh)
    print("padw :", padw)
    image = np.pad(image, ((0, padh), (0, padw), (0, 0)), mode='constant')
    print("padded.shape :", image.shape)

    # transpose
    image = image.transpose(2, 0, 1).astype(np.float32)
    input_tensor = np.asarray([image], dtype=np.float32)
    print("##############\n", input_tensor.shape)

    infer_inputs = InferInput(name="image_feature", datatype='FP32', shape=list(input_tensor.shape),
                              data=input_tensor)
    # infer_inputs.set_data_from_numpy(input_tensor=input_tensor, binary_data=False)
    raw_data = infer_inputs._raw_data

    request_id = '0'
    infer_request = InferRequest(model_name="sam_image_encoder", infer_inputs=[infer_inputs], request_id=request_id)
    rest = infer_request.to_rest()
    rest["inputs"][0]["data"] = raw_data
    rest["inputs"][0]["data"] = input_tensor
    import orjson

    content = orjson.dumps(rest, option=orjson.OPT_SERIALIZE_NUMPY)


    # to_rest func
    infer_inputs_list = []
    for infer_input in infer_request.inputs:
        infer_input_dict = {
            "name": infer_input.name,
            "shape": infer_input.shape,
            "datatype": infer_input.datatype
        }
        if isinstance(infer_input.data, np.ndarray):
            infer_input.set_data_from_numpy(infer_input.data, binary_data=False)
            infer_input_dict["data"] = infer_input.data
        else:
            infer_input_dict["data"] = infer_input.data
        infer_inputs_list.append(infer_input_dict)
    to_rest_res = {'id': '0', 'inputs': infer_inputs_list}

    import struct
    import orjson
    request_body = infer_request.to_rest()
    data = [val.item() for val in input_tensor.flatten()]
    request_body["inputs"][0]
    request_body = orjson.dumps(infer_request.to_rest())
    json_size = len(request_body)
    binary_data = None
    for infer_input in infer_request.inputs:
        raw_data = infer_input._raw_data
        if raw_data is not None:
            if binary_data is not None:
                binary_data += raw_data
            else:
                binary_data = raw_data

    if binary_data is not None:
        request_body = struct.pack(
            '{}s{}s'.format(len(request_body), len(binary_data)),
            request_body.encode(), binary_data)

    unpack = struct.unpack('{}s{}s'.format(len(request_body), len(binary_data)), request_body)

    # triton
    client = httpclient.InferenceServerClient(url="127.0.0.1:8000")
    inputs = httpclient.InferInput("image_feature", input_tensor.shape, datatype="FP32")
    inputs.set_data_from_numpy(input_tensor, binary_data=False)  # FP16일때는 binary 필수
    # outputs = httpclient.InferRequestedOutput("image_embedding", binary_data=True)

    # Inference
    import requests
    start_time = time.time()
    # res = client.infer(model_name="sam_image_encoder", inputs=[inputs], outputs=[outputs]).as_numpy('image_embedding')
    res = client.infer(model_name="sam_image_encoder", inputs=[inputs])
    # request_uri = "http://127.0.0.1:8000/v2/models/sam_image_encoder/versions/1/infer"
    # res = requests.post(request_uri, data=rest)
    # output = res.get_output("image_embedding")
    # output_data = res.as_numpy("image_embedding")
    end_time = time.time()
    print(end_time-start_time)
    output = res.content

    import requests
    import os
    import json

    with open("kserve/input.json", "r") as f:
        input = json.load(f)

    MODEL_NAME = "sam_image_encoder"
    INPUT_PATH = "@./input.json"
    cmd = "kubectl get inferenceservice sam-image-encoder -o jsonpath='{.status.url}' | cut -d \"/\" -f 3"
    SERVICE_HOSTNAME = os.popen(cmd).read().strip()
    INGRESS_HOST = os.popen("minikube ip").read().strip()
    cmd = "kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name==\"http2\")].nodePort}'"
    INGRESS_PORT = os.popen(cmd).read().strip()

    url = f"http://{INGRESS_HOST}:{INGRESS_PORT}/v2/models/{MODEL_NAME}/infer"
    url = "http://sam-image-encoder.default.35.221.172.86.sslip.io/v2/models/sam_image_encoder/infer"

    st = time.time()
    res = requests.post(url, json=input)
    et = time.time()
    print(et-st)

    output = json.loads(res.content)
    data = output["outputs"][0]["data"]
    shape = output["outputs"][0]["shape"]
    res_arr = np.asarray(data).reshape(shape)

    with open("kserve/output.json", "w") as f:
        json.dump(output, f)
