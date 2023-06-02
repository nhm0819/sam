import os
import tensorrt as trt

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()


def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')


# downloaded the arcface model
ONNX_file_path = './checkpoint/image_encoder/image_encoder.onnx'
engine = build_engine(ONNX_file_path)
engine_file_path = './image_encoder_trt.engine'
with open(engine_file_path, "wb") as f:
    f.write(engine.serialize())