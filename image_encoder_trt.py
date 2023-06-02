import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import onnxruntime
import torch
from torch2trt import torch2trt
import torch_tensorrt
from sam.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# ImageEncoderViT init --> window_size=14
# Block init --> self.window_size = float(windoe_size)

def export_trt(pt_path):
    sam_checkpoint = "checkpoint/sam_vit_h.pth"
    onnx_model_path = "checkpoint/sam_vit_h.onnx"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.eval().to("cuda")


    inputs = [
        torch_tensorrt.Input((1, 3, 1024, 1024)),  # Static NCHW input shape for input #1
        torch.randn((1, 3, 1024, 1024))  # Use an example tensor and let torch_tensorrt infer settings
    ]

    enabled_precisions = {torch.float, torch.half}  # Run with fp16

    trt_ts_module = torch_tensorrt.compile(
        sam.image_encoder, inputs=inputs, enabled_precisions=enabled_precisions, truncate_long_and_double=True
    )
    torch.jit.save("trt_ts.ts")

    # traced = torch.jit.load(pt_path)
    # traced = traced.eval().cuda()

    # create example data
    x = torch.ones((1, 3, 1024, 1024)).cuda()

    # convert to TensorRT feeding sample data as input

    model_trt = torch2trt(sam.image_encoder, [x])
    # model_trt = torch2trt(traced, [x])
    torch.save(model_trt.state_dict(), pt_path.replace(".pt", "_trt.pt"))

if __name__ == "__main__":
    pt_path="/home/hongmin/work/sam/checkpoint/sam_image_encoder.pt"
    export_trt(pt_path)