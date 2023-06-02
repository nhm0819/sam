import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import onnxruntime
import torch

if __name__ == '__main__':
    sam_checkpoint = "checkpoint/sam_vit_h.pth"
    onnx_model_path = "checkpoint/sam_vit_h.onnx"
    model_type = "vit_h"
    device = "cuda"

    # ort_session = onnxruntime.InferenceSession(onnx_model_path,
    #                                            providers=["CUDAExecutionProvider"])

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)
    predictor = SamPredictor(sam)

    image = cv2.imread('images/dog.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ### TODO: image encoder export to ONNX model
    predictor.set_image(image)

    input_image = predictor.transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device="cpu")
    transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    # input_size = tuple(transformed_image.shape[-2:])
    input_image = sam.preprocess(transformed_image)
    features = sam.image_encoder(input_image)

    dummy_inputs = {
        "image_feature": input_image,
    }
    dynamic_axes = {
        'image_feature': {0: 'batch', 2: 'height', 3: 'width'},
    }

    output = "/home/hongmin/work/sam/checkpoint/model.onnx/model.onnx"
    torch.onnx.export(
        sam.image_encoder,
        input_image, # tuple(dummy_inputs.values()),
        output,
        export_params=True,
        verbose=False,
        opset_version=14,
        do_constant_folding=True,
        input_names=["image_feature"],
        output_names=["image_embedding"],
        dynamic_axes=dynamic_axes,
    )