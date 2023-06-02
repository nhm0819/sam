import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sam.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import onnxruntime
import torch


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


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
        "image": input_image,
    }
    dynamic_axes = {
        'image': {2: 'height', 3: 'width'},
    }

    output = "/home/hongmin/work/sam/checkpoint/sam_image_encoder/image_encoder.onnx"
    torch.onnx.export(
        sam.image_encoder,
        input_image, # tuple(dummy_inputs.values()),
        output,
        export_params=True,
        verbose=False,
        opset_version=16,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["features"],
        dynamic_axes=dynamic_axes,
    )

    def onnx_test(ort_session, batch_tensor):
        res = []
        res_time = []
        for i in range(len(batch_tensor)):
            st = time.time()
            res.append(ort_session.run(None, {"image": batch_tensor[i].unsqueeze(0).numpy()}))
            # res_torch = sam.image_encoder(input_image)
            et = time.time()
            res_time.append(et - st)
            print(et - st)
        return res, res_time

    ### image encoder
    ort_session = onnxruntime.InferenceSession("/home/hongmin/work/sam/checkpoint/image_encoder/image_encoder.onnx",
                                               providers=["CUDAExecutionProvider"])
                                               # providers = ["CPUExecutionProvider"])
    batch_tensor = torch.randn((100, 3, 1024, 1024))

    ort_session.set_providers(["CUDAExecutionProvider"])
    res_onnx_gpu, res_onnx_gpu_time = onnx_test(ort_session, batch_tensor)

    ort_session.set_providers(["CPUExecutionProvider"])
    res_onnx_cpu, res_onnx_cpu_time = onnx_test(ort_session, batch_tensor)



    # dif = res_onnx[0] - res_torch.detach().numpy()
    # dif[0].mean()

    import torch
    import torch_tensorrt

    inputs = [
        torch_tensorrt.Input((1, 3, 1024, 1024)),  # Static NCHW input shape for input #1
        torch.randn((1, 3, 1024, 1024))  # Use an example tensor and let torch_tensorrt infer settings
    ]

    enabled_precisions = {torch.float, torch.half}  # Run with fp16
    traced = torch.jit.load("/home/hongmin/work/sam/checkpoint/sam_image_encoder.pt")
    traced = traced.eval().cuda()






    ### Visualization

    input_point = np.array([[540, 350], [700,200], [660, 100]])
    input_label = np.array([1, 1, 0])

    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

    onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    ort_inputs = {
        "image_embeddings": image_embedding.cpu().numpy(),
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
    }
    masks, scores, low_res_logits = ort_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold
    masks = masks[0]

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        show_mask(mask, plt.gca(), random_color=True)
        show_points(input_point, input_label, plt.gca())
    # plt.axis('off')
    plt.show()






    mask_generator = SamAutomaticMaskGenerator(sam)
    image = cv2.imread('images/dog.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st = time.time()
    masks = mask_generator.generate(image)
    et = time.time()
    print(f"inference time : {et-st:.2f}s")
    print(len(masks))
    print(masks[0].keys())

    # plt.figure(figsize=(20, 20))
    # plt.imshow(image)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show()

    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    st = time.time()
    masks2 = mask_generator_2.generate(image)
    et = time.time()
    print(f"inference time : {et-st:.2f}s")
    print(len(masks2))
    print(masks2[0].keys())

    # plt.figure(figsize=(20, 20))
    # plt.imshow(image)
    # show_anns(masks2)
    # plt.axis('off')
    # plt.show()

    predictor = SamPredictor(sam)
    st = time.time()
    predictor.set_image(image)

    input_point = np.array([[540, 350]])
    input_label = np.array([1])

    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('on')
    # plt.show()

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    et = time.time()
    print(f"inference time : {et-st:.2f}s")

    print(f"{masks.shape} : (number_of_masks) x H x W")

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show()

    ##################################
    input_point = np.array([[540, 350], [700,200], [660, 100]])
    input_label = np.array([1, 1, 0])
    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    input_point = np.array([[540, 350], [700, 200], [700, 100]])
    input_label = np.array([1, 0, 0])

    st = time.time()
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    et = time.time()
    print(f"inference time : {et-st:.2f}s")

    print(f"{masks.shape} : (number_of_masks) x H x W")
    print(scores)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        show_mask(mask, plt.gca(), random_color=True)
        show_points(input_point, input_label, plt.gca())
    # plt.axis('off')
    plt.show()

