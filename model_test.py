import time
import numpy as np
from segment_anything import sam_model_registry
import onnxruntime
import torch
import pickle


if __name__ == '__main__':
    sam_checkpoint = "checkpoint/sam_vit_h.pth"
    onnx_model_path = "checkpoint/sam_vit_h.onnx"
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    image_encoder = sam.image_encoder
    image_encoder.eval().to(device=device)
    del sam
    # predictor = SamPredictor(sam)

    torch.manual_seed(0)
    batch_tensor = torch.randn((100, 3, 1024, 1024), device=device)

    def torch_test(model, batch_tensor):
        res = []
        res_time = []
        with torch.no_grad():
            for i in range(len(batch_tensor)):
                st = time.time()
                model_res = model(batch_tensor[i].unsqueeze(0))
                res.append(model_res.to("cpu").numpy())
                et = time.time()
                res_time.append(et - st)
                print(et - st)
                del model_res
                torch.cuda.empty_cache()
        return res, res_time


    def onnx_test(ort_session, batch_tensor):
        res = []
        res_time = []
        for i in range(len(batch_tensor)):
            st = time.time()
            res.append(ort_session.run(None, {"image": batch_tensor[i].unsqueeze(0).to("cpu").numpy()}))
            # res_torch = sam.image_encoder(input_image)
            et = time.time()
            res_time.append(et - st)
            print(et - st)
        return res, res_time


    ### image encoder
    res_torch_gpu, res_torch_gpu_time = torch_test(image_encoder, batch_tensor)
    with open("test_result/res_torch_gpu.pkl", 'wb') as f:
        pickle.dump(res_torch_gpu, f)
    with open("test_result/res_torch_gpu_time.pkl", 'wb') as f:
        pickle.dump(res_torch_gpu_time, f)
    del image_encoder
    torch.cuda.empty_cache()

    ort_session = onnxruntime.InferenceSession("/home/hongmin/work/sam/checkpoint/image_encoder/image_encoder.onnx",
                                               providers=["CUDAExecutionProvider"])
    # providers = ["CPUExecutionProvider"])

    ort_session.set_providers(["CPUExecutionProvider"])
    res_onnx_cpu, res_onnx_cpu_time = onnx_test(ort_session, batch_tensor)
    with open("test_result/res_onnx_cpu.pkl", 'wb') as f:
        pickle.dump(res_onnx_cpu, f)
    with open("test_result/res_onnx_cpu_time.pkl", 'wb') as f:
        pickle.dump(res_onnx_cpu_time, f)

    ort_session.set_providers(["CUDAExecutionProvider"])
    res_onnx_gpu, res_onnx_gpu_time = onnx_test(ort_session, batch_tensor)
    with open("test_result/res_onnx_gpu.pkl", 'wb') as f:
        pickle.dump(res_onnx_gpu, f)
    with open("test_result/res_onnx_gpu_time.pkl", 'wb') as f:
        pickle.dump(res_onnx_gpu_time, f)

    ort_session = onnxruntime.InferenceSession("/home/hongmin/work/sam/image_encoder_fp16.onnx",
                                               providers=["CUDAExecutionProvider"])
    res_onnx_fp16, res_onnx_fp16_time = onnx_test(ort_session, batch_tensor)
    with open("test_result/res_onnx_fp16.pkl", 'wb') as f:
        pickle.dump(res_onnx_fp16, f)
    with open("test_result/res_onnx_fp16_time.pkl", 'wb') as f:
        pickle.dump(res_onnx_fp16_time, f)

    del ort_session

    # dif = res_onnx[0] - res_torch.detach().numpy()
    # dif[0].mean()


    with open("test_result/res_torch_gpu.pkl", 'rb') as f:
        res_torch_gpu = np.array(pickle.load(f)).squeeze()
    with open("test_result/res_torch_gpu_time.pkl", 'rb') as f:
        res_torch_gpu_time = np.array(pickle.load(f))

    with open("test_result/res_onnx_gpu.pkl", 'rb') as f:
        res_onnx_gpu = np.array(pickle.load(f)).squeeze()
    with open("test_result/res_onnx_gpu_time.pkl", 'rb') as f:
        res_onnx_gpu_time = np.array(pickle.load(f))

    with open("test_result/res_onnx_cpu.pkl", 'rb') as f:
        res_onnx_cpu = np.array(pickle.load(f)).squeeze()
    with open("test_result/res_onnx_cpu_time.pkl", 'rb') as f:
        res_onnx_cpu_time = np.array(pickle.load(f))

    with open("test_result/res_onnx_fp16.pkl", 'rb') as f:
        res_onnx_fp16 = np.array(pickle.load(f)).squeeze()
    with open("test_result/res_onnx_fp16_time.pkl", 'rb') as f:
        res_onnx_fp16_time = np.array(pickle.load(f))

    mean_time = {"torch_gpu": res_torch_gpu_time.mean(),
                 "onnx_cpu": res_onnx_cpu_time.mean(),
                 "onnx_gpu": res_onnx_gpu_time.mean(),
                 "onnx_fp16": res_onnx_fp16_time.mean(),
                 }
    mean_diff = {"torch_gpu": np.power(res_torch_gpu - res_torch_gpu, 2).sum(),
                 "onnx_cpu": np.power(res_torch_gpu - res_onnx_cpu, 2).sum(),
                 "onnx_gpu": np.power(res_torch_gpu - res_onnx_gpu, 2).sum(),
                 "onnx_fp16": np.power(res_torch_gpu - res_onnx_fp16, 2).sum(),
                 }


