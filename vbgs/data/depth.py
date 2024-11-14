from PIL import Image
import numpy as np
import torch


def load_depth_model(model_name, device):
    if model_name == "zoe":
        return load_zoe(device)
    else:
        return load_dpt(device)


def load_dpt(device):
    model_type = "DPT_Large"
    # model_type = "DPT_Hybrid"
    # model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if "small" in model_type:
        depth_transforms = midas_transforms.dpt_transform
    else:
        depth_transforms = midas_transforms.small_transform
    depth_model = midas

    def preprocess(x):
        return depth_transforms(x).to(device)

    def post_process(x):
        scale = 0.000305
        shift = 0.1378

        depth = scale * x + shift
        depth = 1 / depth.cpu().detach().numpy()
        depth[depth < 1e-8] = 1e-8
        return depth

    return depth_model, preprocess, post_process


def load_zoe(device):
    zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
    zoe = zoe.to(device)

    def model(img):
        return zoe.infer_pil(
            Image.fromarray(img.astype(np.uint8)), output_type="tensor"
        ).unsqueeze(0)

    return model, lambda x: x, lambda x: x


def predict_depth(rgb, depth_model, preprocess, postprocess):
    x = preprocess(rgb)

    with torch.no_grad():
        prediction = depth_model(x)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = postprocess(prediction)

    return depth
