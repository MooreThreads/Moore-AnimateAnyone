import os
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download


def prepare_base_model():
    print(f'Preparing base stable-diffusion-v1-5 weights...')
    local_dir = "./pretrained_weights/stable-diffusion-v1-5"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in ["unet/config.json", "unet/diffusion_pytorch_model.bin"]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue
        hf_hub_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )


def prepare_image_encoder():
    print(f"Preparing image encoder weights...")
    local_dir = "./pretrained_weights"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in ["image_encoder/config.json", "image_encoder/pytorch_model.bin"]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue
        hf_hub_download(
            repo_id="lambdalabs/sd-image-variations-diffusers",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )


def prepare_dwpose():
    print(f"Preparing DWPose weights...")
    local_dir = "./pretrained_weights/DWPose"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "dw-ll_ucoco_384.onnx",
        "yolox_l.onnx",
    ]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="yzd-v/DWPose",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )


def prepare_vae():
    print(f"Preparing vae weights...")
    local_dir = "./pretrained_weights/sd-vae-ft-mse"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "config.json",
        "diffusion_pytorch_model.bin",
    ]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )


def prepare_anyone():
    print(f"Preparing AnimateAnyone weights...")
    local_dir = "./pretrained_weights"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "denoising_unet.pth",
        "motion_module.pth",
        "pose_guider.pth",
        "reference_unet.pth",
    ]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="patrolli/AnimateAnyone",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )

if __name__ == '__main__':
    prepare_base_model()
    prepare_image_encoder()
    prepare_dwpose()
    prepare_vae()
    prepare_anyone()
    