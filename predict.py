# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import subprocess
import time
from cog import BasePredictor, Input, Path
import os
from datetime import datetime

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid

MOORE_ANIMATEANYONE_CACHE = "./pretrained_weights"
MOORE_ANIMATEANYONE_URL = "https://storage.googleapis.com/replicate-weights/Moore-AnimateAnyone/pretrained_weights.tar"


class AnimateController:
    def __init__(
        self,
        config_path="./configs/prompts/animation.yaml",
        weight_dtype=torch.float16,
    ):
        # Read pretrained weights path from config
        self.config = OmegaConf.load(config_path)
        self.pipeline = None
        self.weight_dtype = weight_dtype

    def animate(
        self,
        ref_image,
        pose_video_path,
        width=512,
        height=768,
        length=24,
        num_inference_steps=25,
        cfg=3.5,
        seed=123,
    ):
        generator = torch.manual_seed(seed)
        if isinstance(ref_image, np.ndarray):
            ref_image = Image.fromarray(ref_image)
        if self.pipeline is None:
            vae = AutoencoderKL.from_pretrained(
                self.config.pretrained_vae_path,
                cache_dir=MOORE_ANIMATEANYONE_CACHE,
                local_files_only=True,
            ).to("cuda", dtype=self.weight_dtype)

            reference_unet = UNet2DConditionModel.from_pretrained(
                self.config.pretrained_base_model_path,
                subfolder="unet",
                cache_dir=MOORE_ANIMATEANYONE_CACHE,
                local_files_only=True,
            ).to(dtype=self.weight_dtype, device="cuda")

            inference_config_path = self.config.inference_config
            infer_config = OmegaConf.load(inference_config_path)
            denoising_unet = UNet3DConditionModel.from_pretrained_2d(
                self.config.pretrained_base_model_path,
                self.config.motion_module_path,
                subfolder="unet",
                unet_additional_kwargs=infer_config.unet_additional_kwargs,
            ).to(dtype=self.weight_dtype, device="cuda")

            pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
                dtype=self.weight_dtype, device="cuda"
            )

            image_enc = CLIPVisionModelWithProjection.from_pretrained(
                self.config.image_encoder_path,
                cache_dir=MOORE_ANIMATEANYONE_CACHE,
                local_files_only=True,
            ).to(dtype=self.weight_dtype, device="cuda")
            sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
            scheduler = DDIMScheduler(**sched_kwargs)

            # load pretrained weights
            denoising_unet.load_state_dict(
                torch.load(self.config.denoising_unet_path, map_location="cpu"),
                strict=False,
            )
            reference_unet.load_state_dict(
                torch.load(self.config.reference_unet_path, map_location="cpu"),
            )
            pose_guider.load_state_dict(
                torch.load(self.config.pose_guider_path, map_location="cpu"),
            )

            pipe = Pose2VideoPipeline(
                vae=vae,
                image_encoder=image_enc,
                reference_unet=reference_unet,
                denoising_unet=denoising_unet,
                pose_guider=pose_guider,
                scheduler=scheduler,
            )
            pipe = pipe.to("cuda", dtype=self.weight_dtype)
            self.pipeline = pipe

        pose_images = read_frames(pose_video_path)
        src_fps = get_fps(pose_video_path)

        pose_list = []
        pose_tensor_list = []
        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )
        for pose_image_pil in pose_images[:length]:
            pose_list.append(pose_image_pil)
            pose_tensor_list.append(pose_transform(pose_image_pil))

        video = self.pipeline(
            ref_image,
            pose_list,
            width=width,
            height=height,
            video_length=length,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg,
            generator=generator,
        ).videos

        ref_image_tensor = pose_transform(ref_image)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
        ref_image_tensor = repeat(
            ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=length
        )
        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0)

        # ref_image_tensor = ref_image_tensor[:, :3, :, :, :]
        # video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)

        save_dir = f"./output/gradio"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M")
        out_path = os.path.join(save_dir, f"{date_str}T{time_str}.mp4")
        save_videos_grid(
            video,
            out_path,
            n_rows=1,
            fps=src_fps,
        )

        torch.cuda.empty_cache()

        return out_path


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MOORE_ANIMATEANYONE_CACHE):
            download_weights(MOORE_ANIMATEANYONE_URL, MOORE_ANIMATEANYONE_CACHE)

        self.controller = AnimateController()

    def predict(
        self,
        reference_image: Path = Input(description="Path to the reference image"),
        motion_sequence: Path = Input(description="Path to the motion sequence video"),
        width: int = Input(
            description="Desired width of the output video",
            default=512,
            ge=448,
            le=768,
        ),
        height: int = Input(
            description="Desired height of the output video",
            default=768,
            ge=512,
            le=1024,
        ),
        length: int = Input(
            description="Desired length of the output video in frames",
            default=24,
            ge=24,
            le=128,
        ),
        sampling_steps: int = Input(
            description="Number of sampling steps for the animation",
            default=25,
            ge=10,
            le=30,
        ),
        guidance_scale: float = Input(
            description="Scale for guidance during animation generation",
            default=3.5,
            ge=2.0,
            le=10.0,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        reference_image = Image.open(str(reference_image))
        motion_sequence = str(motion_sequence)

        # Call the animate method from the controller
        animation = self.controller.animate(
            reference_image,
            motion_sequence,
            width,
            height,
            length,
            sampling_steps,
            guidance_scale,
            seed,
        )

        return Path(animation)
