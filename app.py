import os
import random
from datetime import datetime

import gradio as gr
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
            ).to("cuda", dtype=self.weight_dtype)

            reference_unet = UNet2DConditionModel.from_pretrained(
                self.config.pretrained_base_model_path,
                subfolder="unet",
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
                self.config.image_encoder_path
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
        video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)

        save_dir = f"./output/gradio"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M")
        out_path = os.path.join(save_dir, f"{date_str}T{time_str}.mp4")
        save_videos_grid(
            video,
            out_path,
            n_rows=3,
            fps=src_fps,
        )

        torch.cuda.empty_cache()

        return out_path


controller = AnimateController()


def ui():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Moore-AnimateAnyone Demo
            """
        )
        animation = gr.Video(
            format="mp4",
            label="Animation Results",
            height=448,
            autoplay=True,
        )

        with gr.Row():
            reference_image = gr.Image(label="Reference Image")
            motion_sequence = gr.Video(
                format="mp4", label="Motion Sequence", height=512
            )

            with gr.Column():
                width_slider = gr.Slider(
                    label="Width", minimum=448, maximum=768, value=512, step=64
                )
                height_slider = gr.Slider(
                    label="Height", minimum=512, maximum=1024, value=768, step=64
                )
                length_slider = gr.Slider(
                    label="Video Length", minimum=24, maximum=128, value=24, step=24
                )
                with gr.Row():
                    seed_textbox = gr.Textbox(label="Seed", value=-1)
                    seed_button = gr.Button(
                        value="\U0001F3B2", elem_classes="toolbutton"
                    )
                    seed_button.click(
                        fn=lambda: gr.Textbox.update(value=random.randint(1, 1e8)),
                        inputs=[],
                        outputs=[seed_textbox],
                    )
                with gr.Row():
                    sampling_steps = gr.Slider(
                        label="Sampling steps",
                        value=25,
                        info="default: 25",
                        step=5,
                        maximum=30,
                        minimum=10,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        value=3.5,
                        info="default: 3.5",
                        step=0.5,
                        maximum=10,
                        minimum=2.0,
                    )
                submit = gr.Button("Animate")

        def read_video(video):
            return video

        def read_image(image):
            return Image.fromarray(image)

        # when user uploads a new video
        motion_sequence.upload(read_video, motion_sequence, motion_sequence)
        # when `first_frame` is updated
        reference_image.upload(read_image, reference_image, reference_image)
        # when the `submit` button is clicked
        submit.click(
            controller.animate,
            [
                reference_image,
                motion_sequence,
                width_slider,
                height_slider,
                length_slider,
                sampling_steps,
                guidance_scale,
                seed_textbox,
            ],
            animation,
        )

        # Examples
        gr.Markdown("## Examples")
        gr.Examples(
            examples=[
                [
                    "./configs/inference/ref_images/anyone-5.png",
                    "./configs/inference/pose_videos/anyone-video-2_kps.mp4",
                ],
                [
                    "./configs/inference/ref_images/anyone-10.png",
                    "./configs/inference/pose_videos/anyone-video-1_kps.mp4",
                ],
                [
                    "./configs/inference/ref_images/anyone-2.png",
                    "./configs/inference/pose_videos/anyone-video-5_kps.mp4",
                ],
            ],
            inputs=[reference_image, motion_sequence],
            outputs=animation,
        )

    return demo


demo = ui()
demo.launch(share=True)
