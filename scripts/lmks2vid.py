import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List

import av
import cv2
import numpy as np
import torch

# 初始化模型
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
)

import sys
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_lmks2vid_long import Pose2VideoPipeline
from src.models.pose_guider import PoseGuider
from src.utils.util import get_fps, read_frames, save_videos_grid
from tools.facetracker_api import face_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path of inference configs",
        default="./configs/prompts/inference_reenact.yaml"
    )
    parser.add_argument(
        "--save_dir", type=str, help="Path of save results",
        default="./output/stage2_infer"
    )
    parser.add_argument(
        "--source_image_path", type=str, help="Path of source image", 
        default="",
    )
    parser.add_argument(
        "--driving_video_path", type=str, help="Path of driving video", 
        default="",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=320,
        help="Checkpoint step of pretrained model",
    )
    parser.add_argument("--mask_ratio", type=float, default=0.55)   # 0.55~0.6
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    return args


def lmks_vis(img, lms):
    # Visualize the mouth, nose, and entire face based on landmarks
    h, w, c = img.shape
    lms = lms[:, :2]
    mouth = lms[48:66]
    nose = lms[27:36]
    color = (0, 255, 0)
    # Center mouth and nose
    x_c, y_c = np.mean(lms[:, 0]), np.mean(lms[:, 1])
    h_c, w_c = h // 2, w // 2
    img_face, img_mouth, img_nose = img.copy(), img.copy(), img.copy()
    for pt_num, (x, y) in enumerate(mouth):
        x = x - (x_c - w_c)
        y = y - (y_c - h_c)
        x = int(x + 0.5)
        y = int(y + 0.5)
        cv2.circle(img_mouth, (y, x), 1, color, -1)
    for pt_num, (x, y) in enumerate(nose):
        x = x - (x_c - w_c)
        y = y - (y_c - h_c)
        x = int(x + 0.5)
        y = int(y + 0.5)
        cv2.circle(img_nose, (y, x), 1, color, -1)
    for pt_num, (x, y) in enumerate(lms):
        x = int(x + 0.5)
        y = int(y + 0.5)
        if pt_num >= 66:
            color = (255, 255, 0)
        else:
            color = (0, 255, 0)
        cv2.circle(img_face, (y, x), 1, color, -1)
    return img_face, img_mouth, img_nose


def batch_rearrange(pose_len, batch_size=24):
    # To rearrange the pose sequence based on batch size 
    batch_ind_list = []
    for i in range(0, pose_len, batch_size):
        if i + batch_size < pose_len:
            batch_ind_list.append(list(range(i, i + batch_size)))
        else:
            batch_ind_list.append(list(range(i, min(i + batch_size, pose_len))))
    return batch_ind_list


def lmks_video_extract(video_path):
    # To extract the landmark sequence of video (single face video)
    video_stream = cv2.VideoCapture(video_path)
    lmks_list, frames = [], []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        h, w, c = frame.shape
        lmk_img, lmks = face_image(frame)
        if lmks is not None:
            lmks_list.append(lmks)
            frames.append(frame)
    return frames, np.array(lmks_list), [h, w]


def adjust_pose(src_lms_list, src_size, ref_lms, ref_size):
    # To align the center of source landmarks based on reference landmark
    new_src_lms_list = []
    ref_lms = ref_lms[:, :2]
    src_lms = src_lms_list[0][:, :2]
    ref_lms[:, 0] = ref_lms[:, 0] / ref_size[1]
    ref_lms[:, 1] = ref_lms[:, 1] / ref_size[0]
    src_lms[:, 0] = src_lms[:, 0] / src_size[1]
    src_lms[:, 1] = src_lms[:, 1] / src_size[0]
    ref_cx, ref_cy = np.mean(ref_lms[:, 0]), np.mean(ref_lms[:, 1])
    src_cx, src_cy = np.mean(src_lms[:, 0]), np.mean(src_lms[:, 1])
    for item in src_lms_list:
        item = item[:, :2]
        item[:, 0] = item[:, 0] - int((src_cx - ref_cx)) * src_size[1]
        item[:, 1] = item[:, 1] - int((src_cy - ref_cy)) * src_size[0]
        new_src_lms_list.append(item)
    return np.array(new_src_lms_list)
        

def main():
    args = parse_args()
    infer_config = OmegaConf.load(args.config)

    # base_model_path = "./pretrained_weights/huggingface-models/sd-image-variations-diffusers/"
    base_model_path = infer_config.pretrained_base_model_path
    weight_dtype = torch.float16

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        # "./pretrained_weights/huggingface-models/sd-image-variations-diffusers/image_encoder"
        infer_config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")
    vae = AutoencoderKL.from_pretrained(
        # "./pretrained_weights/huggingface-models/sd-vae-ft-mse"
        infer_config.pretrained_vae_path
    ).to("cuda", dtype=weight_dtype)
    # initial reference unet, denoise unet, pose guider
    reference_unet = UNet3DConditionModel.from_pretrained_2d(
        base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "task_type": "reenact",
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
            "mode": "write",
        },
    ).to(device="cuda", dtype=weight_dtype)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        base_model_path,
        "./pretrained_weights/mm_sd_v15_v2.ckpt",
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            infer_config.unet_additional_kwargs
        ),
        # mm_zero_proj_out=True,
    ).to(device="cuda")
    pose_guider1 = PoseGuider(
        conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
    ).to(device="cuda", dtype=weight_dtype)
    pose_guider2 = PoseGuider(
        conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
    ).to(device="cuda", dtype=weight_dtype)
    print("------------------initial all networks------------------")
    # load model from pretrained models
    denoising_unet.load_state_dict(
        torch.load(
            infer_config.denoising_unet_path,
            map_location="cpu",
        ),
        strict=True,
    )
    reference_unet.load_state_dict(
        torch.load(
            infer_config.reference_unet_path,
            map_location="cpu",
        )
    )
    pose_guider1.load_state_dict(
        torch.load(
            infer_config.pose_guider1_path,
            map_location="cpu",
        )
    )
    pose_guider2.load_state_dict(
        torch.load(
            infer_config.pose_guider2_path,
            map_location="cpu",
        )
    )
    print("---------load pretrained denoising unet, reference unet and pose guider----------")
    # scheduler
    enable_zero_snr = True
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    if enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    scheduler = DDIMScheduler(**sched_kwargs)
    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider1=pose_guider1,
        pose_guider2=pose_guider2,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)
    height, width, clip_length = args.H, args.W, args.L
    generator = torch.manual_seed(42)
    date_str = datetime.now().strftime("%Y%m%d")
    save_dir = Path(f"{args.save_dir}/{date_str}")
    save_dir.mkdir(exist_ok=True, parents=True)
    
    ref_image_path, pose_video_path = args.source_image_path, args.driving_video_path
    ref_name = Path(ref_image_path).stem
    pose_name = Path(pose_video_path).stem
    ref_image_pil = Image.open(ref_image_path).convert("RGB")
    ref_image = cv2.imread(ref_image_path)
    ref_h, ref_w, c = ref_image.shape
    ref_pose, ref_pose_lms = face_image(ref_image)
    # To extract landmarks from driving video
    pose_frames, pose_lms_list, pose_size = lmks_video_extract(pose_video_path)
    pose_lms_list = adjust_pose(pose_lms_list, pose_size, ref_pose_lms, [ref_h, ref_w])
    pose_h, pose_w = int(pose_size[0]), int(pose_size[1])
    pose_len = pose_lms_list.shape[0]
    # Truncating the video tail if its frames less than 24 to obtain stable effect.
    pose_len = pose_len // 24 * 24
    batch_index_list = batch_rearrange(pose_len, args.batch_size)
    pose_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )
    videos = []
    zero_map = np.zeros_like(ref_pose)
    zero_map = cv2.resize(zero_map, (pose_w, pose_h))
    for batch_index in batch_index_list:
        pose_list, pose_up_list, pose_down_list = [], [], []
        pose_frame_list = []
        pose_tensor_list, pose_up_tensor_list, pose_down_tensor_list = [], [], []
        batch_len = len(batch_index)
        for pose_idx in batch_index:
            pose_lms = pose_lms_list[pose_idx]
            pose_frame = pose_frames[pose_idx][:, :, ::-1]
            pose_image, pose_mouth_image, _ = lmks_vis(zero_map, pose_lms)
            h, w, c = pose_image.shape
            pose_up_image = pose_image.copy()
            pose_up_image[int(h * args.mask_ratio):, :, :] = 0.
            pose_image_pil = Image.fromarray(pose_image)
            pose_frame = Image.fromarray(pose_frame)
            pose_up_pil = Image.fromarray(pose_up_image)
            pose_mouth_pil = Image.fromarray(pose_mouth_image)
            pose_list.append(pose_image_pil)
            pose_up_list.append(pose_up_pil)
            pose_down_list.append(pose_mouth_pil)
            pose_tensor_list.append(pose_transform(pose_image_pil))
            pose_up_tensor_list.append(pose_transform(pose_up_pil))
            pose_down_tensor_list.append(pose_transform(pose_mouth_pil))
            pose_frame_list.append(pose_transform(pose_frame))
        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0)
        pose_frames_tensor = torch.stack(pose_frame_list, dim=0)  # (f, c, h, w)
        pose_frames_tensor = pose_frames_tensor.transpose(0, 1)
        pose_frames_tensor = pose_frames_tensor.unsqueeze(0)
        ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
        ref_image_tensor = repeat(
            ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=batch_len
        )
        # To disentangle head attitude control (including eyes blink) and mouth motion control
        pipeline_output = pipe(
            ref_image_pil,
            pose_up_list,
            pose_down_list,
            width,
            height,
            batch_len,
            20,
            3.5,
            generator=generator,
        )
        video = pipeline_output.videos
        video = torch.cat([ref_image_tensor, pose_frames_tensor, video], dim=0)
        videos.append(video)
    videos = torch.cat(videos, dim=2)
    time_str = datetime.now().strftime("%H%M")
    save_video_path = f"{save_dir}/{ref_name}_{pose_name}_{time_str}.mp4"
    save_videos_grid(
        videos,
        save_video_path,
        n_rows=3,
        fps=args.fps,
    )
    print("infer results: {}".format(save_video_path))
    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
