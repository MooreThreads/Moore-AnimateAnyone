import json
import random

import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


class HumanDanceDataset(Dataset):
    def __init__(
        self,
        img_size,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fahsion_meta.json"],
        sample_margin=30,
    ):
        super().__init__()

        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.sample_margin = sample_margin

        # -----
        # vid_meta format:
        # [{'video_path': , 'kps_path': , 'other':},
        #  {'video_path': , 'kps_path': , 'other':}]
        # -----
        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]

        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)

        assert len(video_reader) == len(
            kps_reader
        ), f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}"

        video_length = len(video_reader)

        margin = min(self.sample_margin, video_length)

        ref_img_idx = random.randint(0, video_length - 1)
        if ref_img_idx + margin < video_length:
            tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
        elif ref_img_idx - margin > 0:
            tgt_img_idx = random.randint(0, ref_img_idx - margin)
        else:
            tgt_img_idx = random.randint(0, video_length - 1)

        ref_img = video_reader[ref_img_idx]
        ref_img_pil = Image.fromarray(ref_img.asnumpy())
        tgt_img = video_reader[tgt_img_idx]
        tgt_img_pil = Image.fromarray(tgt_img.asnumpy())

        tgt_pose = kps_reader[tgt_img_idx]
        tgt_pose_pil = Image.fromarray(tgt_pose.asnumpy())

        state = torch.get_rng_state()
        tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
        tgt_pose_img = self.augmentation(tgt_pose_pil, self.cond_transform, state)
        ref_img_vae = self.augmentation(ref_img_pil, self.transform, state)
        clip_image = self.clip_image_processor(
            images=ref_img_pil, return_tensors="pt"
        ).pixel_values[0]

        sample = dict(
            video_dir=video_path,
            img=tgt_img,
            tgt_pose=tgt_pose_img,
            ref_img=ref_img_vae,
            clip_images=clip_image,
        )

        return sample

    def __len__(self):
        return len(self.vid_meta)
