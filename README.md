# 🤗 Introduction

**update** 🏋️🏋️🏋️ We release our training codes!! Now you can train your own AnimateAnyone models. See [here](#train) for more details. Have fun!

**update**：🔥🔥🔥 We launch a HuggingFace Spaces demo of Moore-AnimateAnyone at [here](https://huggingface.co/spaces/xunsong/Moore-AnimateAnyone)!!

This repository reproduces [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone). To align the results demonstrated by the original paper, we adopt various approaches and tricks, which may differ somewhat from the paper and another [implementation](https://github.com/guoqincode/Open-AnimateAnyone). 

It's worth noting that this is a very preliminary version, aiming for approximating the performance (roughly 80% under our test) showed in [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone). 

We will continue to develop it, and also welcome feedbacks and ideas from the community. The enhanced version will also be launched on our [MoBi MaLiang](https://maliang.mthreads.com/) AIGC platform, running on our own full-featured GPU S4000 cloud computing platform.

# 📝 Release Plans

- [x] Inference codes and pretrained weights
- [x] Training scripts

# 🎞️ Examples 

Here are some results we generated, with the resolution of 512x768.

https://github.com/MooreThreads/Moore-AnimateAnyone/assets/138439222/f0454f30-6726-4ad4-80a7-5b7a15619057

https://github.com/MooreThreads/Moore-AnimateAnyone/assets/138439222/337ff231-68a3-4760-a9f9-5113654acf48

<table class="center">
    
<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/MooreThreads/Moore-AnimateAnyone/assets/138439222/9c4d852e-0a99-4607-8d63-569a1f67a8d2" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/MooreThreads/Moore-AnimateAnyone/assets/138439222/722c6535-2901-4e23-9de9-501b22306ebd" muted="false"></video>
    </td>
</tr>

<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/MooreThreads/Moore-AnimateAnyone/assets/138439222/17b907cc-c97e-43cd-af18-b646393c8e8a" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/MooreThreads/Moore-AnimateAnyone/assets/138439222/86f2f6d2-df60-4333-b19b-4c5abcd5999d" muted="false"></video>
    </td>
</tr>
</table>

**Limitation**: We observe following shortcomings in current version:
1. The background may occur some artifacts, when the reference image has a clean background
2. Suboptimal results may arise when there is a scale mismatch between the reference image and keypoints. We have yet to implement preprocessing techniques as mentioned in the [paper](https://arxiv.org/pdf/2311.17117.pdf).
3. Some flickering and jittering may occur when the motion sequence is subtle or the scene is static.

These issues will be addressed and improved in the near future. We appreciate your anticipation!

# ⚒️ Installation

## Build Environtment

We Recommend a python version `>=3.10` and cuda version `=11.7`. Then build environment as follows:

```shell
# [Optional] Create a virtual env
python -m venv .venv
source .venv/bin/activate
# Install with pip:
pip install -r requirements.txt
```

## Download weights

**Automatically downloading**: You can run the following command to download weights automatically:

```shell
python tools/download_weights.py
```

Weights will be placed under the `./pretrained_weights` direcotry. The whole downloading process may take a long time.

**Manually downloading**: You can also download weights manually, which has some steps:

1. Download our trained [weights](https://huggingface.co/patrolli/AnimateAnyone/tree/main), which include four parts: `denoising_unet.pth`, `reference_unet.pth`, `pose_guider.pth` and `motion_module.pth`.

2. Download pretrained weight of based models and other components: 
    - [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
    - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
    - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)

3. Download dwpose weights (`dw-ll_ucoco_384.onnx`, `yolox_l.onnx`) following [this](https://github.com/IDEA-Research/DWPose?tab=readme-ov-file#-dwpose-for-controlnet).

Finally, these weights should be orgnized as follows:

```text
./pretrained_weights/
|-- DWPose
|   |-- dw-ll_ucoco_384.onnx
|   `-- yolox_l.onnx
|-- image_encoder
|   |-- config.json
|   `-- pytorch_model.bin
|-- denoising_unet.pth
|-- motion_module.pth
|-- pose_guider.pth
|-- reference_unet.pth
|-- sd-vae-ft-mse
|   |-- config.json
|   |-- diffusion_pytorch_model.bin
|   `-- diffusion_pytorch_model.safetensors
`-- stable-diffusion-v1-5
    |-- feature_extractor
    |   `-- preprocessor_config.json
    |-- model_index.json
    |-- unet
    |   |-- config.json
    |   `-- diffusion_pytorch_model.bin
    `-- v1-inference.yaml
```

Note: If you have installed some of the pretrained models, such as `StableDiffusion V1.5`, you can specify their paths in the config file (e.g. `./config/prompts/animation.yaml`).

# 🚀 Training and Inference 

## Inference

Here is the cli command for running inference scripts:

```shell
python -m scripts.pose2vid --config ./configs/prompts/animation.yaml -W 512 -H 784 -L 64
```

You can refer the format of `animation.yaml` to add your own reference images or pose videos. To convert the raw video into a pose video (keypoint sequence), you can run with the following command:

```shell
python tools/vid2pose.py --video_path /path/to/your/video.mp4
```

## <span id="train"> Training </span>

Note: package dependencies have been updated, you may upgrade your environment via `pip install -r requirements.txt` before training.

### Data Preparation

Extract keypoints from raw videos: 

```shell
python tools/extract_dwpose_from_vid.py --video_root /path/to/your/video_dir
```

Extract the meta info of dataset:

```shell
python tools/extract_meta_info.py --root_path /path/to/your/video_dir --dataset_name anyone 
```

Update lines in the training config file: 

```yaml
data:
  meta_paths:
    - "./data/anyone_meta.json"
```

### Stage1

Put [openpose controlnet weights](https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/tree/main) under `./pretrained_weights`, which is used to initialize the pose_guider.

Put [sd-image-variation](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main) under `./pretrained_weights`, which is used to initialize unet weights.

Run command:

```shell
accelerate launch train_stage_1.py --config configs/train/stage1.yaml
```

### Stage2

Put the pretrained motion module weights `mm_sd_v15_v2.ckpt` ([download link](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt)) under `./pretrained_weights`. 

Specify the stage1 training weights in the config file `stage2.yaml`, for example:

```yaml
stage1_ckpt_dir: './exp_output/stage1'
stage1_ckpt_step: 30000 
```

Run command:

```shell
accelerate launch train_stage_2.py --config configs/train/stage2.yaml
```

# 🎨 Gradio Demo

**HuggingFace Demo**: We launch a quick preview demo of Moore-AnimateAnyone at [HuggingFace Spaces](https://huggingface.co/spaces/xunsong/Moore-AnimateAnyone)!!
We appreciate the assistance provided by the HuggingFace team in setting up this demo.

To reduce waiting time, we limit the size (width, height, and length) and inference steps when generating videos. 

If you have your own GPU resource (>= 16GB vram), you can run a local gradio app via following commands:

`python app.py`

# Community Contributions

- Installation for Windows users: [Moore-AnimateAnyone-for-windows](https://github.com/sdbds/Moore-AnimateAnyone-for-windows)

# 🖌️ Try on Mobi MaLiang

We will launched this model on our [MoBi MaLiang](https://maliang.mthreads.com/) AIGC platform, running on our own full-featured GPU S4000 cloud computing platform. Mobi MaLiang has now integrated various AIGC applications and functionalities (e.g. text-to-image, controllable generation...). You can experience it by [clicking this link](https://maliang.mthreads.com/) or scanning the QR code bellow via WeChat!

<p align="left">
  <img src="assets/mini_program_maliang.png" width="100
  "/>
</p> 

# ⚖️ Disclaimer

This project is intended for academic research, and we explicitly disclaim any responsibility for user-generated content. Users are solely liable for their actions while using the generative model. The project contributors have no legal affiliation with, nor accountability for, users' behaviors. It is imperative to use the generative model responsibly, adhering to both ethical and legal standards.

# 🙏🏻 Acknowledgements

We first thank the authors of [AnimateAnyone](). Additionally, we would like to thank the contributors to the [majic-animate](https://github.com/magic-research/magic-animate), [animatediff](https://github.com/guoyww/AnimateDiff) and [Open-AnimateAnyone](https://github.com/guoqincode/Open-AnimateAnyone) repositories, for their open research and exploration. Furthermore, our repo incorporates some codes from [dwpose](https://github.com/IDEA-Research/DWPose) and [animatediff-cli-prompt-travel](https://github.com/s9roll7/animatediff-cli-prompt-travel/), and we extend our thanks to them as well.
