

<div align="center">
<h1><img src="img/logo.png" alt="Logo" style="height:50px;vertical-align:middle"><i>MineDreamer</i> : Learning to Follow Instructions via </center> <br> <center>Chain-of-Imagination for Simulated-World Control </h1>

🥰 **If you are interested in our work, feel free to star ⭐ or watch 👓 our repo for the latest updates🤗!!**


[![arXiv](https://img.shields.io/badge/arXiv%20papr-2403.12037-b31b1b.svg)](https://arxiv.org/abs/2403.12037)&nbsp;
[![project page](https://img.shields.io/badge/More%20Demo%20video%21-project%20page-lightblue)](https://sites.google.com/view/minedreamer/main)

[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Zhoues/MineDreamer_7B-yellow)](https://huggingface.co/Zhoues/MineDreamer-7B)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Zhoues/Goal_Drift_Dataset-yellow)](https://huggingface.co/datasets/Zhoues/Goal-Drift-Dataset)&nbsp;

</div>


## 🔥 Updates
[2024-04-03] 🔥🔥🔥 *MineDreamer* code is released. Let's enjoy the Imagination ability of the embodied agent!

[2024-03-19] *MineDreamer* is released on [arxiv](https://arxiv.org/abs/2403.12037).

[2024-03-15] The Project page is set up at [here](https://sites.google.com/view/minedreamer/main).



## 😋 Try *MineDreamer*
The code and checkpoints are released and the open-source contents include the following:

- ✅ *MineDreamer* agent and Baseline Code (i.e., VPT, STEVE-1, Multi-Modal Memory)
- ✅ *MineDreamer* Goal Drift Dataset and MineDreamer weights, including MineDreamer-7B of Imaginator and Prompt Generator.
- ✅ *MineDreamer* Training Scripts, including The Imaginator training stages 2 and 3. 

- Note: For Imaginator training stage 1, we only provide pre-trained Q-Former weights. For Prompt Generator, we only provide the weights and if you want to train your own Prompt Generator, please refer to [STEVE-1](https://github.com/Shalev-Lifshitz/STEVE-1/tree/main?tab=readme-ov-file#training) to collect data and train it.
### Directory Structure:
```
.
├── README.md
├── minedreamer
│   ├── All agent code, including baseline and MineDreamer.
├── imaginator
│   ├── All imaginator code including training and inference.
│ 
├── play: Scripts for running the agent for all evaluations.
│   ├── programmatic: run the inference code of Programmatic Evaluation
│   │
│   ├── chaining: run the inference code of Command-Switching Evaluation
│ 
├── scripts
│   ├── Scripts for training and inference of Imaginator.
│  
├── download_baseline_weights.sh: download baseline weights.
│  
├── download_minedreamer_weights.sh: download minedreamer and other pre-trained weights for Imaginator training.
```

### Model Zoo and Dataset

We provide MineDreamer models for you to play with, including all three training stages checkpoints, and datasets. You can be downloaded from the following links:

| model                     | training stage | size   | HF weights🤗                                                  | HF dataset 🤗                                                 |
| ------------------------- | -------------- | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Pre-trained Q-Former      | 1              | 261MB  | [Pretrained-QFormer](https://huggingface.co/Zhoues/Pretrained-QFormer-7B) |                                                              |
| InstructPix2Pix U-Net     | 2              | 3.44GB | [InstructPix2Pix-Unet](https://huggingface.co/Zhoues/MineDreamer-InstructPix2Pix-Unet) | [Goal-Drift-Dataset](https://huggingface.co/datasets/Zhoues/Goal-Drift-Dataset) |
| MineDreamer-Imaginator-7B | 3              | 17.7GB | [MineDreamer-7B](https://huggingface.co/Zhoues/MineDreamer-7B/) | [Goal-Drift-Dataset](https://huggingface.co/datasets/Zhoues/Goal-Drift-Dataset) |








### Step 1: Install MineRL Env and Run Baseline
**It's worth noting that if you wish only to train or test the Imaginator, you can skip Step 1.**

1. We provide two methods for installing the MineRL environment. Detailed instructions can be found in [this repo](https://github.com/Zhoues/minerl-apptainer). Please ensure you complete the final test, otherwise the Agent will not function correctly.
2. Download the weights (Baseline weights + Prompt Generator weights): `sh download_baseline_weights.sh`
3. Run Baseline. If you use the Normal Installation Procedure, ignore the part of [], and if you use cluster like slurm, replace `sudo` with `srun -p <your virtual partition> --gres=gpu:1`.
    ```bash
    # If the server is headful
    [sudo apptainer exec -w --nv --bind /path/to/MineDreamer:/path/to/MineDreamer vgl-env] sh play/programmatic/steve1_play_w_text_prompt.sh mine_block_wood

    # If the server is headless
    [sudo apptainer exec -w --nv --bind /path/to/MineDreamer:/path/to/MineDreamer vgl-env] sh play/programmatic/XVFB_steve1_play_w_text_prompt.sh mine_block_wood

    # GPU rendering via apptainer container
    sudo apptainer exec -w --nv --bind /path/to/MineDreamer:/path/to/MineDreamer vgl-env bash setupvgl.sh play/programmatic/XVFB_steve1_play_w_text_prompt.sh mine_block_wood
    ```

### Step 2: Install Imaginator Env and Run *MineDreamer* Agent

This codebase has strict environmental requirements; we recommend you follow the tutorial below step by step.

1. We recommend running on Linux using a conda environment, with python 3.9: `conda create -n imaginator python=3.9`.
2. Install pytorch for cuda-118: 
    ```
    pip install --pre torch==2.2.0.dev20231010+cu118 torchvision==0.17.0.dev20231010+cu118 torchaudio==2.2.0.dev20231010+cu118 --index-url https://download.pytorch.org/whl/nightly/cu118
    ```
    - Note: The version of the torch may change over time. If you encounter an error that means the following version does not exist, please change the right version by using the error information.
3. Install additional packages: `pip install -r requirements.txt`
4. Install DeepSpeed: `DS_BUILD_AIO=1 DS_BUILD_FUSED_LAMB=1 pip install deepspeed`
    - Note: This step often fails due to the requirement of specific versions of CUDA and GCC. It is expected that `cuda118` and `gcc-7.5.0` are used. To ensure error-free script execution in the future, the commands to activate these versions should be added to the `~/.bashrc` file. Below is a reference for the content to be included in the `~/.bashrc`:
        ```
        ...
        export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc/gcc-7.5.0/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
        export PATH=/mnt/petrelfs/share/gcc/gcc-7.5.0/bin:$PATH

        export PATH=/mnt/petrelfs/share/cuda-11.8/bin:$PATH
        export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.8/lib64:$LD_LIBRARY_PAT
        ...
        ```
        Upon installation, you can enter `ds_report`. If the output appears as shown below, it indicates the installation is correct:
        ```
        fused_adam ............. [YES] ...... [OKAY]
        ```
5. Download the weights (Imaginator weights + pre-trained weights for training): `sh download_minedreamer_weights.sh`
6. Try inferencing the Imaginator and  (InstructPix2Pix). You can find generated images in `inference_valid_*` folder. 
    ```bash
    # InstructPix2Pix 
    bash scripts/inference_IP2P.sh

    # Imaginator
    bash scripts/inference_MineDreamer.sh
    ```
7. To run the *MineDreamer* agent, first you need to launch the backend service of Imaginator.
    ```
    # InstructPix2Pix 
    bash scripts/minedreamer_backend_IP2P.sh

    # Imaginator
    bash scripts/minedreamer_backend_MLLMSD.sh
    ```
    At this point, you'll receive a backend IP address similar to `Running on http://10.140.1.104:25547 (Press CTRL+C to quit)`. Then, you should insert this IP address into the `dreamer_url` field within the `minedreamer/play/config/programmatic/mine_block_wood.yaml` file, similar to:
    ```
    dreamer_url: http://10.140.1.104:25547/
    ```
8. Run the *MineDreamer* Agent. The process is consistent with running the baseline in Step 1, but this time you should execute the `*_dreamer_play_w_text_prompt.sh` script.


### Step 3: Train your own Imaginator
1. First, download the [Goal Drift Dataset](https://huggingface.co/datasets/Zhoues/Goal-Drift-Dataset) and place it in the `data/mllm_diffusion_dataset` directory and unzip it.
2. To train the Unet parameters of InstructPix2Pix, execute: `bash scripts/train_InstructPix2Pix_minecraft.sh`. This checkpoint can also be used as baseline.
3. Remove the original LoRA parameters from Huggingface's LLaVA with: `bash scripts/pre_llava.sh`.
4. Train Imaginator-7B by running: `bash scripts/train_MineDreamer.sh`.

## 🕶️Overview

### The Overview of Chain-of-Imagination within *MineDreamer*
<div align="center"> 
    <img src="img/pipeline_2.png" alt="Logo" style="height:460px;vertical-align:middle">
</div>


### The Overview Framework of Imaginator within *MineDreamer*
<div align="center"> 
    <img src="img/imaginator_2.jpg" alt="Logo" style="height:400px;vertical-align:middle">
</div>




## 📹 Demo video and Imagination Visual Results
More demo videos and Imagination visual results are on our [project webpage](https://sites.google.com/view/minedreamer).

### Imagination Visual Results on Evaluation Set Compared to the Baseline
<div align="center"> 
    <img src="img/evaluation.jpg" alt="Logo" style="height:500px;vertical-align:middle">
</div>

### Imagination Visual Results During Agent Solving Open-ended Tasks
<div align="center"> 
    <img src="img/inference_1.jpg" alt="Logo" style="height:500px;vertical-align:middle">
    <img src="img/inference_2.jpg" alt="Logo" style="height:500px;vertical-align:middle">
</div>


## Building a more generalist embodied agent 
A generalist embodied agent should have a high-level planner capable of perception and planning in an open world, as well as a low-level controller able to act in complex environments. The *MineDreamer* agent can steadily follow short-horizon text instructions, making it suitable as a low-level controller for generating control signals. For high-level planner, including perception and task planning in an open world, one can look to the methods presented in **[CVPR2024's MP5](https://github.com/IranQin/MP5), whose code is also released!** It is adept at planning for tasks that require long-horizon sequencing and extensive environmental awareness. Therefore, combining MP5 with *MineDreamer* presents a promising approach to developing more generalist embodied agents.



## 📑 Citation

If you find *MineDreamer* and MP5 useful for your research and applications, please cite using this BibTeX:
```
@article{zhou2024minedreamer,
  title={MineDreamer: Learning to Follow Instructions via Chain-of-Imagination for Simulated-World Control},
  author={Zhou, Enshen and Qin, Yiran and Yin, Zhenfei and Huang, Yuzhou and Zhang, Ruimao and Sheng, Lu and Qiao, Yu and Shao, Jing},
  journal={arXiv preprint arXiv:2403.12037},
  year={2024}
}

@article{qin2023mp5,
  title={Mp5: A multi-modal open-ended embodied system in minecraft via active perception},
  author={Qin, Yiran and Zhou, Enshen and Liu, Qichang and Yin, Zhenfei and Sheng, Lu and Zhang, Ruimao and Qiao, Yu and Shao, Jing},
  journal={arXiv preprint arXiv:2312.07472},
  year={2023}
}
```
