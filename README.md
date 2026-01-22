# UniSurf: Universal Lifespan Cortical Surface Reconstruction
We propose UniSurf, a universal deep learning framework that jointly performs brain tissue segmentation and cortical surface reconstruction from T1-weighted MRI across the lifespan. The framework integrates three models: (1) a tissue segmentation network that produces brain tissue probability maps from intensity images and their corresponding edge maps; (2) an SDFs prediction network that estimates SDFs of the white matter and pial surfaces; and (3) a pial surface prediction network that infers the pial surface based on the predicted white matter surface and pial SDF.

<div style="text-align: center">
  <img src="figure/framework.png" width="100%" alt="UniSurf framework">
</div>

***

## Installation
To ensure a clean workspace and prevent dependency conflicts, we strongly recommend creating a new Conda environment before running the code.
### Create and Activate Environment
```bash
# Create a new conda environment named 'unisurf' with Python 3.9
conda create -n unisurf python=3.9 -y

# Activate the environment
conda activate unisurf

# Install the required libraries
pip install -r requirements.txt
```

## Inference
### 1. Download model weights
You can download our provided sample model weights for left hemisphere white matter surface reconstruction  through the following link: [UniSurf_Segmentation](https://drive.google.com/file/d/1xOLOaiXPEvqx75T4JudGT5Q-n9SqAOtx/view?usp=drive_link) for tissue segmentation, [UniSurf_LSDF](https://drive.google.com/file/d/1ec_d-w4uXCLfK4dQDQQNc4CGOKqq-EpU/view?usp=drive_link) for SDFs prediction and [UniSurf_LPial](https://drive.google.com/file/d/1uH-uFUp1kC172kKIcS0DnPrNGtcPac17/view?usp=drive_link) for pial surface reconstruction.

### 1.Data preparation
We provide a set of example samples in [Sample](./Sample) and a default data list in [test.xlsx](./test.xlsx), allowing you to run inference immediately using our provided model weights. The data structure is organized as follows:
