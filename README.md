# UniSurf: Universal Lifespan Cortical Surface Reconstruction
We introduce UniSurf, a universal end-to-end deep learning framework designed for cortical surface reconstruction across the entire human lifespan (0-100 years). UniSurf jointly optimizes tissue segmentation and cortical surface reconstruction using a differentiable iso-surface extraction algorithm.

***
## Model overview
<div style="text-align: center">
  <img src="figures/overview.png" width="100%" alt="UniSurf Framework">
</div>

## Results
<div style="text-align: center">
  <img src="figures/wm_results.png" width="100%" alt="Results of WM surface">
</div>

<div style="text-align: center">
  <img src="figures/pial_results.png" width="100%" alt="Results of pial surface">
</div>

***

***
# Get started
## Step 1: Data Preparation
Organize your project directory as follows to reproduce UniSurf on your own data
```bash
data/
├── subject001/
│   ├── brain.nii.gz            # brain image
│   ├── edge.nii.gz             # sobel edge
│   ├── lh.nii.hz               # left tissue map GT
│   ├── rh.nii.gz               # right tissue map GT
│   ├── lh.white                # left wm surface GT
│   ├── rh.white                # right wm surface GT
│   ├── lh.pial                 # left pial surface GT
│   └── rh.pial                 # right pial surface GT
├── subject002 
├── subject003
└── ……
```

## Step 2: Data Prepocessing
To generate SDF for model pre-training, you can run the following first:
```bash
python seg2sdf.py --data_path='/your/path/to/data'
```

## Step 3: Model Training
For training of UniSurf model, please run:
```bash
python train.py --data_path='/your/path/to/data' --excel_path='/your/path/to/data/list' --surf_hemi='left' --n_epochs=100 --output_dir='/your/path/to/save/models'
```
and you can uncomment the code to decide which model to train.

***

# Citation
If you find this work useful in your research, please cite:
> **Zifeng Lian<sup>†</sup>, Jiameng Liu<sup>†</sup>, Jiawei Huang, Shijie Huang, Xiaoye Li, Han Zhang, Zhiming Cui, Feng Shi, Dinggang Shen<sup>*</sup>. UniSurf: Universal Lifespan Cortical Surface Reconstruction. (Under Review)**

# [<font color=#F8B48F size=3>License</font> ](./LICENSE)
```shell
Copyright IDEA Lab, School of Biomedical Engineering, ShanghaiTech University, Shanghai, China.

Licensed under the the GPL (General Public License);
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Repo for UniSurf: Universal Lifespan Cortical Surface Reconstruction
Contact: lianzf2024@shanghaitech.edu.cn
```
