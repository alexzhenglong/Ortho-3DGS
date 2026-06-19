
# Ortho-3DGS: True Digital Orthophoto Generation

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=github)](https://alexzhenglong.github.io/Ortho-3DGS/)

Official implementation of the paper **"Ortho-3DGS: True Digital Orthophoto Generation From Unmanned Aerial Vehicle Imagery Using the Depth-Regulated 3D Gaussian Splatting"**, published in *IEEE J-STARS (2025)*.

---

## 🌟 Key Highlights

* **Non-Invasive Orthorectification**: A rendering strategy that generates True Digital Orthophoto Maps (DOMs) **without modifying the core CUDA rasterizer**, ensuring seamless compatibility with the broader 3DGS ecosystem.
* **Depth-Regulated Optimization**: Integrates depth supervision to prevent Gaussian over-expansion, strictly preserving edge sharpness and geometric fidelity in complex urban environments.
* **Large-Scale Processing**: Employs a **Progressive Chunking** mechanism to efficiently process massive UAV datasets while keeping VRAM usage under strict control.

---

## 🗂️ Datasets

All UAV benchmark datasets utilized in this research are publicly hosted on our **Hugging Face** repository. You can seamlessly download the pre-processed data to verify the training and rendering performance of the Ortho-3DGS pipeline.

🤗 **[Download the Ortho-3DGS Datasets on Hugging Face here](https://huggingface.co/datasets/alexzhenglong/Ortho-3DGS-Datasets)**

---

## 🛠 Hardware & Software Requirements

| Requirement | Specification |
| --- | --- |
| **Operating System** | Windows 10/11 or Ubuntu 22.04 |
| **GPU** | NVIDIA GPU (Compute Capability 7.0+) with **24GB VRAM** recommended |
| **CUDA SDK** | **11.8** *(Crucial: Versions 11.6 and 12.x have known compatibility issues)* |
| **C++ Compiler** | Visual Studio 2019 (Windows) or GCC (Linux) |

---

## 🚀 Workflow Pipeline

The Ortho-3DGS framework is executed through a standardized four-stage pipeline:

### 1. Data Preparation (SfM)

Our dataset format aligns with the `nerfstudio` standard. Prior to training, raw UAV imagery must be processed to extract camera poses and generate sparse point clouds.

If you are using your own captures, please refer to the [nerfstudio custom dataset documentation](https://docs.nerf.studio/quickstart/custom_dataset.html) to run COLMAP and prepare your raw images.

### 2. Depth Prior Generation (Depth Regularization)

To construct high-fidelity true orthophotos, our optimization strategy suppresses "floaters" and geometry degradation common in textureless UAV regions (e.g., roads, flat rooftops). By injecting depth maps as structural priors, Ortho-3DGS coerces the Gaussians to adhere to the true terrain geometry.

For custom UAV datasets, depth maps must be generated before training. We utilize **Depth Anything V2** for reliable depth estimation:

1. Clone the repository:
```bash
git clone [https://github.com/DepthAnything/Depth-Anything-V2.git](https://github.com/DepthAnything/Depth-Anything-V2.git)

```

2. Download the [Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) checkpoint and place it in the `Depth-Anything-V2/checkpoints/` directory.
3. Generate the depth maps:

```bash
python Depth-Anything-V2/run.py --encoder vitl --pred-only --grayscale --img-path <path_to_input_images> --outdir <output_path>

```

4. Generate the `depth_params.json` file to align the monocular depth scale with your COLMAP sparse reconstruction:

```bash
python utils/make_depth_scale.py --base_dir <path_to_colmap> --depths_dir <path_to_generated_depths>

```

### 3. Model Training (Optimizer)

Train the scene using 3D Gaussian ellipsoids anchored by depth-regulated constraints. Note the crucial `-d` parameter, which loads the depth priors.

```bash
conda activate gaussian_splatting

# Train the model with depth regularization
python train.py -s <path_to_data> -d <path_to_depth_maps> -m <path_to_model> --iterations 30000

```

**Key Training Parameters:**

* `--source_path / -s`: Path to the source directory containing the COLMAP dataset.
* `--model_path / -m`: Output path for the trained model checkpoints (default: `output/<random>`).
* `-d`: Path to the generated depth maps directory *(Crucial for Ortho-3DGS)*.
* `--eval`: Flag to enable a MipNeRF360-style training/test split for evaluation.
* `--resolution / -r`: Image downscaling factor before training (`1, 2, 4`, or `8`). Automatically rescales if image width exceeds 1.6K pixels.
* `--data_device`: Default is `cuda`. Set to `cpu` for large/high-resolution UAV datasets to mitigate VRAM bottlenecks.

### 4. DOM Generation (Orthorectification)

The final stage projects the trained 3D Gaussians into a georeferenced True Digital Orthophoto (DOM). We provide two distinct rendering modes:

#### 🔄 Rendering Methods Comparison

| Feature | Option A: Virtual Camera (Default) | Option B: Jacobian-based |
| --- | --- | --- |
| **Logic** | Geometry transformation | Direct CUDA kernel modification |
| **Installation** | **Plug-and-play** (Uses vanilla 3DGS rasterizer) | **Requires installation** of a custom module |
| **Precision** | Standard | Adaptive to complex topography |

#### 🛠 Option B Installation (Optional)

If you require the **Jacobian-based** direct rendering method, compile and install the custom orthographic rasterization module first:

```bash
cd submodules/ortho-rasterization
pip install .

```

#### 🚀 Execution

By default, we recommend utilizing the **Virtual Camera** method (Option A) for streamlined rendering:

```bash
# Render using the default virtual camera approach (No extra compilation needed)
python render_dom.py -m <path_to_model> -s <path_to_data> --mode virtual

```

*(Note: Scripts targeting the Option B direct rendering mechanism will be provided in a separate update.)*

---

## 📂 Repository Structure

* `train.py`: Core optimizer integrating depth regularization.
* `render_dom.py`: Execution script for True Orthophoto projection.
* `environment.yml`: Conda environment dependencies.

---

## 📝 Citation

If you find our code, datasets, or methodology useful for your research, please consider citing our paper:

```bibtex
@ARTICLE{yang2025ortho3dgs,
  author={Yang, Junxing and Cai, Zhenglong and Wang, Tianjiao and Ye, Tong and Gao, Haoran and Huang, He},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  title={Ortho-3DGS: True Digital Orthophoto Generation From Unmanned Aerial Vehicle Imagery Using the Depth-Regulated 3D Gaussian Splatting},
  year={2025},
  volume={18},
  pages={10972-10994},
  doi={10.1109/JSTARS.2025.3552105}
}

```

---

**Acknowledgements**: This work was supported by the *Graduate Innovation Project*. We extend our gratitude to the open-source community for the foundational work in 3D Gaussian Splatting and monocular depth estimation.

```

```
