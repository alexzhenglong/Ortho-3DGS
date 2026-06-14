
# Ortho-3DGS: True Digital Orthophoto Generation

[](https://ieeexplore.ieee.org/document/10930522)
[](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
[](https://opensource.org/licenses/MIT)

Official implementation of the paper **"Ortho-3DGS: True Digital Orthophoto Generation From Unmanned Aerial Vehicle Imagery Using the Depth-Regulated 3D Gaussian Splatting"** published in *IEEE JSTARS (2025)*.

---

## 🌟 Key Highlights

  * **Non-Invasive Orthorectification**: A rendering strategy that generates DOMs **without modifying the core CUDA rasterizer**, ensuring compatibility with the standard 3DGS ecosystem.
  * **Depth-Regulated Optimization**: Incorporates depth supervision to prevent Gaussian over-expansion, preserving edges and geometric fidelity in urban environments.
  * **Large-Scale Processing**: Features a **Progressive Chunking** mechanism to process UAV datasets while managing VRAM usage.

---

## 🛠 Hardware & Software Requirements

| Requirement | Specification |
| :--- | :--- |
| **Operating System** | Windows 10/11 or Ubuntu 22.04 |
| **GPU** | NVIDIA GPU (Compute Capability 7.0+) with **24GB VRAM** recommended |
| **CUDA SDK** | **11.8** (Crucial: 11.6 and 12.x are known to have compatibility issues) |
| **C++ Compiler** | Visual Studio 2019 (Windows) or GCC (Linux) |

---

## 📅 Roadmap / Todo List

- [x] **Depth-Regulated Optimizer**: Official implementation of the geometry-enhanced training pipeline.
- [x] **Non-Invasive Orthorectification**: Rendering scripts for DOM generation.
- [ ] **Pre-trained Models (GCPs)**: Checkpoints for various urban scenes (Coming Soon...).
- [ ] **UAV Benchmarks**: UAV datasets with ground truth (Coming Soon...).

---

## 🚀 Workflow Pipeline

> **Note**: Currently, you can use your own datasets to verify the training and rendering performance.

The Ortho-3DGS pipeline consists of the following stages:

### 1. Data Preparation (SfM)

Our data format is consistent with `nerfstudio`. Before training, raw UAV images must be processed to obtain camera poses and sparse point clouds. 

Please refer to the [nerfstudio custom dataset documentation](https://docs.nerf.studio/quickstart/custom_dataset.html) to download and run COLMAP for processing your raw images.

### 2. Depth Prior Generation (Depth Regularization)

To prevent Gaussian over-expansion and improve geometric accuracy in untextured areas (e.g., roads and building roofs), our pipeline utilizes depth maps as priors during optimization. 

For real-world UAV datasets, you must generate depth maps for your input images before training:

1. Clone the **Depth Anything V2** repository:
   ```bash
   git clone [https://github.com/DepthAnything/Depth-Anything-V2.git](https://github.com/DepthAnything/Depth-Anything-V2.git)

```

2. Download the [Depth-Anything-V2-Large](https://www.google.com/search?q=https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth%3Fdownload%3Dtrue) weights and place them under `Depth-Anything-V2/checkpoints/`.
3. Generate the depth maps:
```bash
python Depth-Anything-V2/run.py --encoder vitl --pred-only --grayscale --img-path <path to input images> --outdir <output path>

```


4. Generate the `depth_params.json` scale file:
```bash
python utils/make_depth_scale.py --base_dir <path to colmap> --depths_dir <path to generated depths>

```



### 3. Model Training (Optimizer)

Train the scene using 3D Gaussian ellipsoids with depth-regulated constraints.

```bash
conda activate gaussian_splatting

# Train the model with depth regularization
python train.py -s <path_to_data> -d <path_to_depth_maps> -m <path_to_model> --iterations 30000

```

**Key Training Parameters:**

* `--source_path / -s`: Path to the source directory containing the COLMAP dataset.
* `--model_path / -m`: Path where the trained model should be stored (default: `output/<random>`).
* `-d`: Path to the generated depth maps directory (Crucial for Ortho-3DGS depth regularization).
* `--eval`: Add this flag to use a MipNeRF360-style training/test split for evaluation.
* `--resolution / -r`: Specifies resolution of the loaded images before training (`1, 2, 4` or `8` for original, 1/2, 1/4 or 1/8). Automatically rescales if width > 1.6K pixels.
* `--data_device`: Default is `cuda`. For large/high-resolution UAV datasets, set to `cpu` to reduce VRAM consumption.
* `--densify_from_iter` / `--densify_until_iter`: Iterations where densification starts (default: `500`) and stops (default: `15_000`).

---

### 4. DOM Generation (Orthorectification)

The final stage transforms the trained 3D Gaussian representation into a georeferenced True Digital Orthophoto (DOM).

#### 🔄 Dual Rendering Methods

| Feature | **Option A: Virtual Camera (Default)** | **Option B: Jacobian-based** |
| --- | --- | --- |
| **Logic** | Geometry transformation | Direct modification of the **CUDA kernel** |
| **Installation** | **Plug-and-play** (No extra installation, uses vanilla 3DGS rasterizer) | **Requires installation** of our custom `ortho_rasterization` module |
| **Precision** | Standard (Commercial-grade template) | High-precision (Adaptive to steep/complex terrain) |
| **Script** | `render_dom.py` (Direct vanilla rendering) | `render_dom.py` (With modified CUDA backend) |

##### 🛠 Option B Installation Steps

如果你想使用高精度的 **Option B (Jacobian-based)** 方法，需要先编译安装我们定制的正射光栅化模块：

```bash
# 进入定制的正射光栅化文件夹
cd submodules/ortho-rasterization

# 编译并安装该模块
pip install .

```

#### 🚀 How to Run

使用以下命令来生成你的 DOM 结果，可以通过 `--mode` 参数在两种渲染模式之间进行切换：

```bash
# 模式一：使用默认的虚拟相机方法直接渲染 (无需额外安装任何光栅化器)
python render_dom.py -m <path_to_model> -s <path_to_data> --mode virtual

# 模式二：使用基于雅可比矩阵的 CUDA 核心直接渲染 (需要提前安装 ortho_rasterization 模块)
python render_dom.py -m <path_to_model> -s <path_to_data> --mode jacobian

```

---

## 📂 Repository Structure

* `train.py`: The main optimizer with depth regulation.
* `render_dom.py`: Script for True Orthophoto rendering (Supports both virtual camera and Jacobian-based modes).
* `environment.yml`: Conda environment configuration.

---

## 📝 Citation

If you find this work helpful for your research, please consider citing our paper:

```bibtex
@ARTICLE{yang2025ortho3dgs,
  author={Yang, Junxing and Cai, Zhenglong ...王天娇..., Ye, Tong and Gao, Haoran and Huang, He},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  title={Ortho-3DGS: True Digital Orthophoto Generation From Unmanned Aerial Vehicle Imagery Using the Depth-Regulated 3D Gaussian Splatting},
  year={2025},
  volume={18},
  pages={10972-10994},
  doi={10.1109/JSTARS.2025.3552105}
}

```

---

**Acknowledgements**: This work was supported by the *Graduate Innovation Project*. We thank the open-source community for the foundations provided by 3DGS and Depth estimation research.

```

```
