
# Ortho-3DGS: True Digital Orthophoto Generation

[](https://ieeexplore.ieee.org/document/10930522)
[](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
[](https://opensource.org/licenses/MIT)

Official implementation of the paper **"Ortho-3DGS: True Digital Orthophoto Generation From Unmanned Aerial Vehicle Imagery Using the Depth-Regulated 3D Gaussian Splatting"** published in *IEEE JSTARS (2025)*.

-----

## 🌟 Key Highlights

  * **Non-Invasive Orthorectification**: A specialized rendering strategy that generates DOMs **without modifying the core CUDA rasterizer**, ensuring maximum compatibility with the standard 3DGS ecosystem.
  * **Depth-Regulated Optimization**: Incorporates depth supervision to prevent Gaussian "over-expansion," ensuring sharp edges and high geometric fidelity in urban environments.
  * **Large-Scale Efficiency**: Features a **Progressive Chunking** mechanism to process extensive UAV datasets while maintaining a low VRAM footprint.

-----

## 🛠 Hardware & Software Requirements

| Requirement | Specification |
| :--- | :--- |
| **Operating System** | Windows 10/11 or Ubuntu 22.04 |
| **GPU** | NVIDIA GPU (Compute Capability 7.0+) with **24GB VRAM** recommended |
| **CUDA SDK** | **11.8** (Crucial: 11.6 and 12.x are known to have compatibility issues) |
| **C++ Compiler** | Visual Studio 2019 (Windows) or GCC (Linux) |

-----

## 🚀 Workflow Pipeline

The Ortho-3DGS pipeline consists of three streamlined stages:

### 1\. Data Preparation (SfM)

Before training, raw UAV images must be processed to obtain camera poses and sparse point clouds via COLMAP.

```bash
# 1. Place raw images in <location>/input
# 2. Run the conversion script
python convert.py -s <location> --resize
```

  * **Result**: Generates `images/` (undistorted) and `sparse/` (SfM data) required for the next step.

### 2\. Model Training (Optimizer)

Train the scene using 3D Gaussian ellipsoids with depth-regulated constraints.

```bash
conda activate gaussian_splatting

# Train the model
python train.py -s <path_to_data> -m <path_to_model> --iterations 30000
```

  * **Pro Tip**: For large-scale scenes, add `--data_device cpu` to offload image data and save GPU memory.

### 3\. DOM Generation (Orthorectification)

The final stage transforms the 3D representation into a 2D True Digital Orthophoto (DOM).

```bash
python render_dom.py -m <path_to_trained_model> -s <path_to_data>
```

> **💡 Innovation: Rasterizer-Independent Correction**
> Unlike other methods that require complex modifications to the `diff-gaussian-rasterization` CUDA kernels, our approach uses a **Virtual Orthographic Camera**. By mathematically transforming the projection matrices to a normalized horizontal datum, we produce georeferenced, distortion-free DOMs using the standard high-speed splatting engine.

-----

## 📂 Repository Structure

  * `train.py`: The main optimizer with depth regulation.
  * `render_dom.py`: Script for True Orthophoto rendering.
  * `convert.py`: COLMAP interface for data preprocessing.
  * `environment.yml`: Conda environment configuration.

-----

## 📝 Citation

If you find this work helpful for your research, please consider citing our paper:

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

-----

**Acknowledgements**: This work was supported by the *Graduate Innovation Project*. We thank the open-source community for the foundations provided by 3DGS and Depth estimation research.
