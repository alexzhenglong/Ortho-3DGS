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

The Ortho-3DGS pipeline consists of three stages:

### 1. Data Preparation (SfM)

Our data format is consistent with `nerfstudio`. Before training, raw UAV images must be processed to obtain camera poses and sparse point clouds. 

Please refer to the [nerfstudio custom dataset documentation](https://docs.nerf.studio/quickstart/custom_dataset.html) to download and run COLMAP for processing your raw images.

### 2. Model Training (Optimizer)

Train the scene using 3D Gaussian ellipsoids with depth-regulated constraints.

```bash
conda activate gaussian_splatting

# Train the model
python train.py -s <path_to_data> -m <path_to_model> --iterations 30000
