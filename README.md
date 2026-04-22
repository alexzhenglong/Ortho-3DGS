

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
## 📅 Roadmap / Todo List

- [x] **Depth-Regulated Optimizer**: Official implementation of the geometry-enhanced training pipeline.
- [x] **Non-Invasive Orthorectification**: Rendering scripts for DOM generation.
- [ ] **Pre-trained Models (GCPs)**: Optimized checkpoints for various urban scenes (Coming Soon...).
- [ ] **UAV Benchmarks**: Our high-resolution drone datasets with ground truth (Coming Soon...).

---

## 🚀 Workflow Pipeline

> **Note**: Currently, you can use your own COLMAP datasets to verify the training and rendering performance.

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

To make this section look professional and academic, we can use a combination of **clean typography**, **emojis for visual cues**, and **structured comparison tables**. This helps users quickly grasp the technical difference between the two methods.

---

### 3. DOM Generation (Orthorectification)

The final stage transforms the trained 3D Gaussian representation into a high-resolution, georeferenced **True Digital Orthophoto (DOM)**. 

#### 🔄 Dual Rendering Methods
We provide two distinct approaches for orthorectification. You can choose the one that best fits your environment:

| Feature | **Option A: Virtual Camera (Default)** | **Option B: Jacobian-based** |
| :--- | :--- | :--- |
| **Logic** | **Non-invasive** geometry transformation | Direct modification of the **CUDA kernel** |
| **Modification** | No changes to the standard rasterizer | Requires custom Jacobian matrix logic |
| **Precision** | Commercial-grade (Standard) | High-precision (Steep/Complex terrain) |
| **Compatibility** | Plug-and-play with vanilla 3DGS | Optimized for research rigor |

---

#### 💡 Innovation Spotlight: Rasterizer-Independent Correction
Unlike traditional NeRF or 3DGS methods that require rewriting complex CUDA kernels, our **Option A** uses a **Virtual Orthographic Camera** strategy.
* **How it works:** By mathematically warping the projection matrices to a normalized horizontal datum, we leverage the standard high-speed splatting engine to produce distortion-free DOMs. 
* **The Result:** Seamless integration with existing 3DGS ecosystems without sacrificing georeferencing accuracy.

---

#### 🚀 How to Run
Use the following command to generate your DOM. You can toggle between methods using the `--mode` flag:

```bash
# Render using the default Virtual Camera method
python render_dom.py -m <path_to_model> -s <path_to_data> --mode virtual

# Render using the Jacobian-based method (Requires custom extension)
python render_dom.py -m <path_to_model> -s <path_to_data> --mode jacobian
```

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
