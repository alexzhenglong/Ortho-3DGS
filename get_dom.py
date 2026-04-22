import os
import sys
import logging
from argparse import ArgumentParser
from os import makedirs
from typing import NamedTuple, Tuple, List, Optional

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn

# Custom module imports
from scene import Scene
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import safe_state
from utils.graphics_utils import (
    focal2fov, getProjectionMatrix, getWorld2View2
)
from gaussian_renderer import GaussianModel, render
from arguments import ModelParams, PipelineParams, get_combined_args

# Configure standard logging for the open-source release
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ==========================================
# Data Structures
# ==========================================
class CameraInfo(NamedTuple):
    """Data structure for storing raw camera parameters and image metadata."""
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    image: Image.Image
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    """Data structure for storing parsed scene information."""
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


# ==========================================
# Core Projection Utilities
# ==========================================
def getPerspectiveToOrthographicMatrix(P: torch.Tensor, ortho_scale: float = 25.0) -> torch.Tensor:
    """
    Converts a perspective projection matrix to an orthographic projection matrix.

    Args:
        P (torch.Tensor): The original perspective projection matrix.
        ortho_scale (float): The scaling factor for the orthographic viewport.
                             Larger values cover a wider geographical area.

    Returns:
        torch.Tensor: The resulting 4x4 orthographic projection matrix.
    """
    znear = 0.01  # Near clipping plane
    zfar = 100.0  # Far clipping plane

    # Compute viewport boundaries based on the scale and original focal properties
    left = -ortho_scale / P[0, 0]
    right = ortho_scale / P[0, 0]
    bottom = -ortho_scale / P[1, 1]
    top = ortho_scale / P[1, 1]

    # Construct the orthographic projection matrix
    ortho_matrix = torch.zeros((4, 4), dtype=torch.float32)
    ortho_matrix[0, 0] = 2.0 / (right - left)
    ortho_matrix[1, 1] = 2.0 / (top - bottom)
    ortho_matrix[2, 2] = -2.0 / (zfar - znear)
    ortho_matrix[0, 3] = -(right + left) / (right - left)
    ortho_matrix[1, 3] = -(top + bottom) / (top - bottom)
    ortho_matrix[2, 3] = -(zfar + znear) / (zfar - znear)
    ortho_matrix[3, 3] = 1.0

    return ortho_matrix


# ==========================================
# Camera Model
# ==========================================
class Camera(nn.Module):
    """
    Differentiable Camera module for 3D Gaussian Splatting rendering.
    Handles both perspective and orthographic projection transformations.
    """

    def __init__(self, colmap_id: int, R: np.ndarray, T: np.ndarray, FoVx: float, FoVy: float,
                 image: torch.Tensor, gt_alpha_mask: Optional[torch.Tensor],
                 image_name: str, uid: int, trans: np.ndarray = np.array([0.0, 0.0, 0.0]),
                 scale: float = 1.0, data_device: str = "cuda"):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            logger.warning(f"Custom device {data_device} failed, fallback to default cuda device: {e}")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale), dtype=torch.float32, device=self.data_device
        ).transpose(0, 1)

        # Note: Default initialization uses orthographic projection for DOM generation.
        proj_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy)
        ortho_proj_matrix = getPerspectiveToOrthographicMatrix(proj_matrix, ortho_scale=25.0)

        self.projection_matrix = torch.tensor(
            ortho_proj_matrix, dtype=torch.float32, device=self.data_device
        ).transpose(0, 1)

        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0)
        ).squeeze(0)

        self.camera_center = self.world_view_transform.inverse()[3, :3]


# ==========================================
# Geometry & Pose Estimation Utilities
# ==========================================
def normalize(v: np.ndarray) -> np.ndarray:
    """Normalizes a vector to unit length."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def compute_ortho_camera_pose(cam_infos: List[CameraInfo]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the optimal orthographic projection plane and camera pose using
    Principal Component Analysis (PCA) on the trajectory of training cameras.

    Args:
        cam_infos (List[CameraInfo]): List of camera meta-information.

    Returns:
        Tuple containing:
            - mean_center_t (np.ndarray): The geometric centroid of the camera trajectory.
            - R (np.ndarray): The 3x3 rotation matrix representing the orthographic viewpoint.
            - z_axis (np.ndarray): The principal normal vector (Z-axis).
    """
    center_t = np.array([cam.T for cam in cam_infos])
    mean_center_t = np.mean(center_t, axis=0)

    # Perform PCA to find the normal vector of the camera trajectory plane
    centered_positions = center_t - mean_center_t
    cov_matrix = np.dot(centered_positions.T, centered_positions)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Z-axis corresponds to the eigenvector with the smallest eigenvalue (normal to the plane)
    z_axis = normalize(eigenvectors[:, np.argmin(eigenvalues)])

    # Select a reference camera to determine the orientation (X and Y axes)
    reference_cam_t = cam_infos[8].T
    reference_vector = normalize(reference_cam_t - mean_center_t)

    x_axis = normalize(np.cross(reference_vector, z_axis))
    y_axis = np.cross(z_axis, x_axis)

    # Construct the optimal rotation matrix
    R = np.stack((x_axis, y_axis, z_axis), axis=-1)
    return mean_center_t, R, z_axis


# ==========================================
# Rendering Pipeline
# ==========================================
def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool, ortho_scale: float,
                resolution_scale: int, elevation: float):
    """
    Main rendering pipeline for orthographic image generation (DOM).
    """
    args.depths = ""
    args.train_test_exp = False
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(dataset.model_path, "ortho_renders")
        makedirs(render_path, exist_ok=True)

        cam_infos = scene.getTrainCameras() + scene.getTestCameras()

        # Compute the global orthographic pose via PCA
        mean_center_t, R, z_axis = compute_ortho_camera_pose(cam_infos)

        # Clone a reference camera and configure orthographic parameters
        new_camera = scene.getTrainCameras()[3]

        # Scale resolution for high-quality DOM output
        new_camera.image_height *= resolution_scale
        new_camera.image_width *= resolution_scale

        # Adjust camera translation along the normal axis
        new_camera.T = mean_center_t + elevation * z_axis
        new_camera.R = R

        # Recompute world-to-view transform
        new_camera.world_view_transform = torch.tensor(
            getWorld2View2(new_camera.R, new_camera.T, translate=np.array([0.0, 0.0, 0.0]), scale=2.0),
            dtype=torch.float32, device="cuda"
        ).transpose(0, 1)

        # Apply target orthographic scale
        proj_matrix = getProjectionMatrix(new_camera.znear, new_camera.zfar, new_camera.FoVx, new_camera.FoVy)
        ortho_proj_matrix = getPerspectiveToOrthographicMatrix(proj_matrix, ortho_scale=ortho_scale)

        new_camera.projection_matrix = torch.tensor(
            ortho_proj_matrix, dtype=torch.float32, device="cuda"
        ).transpose(0, 1)

        # Finalize projection
        new_camera.full_proj_transform = new_camera.world_view_transform.unsqueeze(0).bmm(
            new_camera.projection_matrix.unsqueeze(0)
        ).squeeze(0)
        new_camera.camera_center = new_camera.world_view_transform.inverse()[3, :3]

        # Execute rendering and save output
        logger.info(f"Rendering orthophoto with scale={ortho_scale}, resolution_multiplier={resolution_scale}x")
        rendering = render(new_camera, gaussians, pipeline, background)["render"]
        output_path = os.path.join(render_path, '3DGS_Orthophoto.png')
        torchvision.utils.save_image(rendering, output_path)
        logger.info(f"Successfully saved orthophoto to: {output_path}")


def infer_ortho_scale_from_path(model_path: str) -> float:
    """
    Automatically infers the optimal orthographic scale based on the dataset name.

    Args:
        model_path (str): The path to the trained model/dataset.

    Returns:
        float: The matched orthographic scale, or a default value if not found.
    """
    scale_dict = {
        "siheyuan": 55.0,
        "wall": 20.0,
        "gcp1": 15.0,
        "lake": 17.0,
        "npu": 18.0,
        "huangqi": 17.0
    }

    for scene_name, scale in scale_dict.items():
        if scene_name.lower() in model_path.lower():
            logger.info(f"Auto-detected scene: '{scene_name}'. Applying ortho_scale = {scale}")
            return scale

    logger.info("No matching scene preset found. Utilizing default ortho_scale = 25.0")
    return 25.0


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="3D Gaussian Splatting Orthophoto (DOM) Generator")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int, help="Iteration to load")
    parser.add_argument("--skip_train", action="store_true", help="Skip training camera rendering")
    parser.add_argument("--skip_test", action="store_true", help="Skip testing camera rendering")
    parser.add_argument("--quiet", action="store_true", help="Disable console output")

    # Custom DOM Parameters
    parser.add_argument("--ortho_scale", default=None, type=float,
                        help="Scale factor for orthographic projection (e.g., 55.0 for siheyuan). "
                             "If None, auto-inference will be attempted.")
    parser.add_argument("--resolution_scale", default=10, type=int,
                        help="Multiplier for the output image resolution (default: 10x).")
    parser.add_argument("--elevation", default=10.0, type=float,
                        help="Virtual camera elevation along the normal axis (default: 10.0).")

    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.quiet:
        logger.setLevel(logging.WARNING)

    logger.info(f"Initializing DOM rendering pipeline for model: {args.model_path}")

    # Auto-infer scale if not explicitly provided
    if args.ortho_scale is None:
        args.ortho_scale = infer_ortho_scale_from_path(args.model_path)
    else:
        logger.info(f"Using user-specified ortho_scale = {args.ortho_scale}")

    # Launch renderer
    render_sets(
        dataset=model.extract(args),
        iteration=args.iteration,
        pipeline=pipeline.extract(args),
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        ortho_scale=args.ortho_scale,
        resolution_scale=args.resolution_scale,
        elevation=args.elevation
    )
