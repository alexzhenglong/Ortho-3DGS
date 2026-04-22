import os
import sys
from argparse import ArgumentParser
from os import makedirs
from typing import NamedTuple

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn

# 自定义模块导入
from scene import Scene
from scene.colmap_loader import (
    qvec2rotmat, read_extrinsics_binary, read_extrinsics_text,
    read_intrinsics_binary, read_intrinsics_text
)
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import safe_state
# 注意：这里已经将 getPerspectiveToOrthographicMatrix 移除，改为在本脚本内定义
from utils.graphics_utils import (
    focal2fov, getProjectionMatrix, getWorld2View2
)
from gaussian_renderer import GaussianModel, render
from arguments import ModelParams, PipelineParams, get_combined_args


# ==========================================
# 数据结构定义
# ==========================================
class CameraInfo(NamedTuple):
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
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


# ==========================================
# 投影转换核心函数 (集成到本地，支持动态缩放)
# ==========================================
def getPerspectiveToOrthographicMatrix(P, ortho_scale=25.0):
    """
    将透视投影矩阵 P 转换为平行投影矩阵
    :param P: 透视投影矩阵
    :param ortho_scale: 场景正射缩放因子
    """
    znear = 0.01  # 近裁剪面
    zfar = 100.0  # 远裁剪面

    left = -ortho_scale / P[0, 0]
    right = ortho_scale / P[0, 0]
    bottom = -ortho_scale / P[1, 1]
    top = ortho_scale / P[1, 1]

    # 构建平行投影矩阵
    ortho_matrix = torch.zeros(4, 4)
    ortho_matrix[0, 0] = 2.0 / (right - left)
    ortho_matrix[1, 1] = 2.0 / (top - bottom)
    ortho_matrix[2, 2] = -2.0 / (zfar - znear)
    ortho_matrix[0, 3] = -(right + left) / (right - left)
    ortho_matrix[1, 3] = -(top + bottom) / (top - bottom)
    ortho_matrix[2, 3] = -(zfar + znear) / (zfar - znear)
    ortho_matrix[3, 3] = 1.0

    return ortho_matrix


# ==========================================
# 相机模型定义
# ==========================================
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):
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
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device: {e}")
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

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale), dtype=torch.float32,
                                                 device=self.data_device).transpose(0, 1)

        # 此处的正射投影如果在 Camera 初始化时不需要用到 ortho_scale，可以直接使用透视投影
        # 如果这个 Camera 类是给正常渲染用的，保持原样；如果专门给正射用，在外部会覆盖 projection_matrix
        proj_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy)
        ortho_proj_matrix = getPerspectiveToOrthographicMatrix(proj_matrix, ortho_scale=25.0)  # 默认值，外部正射相机生成时会覆盖

        self.projection_matrix = torch.tensor(ortho_proj_matrix, dtype=torch.float32,
                                              device=self.data_device).transpose(0, 1)

        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0)).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


# ==========================================
# 辅助函数
# ==========================================
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def generate_random_position(mean_center, plane_normal, radius=5.0):
    random_point = np.random.randn(3)
    random_point -= np.dot(random_point, plane_normal) * plane_normal
    random_point = normalize(random_point) * radius
    return mean_center + random_point


def get_scene_bounds(cam_infos):
    positions = [cam.T for cam in cam_infos]
    return np.min(positions, axis=0), np.max(positions, axis=0)


def compute_ortho_camera_pose(cam_infos):
    center_t = np.array([cam.T for cam in cam_infos])
    mean_center_t = np.mean(center_t, axis=0)

    # PCA 计算法向量
    centered_positions = center_t - mean_center_t
    cov_matrix = np.dot(centered_positions.T, centered_positions)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    z_axis = normalize(eigenvectors[:, np.argmin(eigenvalues)])

    reference_cam_t = cam_infos[8].T
    reference_vector = normalize(reference_cam_t - mean_center_t)

    x_axis = normalize(np.cross(reference_vector, z_axis))
    y_axis = np.cross(z_axis, x_axis)

    R = np.stack((x_axis, y_axis, z_axis), axis=-1)
    return mean_center_t, R, z_axis


def ReadCamera(path, images):
    sparse_path = os.path.join(path, "sparse/0")
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(sparse_path, "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(sparse_path, "cameras.bin"))
    except FileNotFoundError:
        cam_extrinsics = read_extrinsics_text(os.path.join(sparse_path, "images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(sparse_path, "cameras.txt"))

    reading_dir = "images" if images is None else images
    cam_infos = []

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write(f'\rReading camera {idx + 1}/{len(cam_extrinsics)}')
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            FovY = focal2fov(intr.params[0], intr.height)
            FovX = focal2fov(intr.params[0], intr.width)
        elif intr.model == "PINHOLE":
            FovY = focal2fov(intr.params[1], intr.height)
            FovX = focal2fov(intr.params[0], intr.width)
        else:
            raise ValueError("Only PINHOLE or SIMPLE_PINHOLE cameras are supported!")

        image_path = os.path.join(path, reading_dir, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(
            uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,
            image=image, image_path=image_path, image_name=image_name,
            width=intr.width, height=intr.height
        )
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return sorted(cam_infos, key=lambda x: x.image_name)


# ==========================================
# 核心渲染流
# ==========================================
def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                ortho_scale: float):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(dataset.model_path, "正射")
        makedirs(render_path, exist_ok=True)

        cam_infos = scene.getTrainCameras() + scene.getTestCameras()

        mean_center_t, R, z_axis = compute_ortho_camera_pose(cam_infos)

        new_camera = scene.getTrainCameras()[3]
        pix = 10
        new_camera.image_height *= pix
        new_camera.image_width *= pix

        elevation = 10.0
        new_camera.T = mean_center_t + elevation * z_axis
        new_camera.R = R

        new_camera.world_view_transform = torch.tensor(
            getWorld2View2(new_camera.R, new_camera.T, translate=np.array([0.0, 0.0, 0.0]), scale=2.0),
            dtype=torch.float32, device="cuda"
        ).transpose(0, 1)

        # 正射投影变换 (这里接入了外部传进来的 ortho_scale)
        proj_matrix = getProjectionMatrix(new_camera.znear, new_camera.zfar, new_camera.FoVx, new_camera.FoVy)
        ortho_proj_matrix = getPerspectiveToOrthographicMatrix(proj_matrix, ortho_scale=ortho_scale)

        new_camera.projection_matrix = torch.tensor(ortho_proj_matrix, dtype=torch.float32, device="cuda").transpose(0,
                                                                                                                     1)

        new_camera.full_proj_transform = new_camera.world_view_transform.unsqueeze(0).bmm(
            new_camera.projection_matrix.unsqueeze(0)).squeeze(0)
        new_camera.camera_center = new_camera.world_view_transform.inverse()[3, :3]

        rendering = render(new_camera, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '3dgs正射.png'))


def get_scale_from_path(model_path):
    """根据路径自动匹配缩放参数"""
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
            print(f"\n[Info] 自动匹配到场景: {scene_name}, 使用正射缩放比例 ortho_scale = {scale}")
            return scale

    print("\n[Info] 未匹配到预设场景，使用默认正射缩放比例 ortho_scale = 25.0")
    return 25.0


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # 将 default 设置为 None，以便判断用户是否手动输入了该参数
    parser.add_argument("--ortho_scale", default=None, type=float,
                        help="正射投影场景缩放因子 (例如: siheyuan设为55, wall设为20)")
    args = get_combined_args(parser)

    # 智能匹配逻辑：如果命令行没传 --ortho_scale，则根据路径自动推断
    if args.ortho_scale is None:
        args.ortho_scale = get_scale_from_path(args.model_path)
    else:
        print(f"\n[Info] 使用用户指定的正射缩放比例 ortho_scale = {args.ortho_scale}")

    print("Rendering " + args.model_path)
    safe_state(args.quiet)

    # 传递 args.ortho_scale
    render_sets(model.extract(args),
                args.iteration,
                pipeline.extract(args),
                args.skip_train,
                args.skip_test,
                args.ortho_scale)