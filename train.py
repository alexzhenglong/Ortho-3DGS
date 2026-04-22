#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint

# 包含一些用于计算损失的工具函数，例如 L1 损失和 SSIM（结构相似性指数）
from utils.loss_utils import l1_loss, ssim
# 导入 render 和 network_gui 函数、render 函数用于渲染场景，network_gui 可能是与网络图形用户界面（GUI）相关的功能
from gaussian_renderer import render, network_gui
import sys

# 导入了 Scene 和 GaussianModel 类，可能包含有关场景和高斯模型的定义和操作
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr

# 导入了 ModelParams、PipelineParams 和 OptimizationParams 类，这些类可能用于定义模型、流水线和优化的参数
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # 创建高斯模型的实例
    """这是一个空的高斯模型实例"""
    gaussians = GaussianModel(dataset.sh_degree)

    # 创建 Scene 的实例 (scene)，该实例可能包含了有关场景、摄像机、图像等的信息
    scene = Scene(dataset, gaussians)

    gaussians.training_setup(opt)
    # 检查是否有模型加载
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    """opt.iterations的默认值10_000"""

    for iteration in range(first_iter, opt.iterations + 1):

        """尝试链接网络可视化训练
        这里不使用"""
        # 检查网络连接是否为空，尝试连接
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        #
        # # 当网络连接存在时，进入主循环
        # while network_gui.conn != None:
        #     try:
        #         # 初始化网络图像字节为None
        #         net_image_bytes = None
        #
        #         # 从网络接收一系列参数
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #
        #         # 如果接收到了custom_cam，执行渲染操作
        #         if custom_cam != None:
        #             # 使用参数渲染图像
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #
        #             # 将渲染后的图像转换为字节数组
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
        #                                                                                                        0).contiguous().cpu().numpy())
        #
        #         # 将渲染后的图像和数据发送回网络
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #
        #         # 如果需要进行训练，并且满足训练条件，跳出循环
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #
        #     except Exception as e:
        #         # 发生异常时，将网络连接设置为None，以便重新尝试建立连接
        #         network_gui.conn = None
        """开始训练"""
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # 渲染图像
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        """渲染生成的图像、视图空间的点张量、可见性过滤器和半径等信息"""
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 求Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        """在当前代码位置创建一个 GPU 事件，标记为 iter_end。这就是一个时间戳，用于记录当前的时间点"""
        iter_end.record()


        """下面的代码不需要梯度计算"""
        with torch.no_grad():
            # 进度条
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # 每10个迭代更新一次进度条的显示
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 记录和保存-------------------------------------------------------------------------------------------------
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            # ----------------------------------------------------------------------------------------------------------

            # 如果当前迭代小于指定的稠密化迭代次数，则执行稠密化操作---------------------------------------------------------
            if iteration < opt.densify_until_iter:
                # 记录每个像素位置上的最大半径，用于稠密化的修剪
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                # 记录稠密化的统计信息
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                #  当前迭代次数超过指定的起始迭代次数且确保当前迭代次数能够整除设定的稠密化间隔
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    """设置一个额外的尺寸阈值、具体数值是20
                    只有在当前迭代次数 iteration 超过了
                    预先设定的 opt.opacity_reset_interval 时才会使用这个值
                    否则 size_threshold 就保持为 None
                    """
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # 执行稠密化和修剪操作
                    """
                    scene.cameras_extent是相机的范围或者说视野的边界
                    max_grad: 用于计算梯度的参数，防止 NaN 值。
                    min_opacity: 透明度的最小阈值，用于剪枝条件之一。
                    extent: 场景的尺寸，用于剪枝条件之一。
                    max_screen_size: 屏幕尺寸的最大阈值，用于剪枝条件之一
                    densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size)
                    """
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    """
                    0.005 最小透明度阈值 低于这个就删了
                    """

                """每3000次将alpha设置为0、以防止相机前的漂浮
                重置透明度
                """
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    # 重置透明度
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                # 执行优化器步骤
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                # 保存检查点
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    """Start GUI server"""
    # network_gui.init(args.ip, args.port)
    # 配置 PyTorch 运行时以便更好地进行梯度异常检测
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args),
             op.extract(args),
             pp.extract(args),
             args.test_iterations,
             args.save_iterations,
             args.checkpoint_iterations,
             args.start_checkpoint,
             args.debug_from)


    # All done
    print("\nTraining complete.")
