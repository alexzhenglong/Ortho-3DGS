import os
import cv2
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Standard model configurations for Depth Anything V2
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


def load_model(encoder: str, checkpoint_path: str, device: torch.device) -> DepthAnythingV2:
    """Loads the Depth Anything V2 model with the specified encoder and weights."""
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"Unknown encoder '{encoder}'. Choose from {list(MODEL_CONFIGS.keys())}")

    logger.info(f"Initializing DepthAnythingV2 ({encoder})...")
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    logger.info(f"Loading weights from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    model = model.to(device).eval()
    return model


def process_images(args):
    """Main pipeline for batch depth extraction."""
    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')  # For Apple Silicon
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # 2. Setup Directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_viz:
        viz_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

    # 3. Load Model
    model = load_model(args.encoder, args.checkpoint, device)

    # 4. Gather Images
    valid_extensions = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')
    image_files = [f for f in os.listdir(args.input_dir) if f.endswith(valid_extensions)]

    if not image_files:
        logger.warning(f"No images found in {args.input_dir}")
        return

    logger.info(f"Found {len(image_files)} images. Starting extraction...")

    # 5. Inference Loop
    for filename in tqdm(image_files, desc="Extracting Depth", unit="img"):
        img_path = os.path.join(args.input_dir, filename)
        raw_img = cv2.imread(img_path)

        if raw_img is None:
            logger.error(f"Failed to read {filename}. Skipping.")
            continue

        with torch.no_grad():
            # Infer absolute/relative depth based on metric mode
            depth = model.infer_image(raw_img)

        # Save high-precision raw depth (.npy) for 3DGS/NeRF
        base_name = os.path.splitext(filename)[0]
        np_save_path = os.path.join(args.output_dir, f"{base_name}.npy")
        np.save(np_save_path, depth)

        # Save colorized visualization (.png) if requested
        if args.save_viz:
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth_normalized = depth_normalized.astype(np.uint8)
            # Apply a colormap (INFERNO is standard for depth)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
            viz_save_path = os.path.join(viz_dir, f"{base_name}_depth.png")
            cv2.imwrite(viz_save_path, depth_colored)

    logger.info(f"Processing complete. Results saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch Monocular Depth Extraction using Depth Anything V2")

    # Required arguments
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Directory containing input images.')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory to save the high-precision .npy depth maps.')
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint (.pth file).')

    # Optional arguments
    parser.add_argument('-e', '--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model size/encoder type (default: vitl).')
    parser.add_argument('--save_viz', action='store_true',
                        help='If set, saves colorized depth maps (.png) for visualization.')

    args = parser.parse_args()
    process_images(args)