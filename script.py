import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import logging
import gc
import time
import os

from Models import UNet, TRANSFORMS, load_model

# Setup logging
def setup_logging(log_level, log_file):
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    logging.basicConfig(
        level=log_levels.get(log_level, logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("ImageSegmentation")

# Process image
def process_image(source: str | np.ndarray, logger):
    try:
        if isinstance(source, str):
            logger.info(f"Loading image from file: {source}")
            image = Image.open(source).convert("RGB")
        elif isinstance(source, np.ndarray):
            logger.info("Processing image from numpy array.")
            image = Image.fromarray(source).convert("RGB")
        else:
            logger.error("Invalid input type for image source.")
            raise ValueError("Input should be a file path (str) or a numpy ndarray.")
        return TRANSFORMS(image).unsqueeze(0), image.size
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

# Generate overlay
def generate_overlay(image, output_mask, alpha):
    try:
        NUM_CLASSES = 91
        colors = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)
        mask_color = colors[output_mask]
        IMAGE_SIZE = (224, 224)
        image_np = np.array(image.resize(IMAGE_SIZE))
        binary_mask = (output_mask > 0).astype(np.uint8) * 255
        binary_mask_color = np.stack([binary_mask] * 3, axis=-1)
        overlay = cv2.addWeighted(image_np, 1 - alpha, binary_mask_color, alpha, 0)
        img = np.where(binary_mask_color > 0, image_np, 0)
        return image_np, binary_mask, overlay, img
    except Exception as e:
        logging.error(f"Error generating overlay: {e}")
        raise

# Visualize image segmentation
def image_visualize(model, device, image_path, alpha, logger):
    logger.info(f"Starting segmentation visualization for image: {image_path}")
    input_tensor, image_size = process_image(image_path, logger)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
    output_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    image = Image.open(image_path).convert("RGB")
    image_np, binary_mask, overlay, img = generate_overlay(image, output_mask, alpha)
    logger.info("Displaying segmented image.")
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    titles = ["Original Image", "Merged Binary Mask", "Overlay", "Segmented Image"]
    images = [image_np, binary_mask, overlay, img]
    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img, cmap='gray' if title == "Merged Binary Mask" else None)
        ax.set_title(title)
        ax.axis("off")
    gc.collect()
    plt.show()

# Process camera feed
def using_camera(model, device, alpha, logger):
    logger.info("Starting real-time camera segmentation.")
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame from camera.")
                break
            input_tensor, image_size = process_image(frame, logger)
            input_tensor = input_tensor.to(device)
            with torch.no_grad():
                output = model(input_tensor)
            output_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            image = Image.fromarray(frame)
            _, _, overlay, _ = generate_overlay(image, output_mask, alpha)
            result = cv2.resize(overlay, image_size)
            gc.collect()
            cv2.imshow("Overlay", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exiting camera loop.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Main function
def main():
    parser = argparse.ArgumentParser(description="Image segmentation with U-Net model.")
    parser.add_argument('--image', type=str, help="Path to the image file for segmentation.")
    parser.add_argument('--camera', action='store_true', help="Use real-time camera segmentation.")
    parser.add_argument('--model', type=str, default='models/unet_v1.pth', help="Path to trained model.")
    parser.add_argument('-v', '--verbose', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],help="Logging verbosity level.")
    parser.add_argument('-l', '--logfile', default=f'logs/{int(time.time())}.log', type=str, help="Log file.")
    parser.add_argument('-d', '--device', choices=['cpu', 'cuda'], default='cpu', help="Device to run the model on.")
    parser.add_argument('--alpha', type=float, default=0.5, help="Transparency for the overlay.")
    
    logfile_directory = os.path.dirname(args.logfile)
    if not os.path.exists(logfile_directory):
        os.makedirs(logfile_directory)
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose, args.logfile)
    logger.info(f"Using device: {args.device}")
    try:
        model = load_model(args.device, model_path=args.model)
        logger.info(f"Model loaded successfully from {args.model}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    if args.camera:
        using_camera(model, args.device, args.alpha, logger)
    elif args.image:
        image_visualize(model, args.device, args.image, args.alpha, logger)
    else:
        logger.error("Please provide an image path or use the camera option.")

if __name__ == '__main__':
    main()