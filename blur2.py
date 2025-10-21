import os
import cv2

# Path to the root folder containing 'train', 'test', and 'valid' subfolders
ROOT_DIR = "split"

# Output folder suffix for processed images
OUTPUT_SUFFIX = "_processed4"

# Gaussian blur sigma value
SIGMA = 1


def process_folder(input_dir: str, output_dir: str):
    """
    Load each image in RGB, apply Gaussian blur (σ=3),
    perform histogram equalization on the luminance channel only,
    and save colored output preserving original hues.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        # Process only image files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            continue

        # Read image in BGR (default for OpenCV)
        img_bgr = cv2.imread(os.path.join(input_dir, filename))
        if img_bgr is None:
            print(f"Warning: could not read {filename}")
            continue

        # Convert to RGB to keep color channels consistent
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply Gaussian blur in RGB space
        blurred_rgb = cv2.GaussianBlur(img_rgb, ksize=(0, 0), sigmaX=SIGMA, sigmaY=SIGMA)

        # Convert to YCrCb color space for luminance equalization
        ycrcb = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # Equalize the Y (luminance) channel only
        y_eq = cv2.equalizeHist(y)

        # Merge channels and convert back to RGB
        ycrcb_eq = cv2.merge((y_eq, cr, cb))
        rgb_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)

        # Convert RGB back to BGR for saving
        out_bgr = cv2.cvtColor(rgb_eq, cv2.COLOR_RGB2BGR)

        # Save as colored image
        cv2.imwrite(os.path.join(output_dir, filename), out_bgr)

    print(f"Processed colored images saved to: {output_dir}")


if __name__ == '__main__':
    for subset in ['train', 'test', 'val']:
        in_dir = os.path.join(ROOT_DIR, subset)
        out_dir = os.path.join(ROOT_DIR, subset + OUTPUT_SUFFIX)
        process_folder(in_dir, out_dir)

    print("All subsets processed; outputs are colored and saved.")
