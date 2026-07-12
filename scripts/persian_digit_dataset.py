import cv2
import json
import numpy as np
import random
from pathlib import Path
from PIL import Image as PILImage, ImageDraw, ImageFont

PERSIAN_DIGITS = ["", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"]
CANVAS = (1600, 1200)

# FONT_PATH = "arial.ttf" 
FONT_PATH = str(Path(__file__).resolve().parent / "Yekan.ttf")

def generate_random_grid() -> np.ndarray:
    """Generates a random 9x9 grid layout where 0 represents an empty cell."""
    grid = np.zeros((9, 9), dtype=int)
    for r in range(9):
        for c in range(9):
            # ~40% chance a cell contains a digit
            if random.random() > 0.6:
                grid[r, c] = random.randint(1, 9)
    return grid


def synthesize_puzzle_persian(grid: np.ndarray, side: int) -> np.ndarray:
    """Creates a clean Sudoku grid image using Persian digits."""
    cell = side // 9
    image = np.full((side, side), 255, np.uint8)
    
    # Draw Grid Lines
    for index in range(10):
        thickness = 7 if index % 3 == 0 else 2
        position = min(index * cell, side - 1)
        cv2.line(image, (position, 0), (position, side), 0, thickness)
        cv2.line(image, (0, position), (side, position), 0, thickness)
    
    # Convert to PIL to render Persian characters safely
    pil_img = PILImage.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype(FONT_PATH, int(cell * 0.6))
    except IOError:
        # Fallback to default if font asset isn't found
        font = ImageFont.load_default()

    for row in range(9):
        for col in range(9):
            digit_val = grid[row, col]
            if digit_val == 0:
                continue
            
            char = PERSIAN_DIGITS[digit_val]
            
            # Calculate text positioning to keep it centered
            bbox = draw.textbbox((0, 0), char, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            x = col * cell + (cell - w) // 2
            y = row * cell + (cell - h) // 2 - bbox[1] # Adjust for baseline offset
            
            draw.text((x, y), char, fill=0, font=font)
            
    return np.array(pil_img)


def place_on_background_random(puzzle: np.ndarray, side: int) -> np.ndarray:
    width, height = CANVAS
    
    # Randomize background base gradient slightly
    bg_start = random.randint(100, 130)
    bg_end = random.randint(170, 210)
    ramp = np.linspace(bg_start, bg_end, width, dtype=np.uint8)
    background = np.tile(ramp, (height, 1))

    # Baseline corners from your code
    base_corners = np.array([[320, 140], [1280, 170], [1250, 1090], [350, 1060]], np.float32)
    
    # Add minor random translation variations to corners (-25 to +25 pixels)
    corner_noise = np.random.uniform(-25, 25, base_corners.shape).astype(np.float32)
    corners = base_corners + corner_noise
    
    source = np.array([[0, 0], [side, 0], [side, side], [0, side]], np.float32)
    matrix = cv2.getPerspectiveTransform(source, corners)
    warped = cv2.warpPerspective(puzzle, matrix, (width, height))
    mask = cv2.warpPerspective(np.full_like(puzzle, 255), matrix, (width, height))
    
    return np.where(mask > 0, warped, background)


def apply_random_degradations(image: np.ndarray) -> np.ndarray:
    # 1. Random Subtle Rotation / Scaling
    if random.random() > 0.5:
        h, w = image.shape[:2]
        angle = random.uniform(-4, 4)
        scale = random.uniform(0.95, 1.0)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        image = cv2.warpAffine(image, matrix, (w, h), borderValue=150)

    # 2. Random Low Light / Exposure
    if random.random() > 0.5:
        factor = random.uniform(0.65, 0.95)
        image = (image.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)

    # 3. Random Gaussian Noise
    if random.random() > 0.6:
        sigma = random.uniform(2, 7)
        noise = np.random.normal(0, sigma, image.shape)
        image = (image.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)

    # 4. Random Blur
    if random.random() > 0.6:
        k_size = 3
        image = cv2.GaussianBlur(image, (k_size, k_size), 0)

    # 5. Random Shadow Gradients
    if random.random() > 0.5:
        h, w = image.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        gradient = ((xx / w + yy / h - random.uniform(0.4, 0.9)) * random.uniform(1.5, 3.0)).clip(0, 1)
        factor = 1.0 - random.uniform(0.15, 0.3) * gradient
        image = (image.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)

    return image


def main():
    output_dir = Path("out/test_persian")
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    total_images = 6
    print(f"Generating {total_images} randomized Persian Sudoku images...")

    for i in range(1, total_images + 1):
        # 1. Randomize grid dimensions slightly per iteration (e.g., between 860 and 940)
        grid_side = random.randint(860, 940)
        
        # 2. Generate random inner puzzle state
        grid_labels = generate_random_grid()
        
        # 3. Build & distort image
        clean_puzzle = synthesize_puzzle_persian(grid_labels, grid_side)
        scened_puzzle = place_on_background_random(clean_puzzle, grid_side)
        final_image = apply_random_degradations(scened_puzzle)
        
        # 4. Save Assets
        file_prefix = f"sudoku_{i:03d}"
        
        # Save image matrix
        cv2.imwrite(str(images_dir / f"{file_prefix}.png"), final_image)
        
        # Save label ground truth matrix (represented as standard 0-9 CSV format)
        np.savetxt(labels_dir / f"{file_prefix}.txt", grid_labels, fmt="%d", delimiter=",")
        
        if i % 25 == 0 or i == total_images:
            print(f" Progress: {i}/{total_images} images processed.")

    print(f"\nDone! Dataset successfully compiled at: {output_dir.resolve()}")


if __name__ == "__main__":
    main()