from translate_img import translate_image
from rotate_img import rotate_image
from scale_img import scale_text_image
from pathlib import Path
import cv2
import numpy as np

mask_dir = Path.cwd() / "data" / "train_bars_basis"
imgs_dir = Path.cwd() / "data" / "train_images_basis"

mask_dir_to = Path.cwd() / "data" / "train_bars_augmented"
mask_dir_to.mkdir(exist_ok=True)
imgs_dir_to = Path.cwd() / "data" / "train_images_augmented"
imgs_dir_to.mkdir(exist_ok=True)

idx = 0

for barfile, imgfile in zip(
    list(mask_dir.iterdir()),
    list(imgs_dir.iterdir())
):

    mask = cv2.imread(str(barfile), cv2.IMREAD_GRAYSCALE)
    img_mask = cv2.imread(str(imgfile), cv2.IMREAD_GRAYSCALE)

    for y in range(-50, 50, 5):
        moved_img = translate_image(img_mask, 0, y)
        moved_bars = translate_image(mask, 0, y)
        cv2.imwrite(imgs_dir_to / f"img_{idx}.png", moved_img)
        cv2.imwrite(mask_dir_to / f"bars_{idx}.png", moved_bars)
        idx += 1

    for sc in np.linspace(0.4, 0.9, 20):
        scaled_img = scale_text_image(img_mask, sc)
        scaled_bars = scale_text_image(mask, sc)
        cv2.imwrite(imgs_dir_to / f"img_{idx}.png", scaled_img)
        cv2.imwrite(mask_dir_to / f"bars_{idx}.png", scaled_bars)
        idx += 1

    for angle in range(-45, 44, 5):
        rot_img = rotate_image(img_mask, angle)
        rot_bars = rotate_image(mask, angle)
        cv2.imwrite(imgs_dir_to / f"img_{idx}.png", rot_img)
        cv2.imwrite(mask_dir_to / f"bars_{idx}.png", rot_bars)
        idx += 1