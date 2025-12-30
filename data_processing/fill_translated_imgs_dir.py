from translate_img import translate_image
from pathlib import Path
import cv2

mask_dir = Path.cwd() / "data" / "train_bars_basis"
imgs_dir = Path.cwd() / "data" / "train_images_basis"

mask_dir_to = Path.cwd() / "data" / "train_bars_translated"
mask_dir_to.mkdir(exist_ok=True)
imgs_dir_to = Path.cwd() / "data" / "train_images_translated"
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