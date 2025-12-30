import numpy as np
import cv2

def scale_text_image(img, scale_factor, target_size=(256, 256)):
    # Load grayscale img

    h, w = img.shape

    # Resize text area by scale factor
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create white canvas of target size
    canvas = np.ones(target_size, dtype=np.uint8) * 255

    # Compute top-left corner for centering scaled image on canvas
    x_offset = (target_size[1] - new_w) // 2
    y_offset = (target_size[0] - new_h) // 2

    # Paste scaled image on canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = scaled_img

    return canvas