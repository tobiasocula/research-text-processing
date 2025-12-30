import numpy as np
import cv2

def rotate_image(img, angle, expand_canvas=False):
    # Load image as grayscale

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    # Rotation matrix for the given angle
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    if expand_canvas:
        # Compute new bounding dimensions after rotation
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to consider the translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(img, M, (new_w, new_h), borderValue=255)
    else:
        # Keep same size, parts may be cropped
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=255)

    return rotated