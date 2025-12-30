import numpy as np
import cv2

def bars_from_mask(mask, max_bars=100, min_area=3):
    # if mask.max() <= 1:
    #     bin_mask = (mask > 0.5).astype(np.uint8) * 255
    # else:
    #     bin_mask = (mask > 0).astype(np.uint8) * 255

    bin_mask = (mask != 255).astype(np.uint8) * 255

    H, W = bin_mask.shape

    """
    labels[y, x] tells you which component that pixel belongs to.
    num_labels is just # of connected components
    stats: shape (num_labels, 5)
    Each row corresponds to one label (0..num_labels-1).

Each column stores a property of that component:

stats[label, cv2.CC_STAT_LEFT] → x of left side of bounding box

stats[label, cv2.CC_STAT_TOP] → y of top of bounding box

stats[label, cv2.CC_STAT_WIDTH] → width of bounding box (in pixels)

stats[label, cv2.CC_STAT_HEIGHT] → height of bounding box (in pixels)

stats[label, cv2.CC_STAT_AREA] → number of pixels in the component
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask)

    bar_info = []

    """
    for each component (non-background):
    read boundary box, skip tiny noise, compute x-center and height, normalize to [0, 1] for training
    """
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        x_center = x + w / 2.0  # pixel x center
        height = h              # pixel height
        y_top = y
        bar_info.append((x_center, height, y_top))

    bar_info = sorted(bar_info, key=lambda b: b[0])[:max_bars]
    x_coords = np.array([b[0] for b in bar_info] + [0.0] * (max_bars - len(bar_info)))
    heights = np.array([b[1] for b in bar_info] + [0.0] * (max_bars - len(bar_info)))
    y_tops = np.array([b[2] for b in bar_info] + [0.0] * (max_bars - len(bar_info)))
    return x_coords, heights, y_tops