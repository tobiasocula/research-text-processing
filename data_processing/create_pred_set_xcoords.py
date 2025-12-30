from pathlib import Path
import cv2
import numpy as np
from bars_from_mask import bars_from_mask

dir_from = Path.cwd() / "data" / "train_bars_translated"
dir_to = Path.cwd() / "data" / "xdata"

dir_to.mkdir(exist_ok=True)

idx = 0

for barfile in dir_from.iterdir():
    mask = cv2.imread(barfile, cv2.IMREAD_GRAYSCALE)
    x_labels, h_labels, y_labels = bars_from_mask(mask)

    np.save(str(dir_to / f"x_labels_{idx}.npy"), x_labels)
    np.save(str(dir_to / f"h_labels_{idx}.npy"), h_labels)
    np.save(str(dir_to / f"y_labels_{idx}.npy"), y_labels)

    idx += 1