from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import sys

model_name = "model_3"
model_path = Path.cwd() / "constructed_models" / "xdata_models" / f"{model_name}.keras"
img_path = Path.cwd() / "data" / "test_images" / "sample_text_scaled.png"
#save_to = Path.cwd() / "running_models" / "output_predicts_xmodels" / model_name
save_to = Path.cwd() / "running_models" / "output_predicts_xmodels_scaled" / model_name
Path.mkdir(save_to)

model = tf.keras.models.load_model(model_path)

raw_image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
H, W = 256, 256
raw_image = cv2.resize(raw_image, (W, H), interpolation=cv2.INTER_LINEAR)

# Make two copies with different processing
model_input = raw_image.astype("float32") / 255.0  # [H,W] → [H,W] normalized [0,1]
model_input = np.expand_dims(model_input, axis=-1)  # [H,W,1]
model_input = np.expand_dims(model_input, axis=0)   # [1,H,W,1] → MODEL READY

vis_image = raw_image.copy()  # [H,W] uint8 [0,255] → VISUALIZATION READY
vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)  # [H,W,3] for drawing

# Predict - LIST of 3 heatmaps
predictions = model.predict(model_input, verbose=0)
# predictions: contains 3 heatmaps: tensors of shape (1, 256, 256, 1)

# Extract: [x_coords, y_tops, heights]
x_heatmap = predictions[0][0, :, :, 0]  # [256,256]
y_heatmap = predictions[1][0, :, :, 0]  # [256,256]  
h_heatmap = predictions[2][0, :, :, 0]  # [256,256]

# Visualize everything
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0,0].imshow(raw_image, cmap='gray')
axes[0,0].set_title('Original Image')
axes[0,0].axis('off')

im1 = axes[0,1].imshow(x_heatmap, cmap='hot', vmin=0, vmax=1)
axes[0,1].set_title('X_coords Heatmap')
plt.colorbar(im1, ax=axes[0,1])

im2 = axes[1,0].imshow(y_heatmap, cmap='hot', vmin=0, vmax=1)
axes[1,0].set_title('Y_tops Heatmap') 
plt.colorbar(im2, ax=axes[1,0])

im3 = axes[1,1].imshow(h_heatmap, cmap='hot', vmin=0, vmax=1)
axes[1,1].set_title('Heights Heatmap')
plt.colorbar(im3, ax=axes[1,1])

plt.tight_layout()
#plt.show()
plt.savefig(str(save_to / f"heatmap_{model_name}.png"))

# Simple bar extraction and overlay
bars = []
for col in range(W):
    if np.max(x_heatmap[:, col]) > 0.5:  # Confidence threshold
        x = col

        col_y = y_heatmap[:, col]
        y_top = int(np.argmax(col_y))

        col_h = h_heatmap[:, col]

        # Only consider heights BELOW the top
        below = col_h[y_top:]
        # Simple rule: height = number of pixels above threshold
        thresh = 0.3
        active = below > thresh
        if np.any(active):
            height = int(active.sum())
        else:
            height = 0

        bars.append((x, y_top, height))

# Draw
for x, y_top, height in bars:
    x, y_top = int(x), int(y_top)
    height = int(np.clip(height, 5, H - y_top))   # clamp within image
    bottom = y_top + height

    # show
    cv2.rectangle(vis_image, (x-1, y_top), (x+1, bottom), (0, 0, 255), 2)

fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
ax.set_title(f"Original + Predicted Bars (red) - {len(bars)} bars")
ax.axis('off')
plt.savefig(str(save_to / f"pred_bars_{model_name}.png"))
#plt.show()