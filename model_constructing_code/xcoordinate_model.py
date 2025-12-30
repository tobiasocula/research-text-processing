import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, List, Iterator, Iterable
from loss_and_score_funcs import *
import json
from pathlib import Path
import os
import argparse
import itertools
import sys

def with_last_flag(it: Iterable) -> Iterator[Tuple[Tuple, bool]]:
    """Yield (item, is_last) pairs while iterating an iterable.

    Helps dispatch parameter combos in fixed-size batches and flush the last batch.
    """
    it = iter(it)
    try:
        prev = next(it)
    except StopIteration:
        return
    for val in it:
        yield prev, False
        prev = val
    yield prev, True


def conv_block(x: tf.Tensor, filters: int, dropout: float = 0.0) -> Tuple[tf.Tensor, tf.Tensor]:
    """Conv block with skip connection output."""
    skip = x
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)  # Fixed: removed double tf.keras
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)  # Better than Activation("relu")
    if dropout > 0:
        x = tf.keras.layers.SpatialDropout2D(dropout)(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x, skip

def build_spatial_regression_unet(
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    base_filters: int = 32,
    depth: int = 4,
    dropout: float = 0.1,
):
    """U-Net with spatial regression heads for x/y/height heatmaps."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Encoder with skip connections
    skips = []
    x = inputs
    filters = base_filters
    
    # Downsampling path
    for i in range(depth):
        x, skip = conv_block(x, filters, dropout)
        skips.append(skip)
        x = tf.keras.layers.MaxPool2D(2)(x)
        filters *= 2
    
    # Bottleneck
    x, _ = conv_block(x, filters, dropout)
    
    # Decoder with skip connections
    for i in reversed(range(depth)):
        x = tf.keras.layers.UpSampling2D(2)(x)
        skip = skips[i]
        skip_channels = skip.shape[-1]
        x = tf.keras.layers.Concatenate()([x, skip])
        x, _ = conv_block(x, filters // 2, dropout)
        filters //= 2
    
    # Final conv to match input resolution
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    
    # Spatial regression heads (1x1 convs at full resolution)
    x_map = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    
    # Sigmoid outputs [0,1] normalized, later scaled to image size
    x_coords = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="x_coords")(x_map)
    y_tops = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="y_tops")(x_map)
    heights = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="heights")(x_map)

    model = tf.keras.Model(
    inputs,
    [x_coords, y_tops, heights],  # List, not dict
    name="spatial_regression_unet",
)

    return model



def _read_gray(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def make_paths():
    """
    root_dir/
      imgs/    -> img_0.png, img_1.png, ...
      xdata/   -> x_labels_0.npy, h_labels_0.npy, ...
    Returns:
      image_paths, index_strings (e.g. ['0','1',...]) so we can load npy by index.
    """
    files = sorted(os.listdir(imgs_dir))
    image_paths = [os.path.join(imgs_dir, f) for f in files]
    indices = [str(i) for i in range(len(image_paths))]
    return image_paths, indices

def build_coord_dataset(
    image_size: Tuple[int, int],
    batch_size: int,
    shuffle: bool,
    pct_train_data: float
) -> tf.data.Dataset:
    imgs, indices = make_paths()

    ds = tf.data.Dataset.from_tensor_slices((imgs, indices))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(imgs), reshuffle_each_iteration=True)

    def _load_sample(img_path: tf.Tensor, idx_str: tf.Tensor):
        img = _read_gray(img_path, image_size)  # [H,W,1]
        img_h, img_w = image_size

        def _load_npy_labels_and_make_maps(idx_bytes: bytes):
            idx = idx_bytes.decode("utf-8")
            x_vec = np.load(os.path.join(xdata_dir, f"x_labels_{idx}.npy")).astype("float32")
            h_vec = np.load(os.path.join(xdata_dir, f"h_labels_{idx}.npy")).astype("float32")
            y_vec = np.load(os.path.join(xdata_dir, f"y_labels_{idx}.npy")).astype("float32")

            # heatmaps: [H, W]
            x_map = np.zeros((img_h, img_w), np.float32)
            y_map = np.zeros((img_h, img_w), np.float32)
            h_map = np.zeros((img_h, img_w), np.float32)

            # assume x_vec, y_vec are in pixel coords, h_vec is height in pixels
            num_bars = x_vec.shape[0]
            for i in range(num_bars):
                xc = int(x_vec[i])
                yt = int(y_vec[i])
                hh = int(h_vec[i])

                if xc < 0 or xc >= img_w:
                    continue
                y_bottom = min(img_h, yt + hh)

                # very simple “solid bar” encoding; you can replace with Gaussians
                x_map[yt:y_bottom, xc] = 1.0
                y_map[yt, xc] = 1.0
                h_map[yt:y_bottom, xc] = 1.0

            # add channel dim
            x_map = x_map[..., None]
            y_map = y_map[..., None]
            h_map = h_map[..., None]
            return x_map, h_map, y_map

        x_map, h_map, y_map = tf.numpy_function(
            _load_npy_labels_and_make_maps, [idx_str],
            Tout=(tf.float32, tf.float32, tf.float32),
        )

        # set static shapes so Keras knows them
        x_map.set_shape((*image_size, 1))
        h_map.set_shape((*image_size, 1))
        y_map.set_shape((*image_size, 1))

        # In _load_sample():
        targets = {
            "x_coords": x_map,
            "y_tops": y_map, 
            "heights": h_map
        }
        return img, targets

    train_size = pct_train_data * len(ds)

    train_ds = ds.take(train_size).map(_load_sample, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = ds.skip(train_size).map(_load_sample, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds
    

def run(
    image_height: int,
    image_width: int,
    pct_train_data: float,
    base_filters: int,
    depth: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,    
    model_index: int
):

    # Each combination of parameters gets its own model
    """Build, train, and save a single U-Net model for one parameter combo.

    Returns training history (for JSON logging) and saved model path.
    """
    image_size = (image_height, image_width)
    
    train_ds, val_ds = build_coord_dataset(
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        pct_train_data=pct_train_data,
    )

    model = build_spatial_regression_unet(
        (image_height, image_width, 1),  # tuple, not just image_size
        base_filters=base_filters, 
        depth=depth, 
        dropout=dropout
    )
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss={
        "x_coords": tf.keras.losses.Huber(),
        "y_tops": tf.keras.losses.Huber(),
        "heights": tf.keras.losses.Huber()
    },
    loss_weights={"x_coords": 1.0, "y_tops": 1.0, "heights": 1.0}
)

    # Train; cap steps_per_epoch for faster iterations on large datasets
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    model.save(str(models_dir / f"model_{model_index}.keras"))

    return hist.history


# GLOBAL PARAMS

models_dir = Path.cwd() / "constructed_models" / "xdata_models"

imgs_dir = Path.cwd() / "data" / "train_images_translated"
xdata_dir = Path.cwd() / "data" / "xdata"

models_dir.mkdir(parents=True, exist_ok=True)

def main():
    """Entry point: grid over hyperparameters, train models, and log results to JSON."""
    

    # Hyperparameter grid (edit these lists to explore the search space)

    # model architecture
    image_heights = [256]
    image_widths = [256]
    pct_train_datas = [0.9]

    depths = [3] # U-Net depth (downsampling levels), maybe add 4
    base_filters_list = [8] # Number of filters in first layer, maybe add 16
    dropouts = [0.1] # regularization

    # training
    lrs = [1e-3] # learning rate
    epochs_list = [10] # training rounds
    batch_sizes = [2] # samples per batch

    session_vars = []
    model_counter = 0

    json_path = Path.cwd() / "constructed_models" / "xdata_models" / "info.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            json_data = json.load(f)
            print('loaded json data')
    else:
        print('no json file yet')
        json_data = {}

    param_product = itertools.product(
        image_heights, image_widths, pct_train_datas,
        base_filters_list, depths, dropouts, lrs, epochs_list, batch_sizes
    )

    # iterate over combinations of params and train one model for each
    for params, is_last in with_last_flag(param_product):
        ih, iw, ptd, bf, dp, dr, lr, ep, bs = params
        session_vars.append(
            (
                ih, iw, ptd, bf, dp, dr, lr, ep, bs, model_counter
            )
        )

        results = [run(*sv) for sv in session_vars]  # Single process only

        for hist, save_path in results:
            # Persist run configuration and learning curves
            json_data[f"model_{model_counter}"] = {
                "params": {
                    "image_height": ih,
                    "image_width": iw,
                    "pct_train_data": ptd,
                    "base_filters": bf,
                    "depth": dp,
                    "dropout": dr,
                    "lr": lr,
                    "epochs": ep,
                    "batch_size": bs
                },
                "training_loss_per_epoch": [float(x) for x in hist.get("loss", [])],
                "validation_loss_per_epoch": [float(x) for x in hist.get("val_loss", [])],
                "saved_model_path": save_path,
            }

        with open(json_path, "w") as f:
            json.dump(json_data, f)

if __name__ == "__main__":
    main()