import os
import json
import itertools
import argparse
from pathlib import Path
from typing import Tuple, List, Iterable, Iterator
from loss_and_score_funcs import *

@register_keras_serializable()
def resize_with_tf(tensors):
    return tf.image.resize(
        tensors[0],
        [tf.shape(tensors[1])[1], tf.shape(tensors[1])[2]],
        method=tf.image.ResizeMethod.BILINEAR,
    )

import numpy as np
import tensorflow as tf

# Global seeds so determinism works with random ops (e.g., shuffle)
import random as _py_random
_py_random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Enable GPU memory growth to avoid upfront full allocation
gpus = tf.config.list_physical_devices('GPU')
for _gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        pass

# Disable XLA JIT to avoid precision-mismatch issues with mixed dtypes
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass


def conv_block(x: tf.Tensor, filters: int, kernel_size: int = 3, dropout: float = 0.0) -> tf.Tensor:
    """Two Conv-BN-ReLU layers, optional spatial dropout.

    This is the basic building block used throughout U-Net for both encoder and decoder paths.
    """
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    if dropout > 0.0:
        x = tf.keras.layers.SpatialDropout2D(dropout)(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def build_unet(
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    base_filters: int = 32,
    depth: int = 4,
    dropout: float = 0.0,
    output_channels: int = 1,
) -> tf.keras.Model:
    """Construct a standard U-Net.

    - Encoder: repeated conv blocks + max pooling (downsampling)
    - Bottleneck: deepest conv block
    - Decoder: upsampling + skip connections from encoder
    - Output: 1-channel sigmoid mask (vertical bars)
    """
    inputs = tf.keras.Input(shape=input_shape)

    skips: List[tf.Tensor] = []
    x = inputs
    filters = base_filters
    for _ in range(depth):
        x = conv_block(x, filters, dropout=dropout)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D(2)(x)
        filters *= 2

    x = conv_block(x, filters, dropout=dropout)

    for skip in reversed(skips):
        filters //= 2
        x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
        # Resize skip using both tensors to avoid symbolic shape issues in Lambda
        skip_resized = tf.keras.layers.Lambda(
            resize_with_tf,
            output_shape=lambda shapes: (None, shapes[1][1], shapes[1][2], shapes[0][3])
        )([skip, x])
        x = tf.keras.layers.Concatenate()([x, skip_resized])

        x = conv_block(x, filters, dropout=dropout)

    outputs = tf.keras.layers.Conv2D(output_channels, 1, activation="sigmoid")(x)
    # Ensure model outputs match the input spatial dimensions exactly
    outputs = tf.keras.layers.Lambda(
        resize_with_tf,
        output_shape=lambda shapes: (None, shapes[1][1], shapes[1][2], shapes[0][3])
    )([outputs, inputs])
    return tf.keras.Model(inputs, outputs, name="unet_vertical_bars")

def compile_model(model: tf.keras.Model, kval: int, lr: float = 1e-3) -> tf.keras.Model:
    """Compile model with optimizer, loss, and useful segmentation metrics."""
    lossfunc = make_soft_cldice_loss(k=kval)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        # loss function
        # loss=bce_dice_loss
        # loss=weighted_bce,
        loss=lossfunc,
        metrics=[dice_coef, iou_score, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    return model


def _read_grayscale_image(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    """Load image from path as float32 [0,1], resize to image_size, single channel."""
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_png(image_bytes, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BILINEAR)
    return image


def get_matched_paths(text_dir: Path, bars_dir: Path) -> Tuple[List[str], List[str]]:
    """List files present in both folders (matched by numeric part of filename)."""
    # import re
    
    # # Get all image files from both directories
    # text_files = [f for f in os.listdir(text_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    # bar_files = [f for f in os.listdir(bars_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    # # Extract numeric parts from filenames
    # def extract_number(filename):
    #     match = re.search(r'(\d+)', filename)
    #     return int(match.group(1)) if match else None
    
    # # Create dictionaries mapping numbers to filenames
    # text_by_number = {extract_number(f): f for f in text_files if extract_number(f) is not None}
    # bar_by_number = {extract_number(f): f for f in bar_files if extract_number(f) is not None}
    
    # # Find common numbers
    # common_numbers = set(text_by_number.keys()) & set(bar_by_number.keys())
    # if len(common_numbers) == 0:
    #     raise ValueError("No matching numeric parts between text_dir and bars_dir files.")
    
    # # Sort by number to ensure consistent ordering
    # common_numbers = sorted(common_numbers)
    
    # # Build matched file lists
    # text_paths = [os.path.join(text_dir, text_by_number[num]) for num in common_numbers]
    # bar_paths = [os.path.join(bars_dir, bar_by_number[num]) for num in common_numbers]
    
    # return text_paths, bar_paths
    # FOR STANDARD IMPLEMENTATION
    #text_paths = [str(Path(text_dir) / f"Ltype{i}.png") for i in range(1, 23)]
    #bar_paths = [str(Path(bars_dir) / f"Lbarc{i}.png") for i in range(1, 23)]
    
    # FOR ALT IMPLEMENTATION
    
    
    text_paths = [str(k) for k in Path(text_dir).iterdir()]
    bar_paths = [str(k) for k in Path(bars_dir).iterdir()]

    
    print('returning:')
    print('text paths:'); print(text_paths)
    print('bar paths:'); print(bar_paths)
    return text_paths, bar_paths



def _dataset_from_paths(
    text_paths: List[str],
    bar_paths: List[str],
    image_size: Tuple[int, int],
    batch_size: int,
    bar_drop_prob: float,
    shuffle: bool,
    augment: bool,
    patch_size: Tuple[int, int] | None = None,
) -> tf.data.Dataset:
    """Build tf.data pipeline from matched path lists.

    - Loads paired (text, bar) images
    - Applies simple augmentations
    - Builds two-channel input [text, bars]; can randomly drop bar channel
    - Batches and prefetches
    """
    ds = tf.data.Dataset.from_tensor_slices((text_paths, bar_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(text_paths), reshuffle_each_iteration=True)

    def _load_pair(text_path: tf.Tensor, bar_path: tf.Tensor):
        text_img = _read_grayscale_image(text_path, image_size)
        bar_img = _read_grayscale_image(bar_path, image_size)
        # Optional random crop to reduce memory footprint while keeping architecture
        if patch_size is not None:
            ph, pw = patch_size
            if ph > 0 and pw > 0:
                concat = tf.concat([text_img, bar_img], axis=-1)
                concat = tf.image.random_crop(concat, size=[ph, pw, tf.shape(concat)[-1]])
                text_img = concat[:, :, :1]
                bar_img = concat[:, :, 1:2]
        if augment:
            if tf.random.uniform(()) > 0.5:
                text_img = tf.image.flip_left_right(text_img)
                bar_img = tf.image.flip_left_right(bar_img)
            text_img = tf.image.random_brightness(text_img, max_delta=0.1)
        # Two-channel input; optionally zero-out bar channel with probability bar_drop_prob
        # if bar_drop_prob > 0.0:
        #     r = tf.random.uniform(())
        #     bar_img_used = tf.cond(r < bar_drop_prob, lambda: tf.zeros_like(bar_img), lambda: bar_img)
        # else:
        #     bar_img_used = bar_img
        bar_img_used = bar_img
        #model_input = tf.concat([text_img, bar_img_used], axis=-1)
        model_input = text_img
        return model_input, bar_img

    ds = ds.map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds



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


def split_paths(
    text_paths: List[str], bar_paths: List[str], pct_train: float
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split matched file lists into train/val using a fraction of training data."""
    #print('inputted to split paths:', text_paths)
    n = len(text_paths)
    idx = int(n * pct_train)
    return text_paths[:idx], bar_paths[:idx], text_paths[idx:], bar_paths[idx:]


def create_ds_from_split(
    text_paths: List[str],
    bar_paths: List[str],
    image_size: Tuple[int, int],
    batch_size: int,
    bar_drop_prob: float,
    augment: bool,
    patch_size: Tuple[int, int] | None,
) -> tf.data.Dataset:
    """Dataset builder for a given split (train or val)."""
    return _dataset_from_paths(
        text_paths,
        bar_paths,
        image_size=image_size,
        batch_size=batch_size,
        bar_drop_prob=bar_drop_prob,
        shuffle=True,
        augment=augment,
        patch_size=patch_size,
    )

def run(
    text_paths: List[str],
    bar_paths: List[str],
    image_height: int,
    image_width: int,
    pct_train_data: float,
    base_filters: int,
    depth: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    bar_drop_prob: float,
    augment: bool,
    save_dir: str,
    model_name: str,
    model_index: int,
    patch_h: int,
    patch_w: int,
    disable_dropout: bool,
    kval: int
):

    # Each combination of parameters gets its own model
    """Build, train, and save a single U-Net model for one parameter combo.

    Returns training history (for JSON logging) and saved model path.
    """
    image_size = (image_height, image_width)
    Xtr, Ytr, Xval, Yval = split_paths(text_paths, bar_paths, pct_train_data)
    print('returned from split_paths:', Xval)

    patch_size = (patch_h, patch_w) if (patch_h > 0 and patch_w > 0) else None
    train_ds = create_ds_from_split(
        Xtr, Ytr, image_size=image_size, batch_size=batch_size, bar_drop_prob=bar_drop_prob, augment=augment, patch_size=patch_size
    )
    val_ds = create_ds_from_split(
        Xval, Yval, image_size=image_size, batch_size=batch_size, bar_drop_prob=0.0, augment=False, patch_size=patch_size
    )

    # Build and compile U-Net for this hyperparameter set (2 input channels: text + (optional) bars)
    effective_dropout = 0.0 if disable_dropout else dropout
    model = build_unet(input_shape=(image_height, image_width, 1), base_filters=base_filters, depth=depth, dropout=effective_dropout)
    model = compile_model(model, lr=lr, kval=kval)

    # Train; cap steps_per_epoch for faster iterations on large datasets
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, steps_per_epoch=min(200, max(1, len(Xtr)//batch_size)))

    save_path = os.path.join(save_dir, f"{model_name}_{model_index}.keras")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save(save_path, include_optimizer=False)

    return hist.history, save_path


def main():
    """Entry point: grid over hyperparameters, train models, and log results to JSON."""
    parser = argparse.ArgumentParser(description="Grid-train U-Net to predict vertical bar masks (Keras style).")
    parser.add_argument("--text_dir", type=str, required=True, help="Directory of text images (grayscale).")
    parser.add_argument("--bars_dir", type=str, required=True, help="Directory of vertical bar masks (grayscale).")
    parser.add_argument("--models_root", type=str, default="models", help="Root folder to save models and JSON.")
    parser.add_argument("--model_name", type=str, default="skeleton_models", help="Model family name.")
    parser.add_argument("--n_procs", type=int, default=1, help="Number of parallel processes.")
    parser.add_argument("--patch_h", type=int, default=0, help="Optional patch height (0 disables patching)")
    parser.add_argument("--patch_w", type=int, default=0, help="Optional patch width (0 disables patching)")
    parser.add_argument("--disable_dropout", action="store_true", help="Disable dropout layers to reduce memory")
    args = parser.parse_args()

    root = Path.cwd()
    models_dir = root / args.models_root / args.model_name
    json_dir = root / args.models_root
    json_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    json_path = json_dir / f"{args.model_name}_info.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            json_data = json.load(f)
    else:
        json_data = {}

    model_desc = "U-Net predicting vertical bar masks from text or bar inputs"

    # Hyperparameter grid (edit these lists to explore the search space)

    # model architecture
    image_heights = [256]
    image_widths = [256]
    pct_train_datas = [0.9]
    base_filters_list = [16, 32] # Number of filters in first layer
    depths = [3] # U-Net depth (downsampling levels)
    dropouts = [0.1] # regularization
    k_values = [10, 15]

    # training
    lrs = [1e-3] # learning rate
    epochs_list = [20, 30] # training rounds
    batch_sizes = [2] # samples per batch

    # data strategy
    bar_drop_probs = [0.0]
    augments = [True]

    # Discover data and ensure filenames match across folders
    if args.text_dir and args.bars_dir and os.path.isdir(args.text_dir) and os.path.isdir(args.bars_dir):
        td = Path.cwd() / args.text_dir
        bd = Path.cwd() / args.bars_dir
        print('td:', td)
        text_paths, bar_paths = get_matched_paths(td, bd)
    else:
        raise ValueError("Please provide valid --text_dir and --bars_dir with matched filenames.")

    session_vars = []
    model_counter = 3

    # Cartesian product of hyperparameters (like your example pipeline)
    param_product = itertools.product(
        image_heights, image_widths, pct_train_datas,
        base_filters_list, depths, dropouts, lrs, epochs_list, batch_sizes, bar_drop_probs, augments,
        k_values
    )

    for params, is_last in with_last_flag(param_product):
        ih, iw, ptd, bf, dp, dr, lr, ep, bs, bdp, aug, kval = params
        session_vars.append(
            (
                text_paths, bar_paths, ih, iw, ptd, bf, dp, dr, lr, ep, bs, bdp, aug,
                str(models_dir), args.model_name, model_counter,
                int(args.patch_h), int(args.patch_w), bool(args.disable_dropout),
                kval
            )
        )

        # Batch Processing
        # Dispatch in batches of size n_procs; flush on last batch
        if (len(session_vars) % max(1, args.n_procs) == 0) or is_last:
            if args.n_procs > 1:
                from multiprocessing import Pool
                with Pool(args.n_procs) as pool:
                    results = pool.starmap(run, session_vars)
            else:
                results = [run(*sv) for sv in session_vars]

            for hist, save_path in results:
                # Persist run configuration and learning curves
                json_data[f"{args.model_name}_{model_counter}"] = {
                    "params": {
                        "image_height": ih,
                        "image_width": iw,
                        "pct_train_data": ptd,
                        "base_filters": bf,
                        "depth": dp,
                        "dropout": dr,
                        "lr": lr,
                        "epochs": ep,
                        "batch_size": bs,
                        "bar_drop_prob": bdp,
                        "augment": aug,
                    },
                    "training_loss_per_epoch": [float(x) for x in hist.get("loss", [])],
                    "validation_loss_per_epoch": [float(x) for x in hist.get("val_loss", [])],
                    "description": model_desc,
                    "saved_model_path": save_path,
                }
                model_counter += 1

            session_vars = []

            # Save/append JSON after each dispatched batch
            with open(json_path, "w") as f:
                json.dump(json_data, f)


if __name__ == "__main__":
    main()