import os
import json
import itertools
import argparse
from pathlib import Path
from typing import Tuple, List, Iterable, Iterator
from loss_and_score_funcs import *
import numpy as np
import tensorflow as tf



import random as _py_random
_py_random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)



def spatial_attention(x):
    avg_pool = tf.keras.layers.Lambda(
        avg_pool_func,
        output_shape=lambda input_shape: (input_shape[0], input_shape[1], input_shape[2], 1)
    )(x)

    max_pool = tf.keras.layers.Lambda(
        avg_pool_func,
        output_shape=lambda input_shape: (input_shape[0], input_shape[1], input_shape[2], 1)
    )(x)
    concat = tf.keras.layers.Concatenate()([avg_pool, max_pool])
    attn = tf.keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid')(concat)
    return tf.keras.layers.Multiply()([x, attn])


# --- Encoder Block

def encoder_block(x, filters, pool=True):
    c = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    c = tf.keras.layers.BatchNormalization()(c)
    c = tf.keras.layers.Activation('relu')(c)
    c = tf.keras.layers.Conv2D(filters, 3, padding='same')(c)
    c = tf.keras.layers.BatchNormalization()(c)
    c = tf.keras.layers.Activation('relu')(c)
    p = tf.keras.layers.MaxPooling2D(2)(c) if pool else c
    return c, p

def d2hu_net(input_shape=(256,256,1), base_filters=16):
    inputs = tf.keras.layers.Input(input_shape)

    """
    standard shape of tensor is: (batch, height, width, channels)
    batch: Number of images in a bat
    height, width: Spatial size
    channels: number of feature maps NN is learning
    (each channel picks up a different learned pattern)

    Each pooling (downsampling): halves height and width
    Each block's output: increases channels to let the model store more features as spatial info shrinks

    At each upsampling stage, spatial size doubles

    """
    
    # Shallow branch (for fine edges)
    # sp = downsampled version of convoluted layer s
    s1, sp1 = encoder_block(inputs, base_filters) # shape (None, 256, 256, 16)
    s2, sp2 = encoder_block(sp1, base_filters*2) # shape (None, 128, 128, 32)
    s3, sp3 = encoder_block(sp2, base_filters*4) # shape (None, 64, 64, 64)

    # Deepest shallow branch encoder block; highest level features extracted
    s4, sp4 = encoder_block(sp3, base_filters*8) # shape (None, 32, 32, 128)
    
    # Deep branch (for context)
    # initially: downsample input to reduce spatial size for deep branch
    # Focuses on capturing broader, more global features
    d_p1 = tf.keras.layers.MaxPooling2D(2)(inputs)
    d1, dp1 = encoder_block(d_p1, base_filters) # shape (None, 128, 128, 16)
    d2, dp2 = encoder_block(dp1, base_filters*2) # shape (None, 64, 64, 32)
    d3, dp3 = encoder_block(dp2, base_filters*4) # shape (None, 32, 32, 64)
    d4, _   = encoder_block(dp3, base_filters*8, pool=False) # shape (None, 16, 16, 128)
    
    # Merges the deepest pooled shallow features and deep branch features
    # by concatenation along channels
    bottleneck = tf.keras.layers.Concatenate()([sp4, d4])
    # Combines detailed local and global context in one bottleneck representation
    bottleneck = spatial_attention(bottleneck)
    
    # Decoder with multi-scale skip connections

    # Upsamples bottleneck feature map to higher spatial resolution
    # then: gradually restore spatial resolution
    # Fuse skip features for detailed mask prediction
    u1 = tf.keras.layers.UpSampling2D(2)(bottleneck) # shape u1: (None, 32, 32, 256)
    u1 = tf.keras.layers.Concatenate()([u1, s4, d3])
    u1 = tf.keras.layers.Conv2D(base_filters*4, 3, padding='same', activation='relu')(u1)
    
    u2 = tf.keras.layers.UpSampling2D(2)(u1)
    u2 = tf.keras.layers.Concatenate()([u2, s3, d2])
    u2 = tf.keras.layers.Conv2D(base_filters*2, 3, padding='same', activation='relu')(u2)
    
    u3 = tf.keras.layers.UpSampling2D(2)(u2)
    u3 = tf.keras.layers.Concatenate()([u3, s2, d1])
    u3 = tf.keras.layers.Conv2D(base_filters, 3, padding='same', activation='relu')(u3)

    u4 = tf.keras.layers.UpSampling2D(2)(u3)
    u4 = tf.keras.layers.Concatenate()([u4, s1])
    u4 = tf.keras.layers.Conv2D(base_filters, 3, padding='same', activation='relu')(u4)
    
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(u4)
    model = tf.keras.models.Model(inputs, outputs)
    return model

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
):
    print('HAVING CALLED RUN')
    image_size = (image_height, image_width)
    Xtr, Ytr, Xval, Yval = split_paths(text_paths, bar_paths, pct_train_data)

    patch_size = (patch_h, patch_w) if (patch_h > 0 and patch_w > 0) else None

    train_ds = create_ds_from_split(
        Xtr, Ytr, image_size=image_size, batch_size=batch_size, bar_drop_prob=bar_drop_prob, augment=augment, patch_size=patch_size
    )
    val_ds = create_ds_from_split(
        Xval, Yval, image_size=image_size, batch_size=batch_size, bar_drop_prob=0.0, augment=False, patch_size=patch_size
    )

    effective_dropout = 0.0 if disable_dropout else dropout
    model = d2hu_net(input_shape=(image_height, image_width, 1))
    print('compiling model')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        # loss function
        # loss=bce_dice_loss
        # loss=weighted_bce,
        loss=soft_cldice_loss,
        metrics=[dice_coef, iou_score, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    print('fitting model')

    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, steps_per_epoch=min(200, max(1, len(Xtr)//batch_size)))

    save_path = os.path.join(save_dir, f"{model_name}_{model_index}.keras")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save(save_path)

    return hist.history, save_path

def _read_grayscale_image(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    """Load image from path as float32 [0,1], resize to image_size, single channel."""
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_png(image_bytes, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BILINEAR)
    return image

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
                print('concat object:', concat)
                print('concat shape:', concat.shape)
                concat = tf.image.random_crop(concat, size=[ph, pw, tf.shape(concat)[-1]])
                print('concat shape after crop:', concat.shape)
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

import sys

def main():
    """Entry point: grid over hyperparameters, train models, and log results to JSON."""
    parser = argparse.ArgumentParser(description="Grid-train U-Net to predict vertical bar masks (Keras style).")
    parser.add_argument("--text_dir", type=str, required=True, help="Directory of text images (grayscale).")
    parser.add_argument("--bars_dir", type=str, required=True, help="Directory of vertical bar masks (grayscale).")
    parser.add_argument("--models_root", type=str, default="models", help="Root folder to save models and JSON.")
    parser.add_argument("--model_name", type=str, default="dual_models", help="Model family name.")
    parser.add_argument("--n_procs", type=int, default=1, help="Number of parallel processes.")
    parser.add_argument("--patch_h", type=int, default=0, help="Optional patch height (0 disables patching)")
    parser.add_argument("--patch_w", type=int, default=0, help="Optional patch width (0 disables patching)")
    parser.add_argument("--disable_dropout", action="store_true", help="Disable dropout layers to reduce memory")
    args = parser.parse_args()

    root = Path.cwd()
    models_dir = root / args.models_root / args.model_name
    print('MODELS DIR:', models_dir)
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
    base_filters_list = [2] # Number of filters in first layer # normal: 8
    depths = [3] # U-Net depth (downsampling levels) # normal: 3, 4
    dropouts = [0.1] # regularization
    #loss_funcs = [focal_loss, dice_bce_loss]

    # training
    lrs = [1e-3] # learning rate
    epochs_list = [2] # training rounds # normal: 10
    batch_sizes = [2] # samples per batch

    # data strategy
    bar_drop_probs = [0.0]
    augments = [True]

    # Discover data and ensure filenames match across folders
    if args.text_dir and args.bars_dir and os.path.isdir(args.text_dir) and os.path.isdir(args.bars_dir):        
        text_paths = [str(k) for k in Path(args.text_dir).iterdir()]
        bar_paths = [str(k) for k in Path(args.bars_dir).iterdir()]
    else:
        raise ValueError("Please provide valid --text_dir and --bars_dir with matched filenames.")

    session_vars = []
    model_counter = 0

    # Cartesian product of hyperparameters (like your example pipeline)
    param_product = itertools.product(
        image_heights, image_widths, pct_train_datas,
        base_filters_list, depths, dropouts, lrs, epochs_list, batch_sizes, bar_drop_probs, augments
    )

    print('iteration begins..')

    for params, is_last in with_last_flag(param_product):
        ih, iw, ptd, bf, dp, dr, lr, ep, bs, bdp, aug = params
        session_vars.append(
            (
                text_paths, bar_paths, ih, iw, ptd, bf, dp, dr, lr, ep, bs, bdp, aug,
                str(models_dir), args.model_name, model_counter,
                int(args.patch_h), int(args.patch_w), bool(args.disable_dropout),
            )
        )

        # Batch Processing
        # Dispatch in batches of size n_procs; flush on last batch
        print('beginning pooling')
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