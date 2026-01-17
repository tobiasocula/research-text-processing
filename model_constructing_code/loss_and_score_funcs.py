import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
def spatial_loss(y_true, y_pred):

    def render_bars_from_predictions(
            x_coords, y_tops, heights,
            img_shape, bar_width = 2,
            conf_threshold = 0.5
        ):
            """Convert spatial predictions to vertical black bars"""
            h, w = img_shape[:2]
            batch_size = tf.shape(x_coords)[0]
            
            def render_single(batch_idx):
                # Per-pixel confidence: max across spatial dims
                conf = tf.reduce_max(tf.concat([x_coords, y_tops, heights], -1), axis=[1,2])
                conf = tf.cast(conf > conf_threshold, tf.float32)[batch_idx]
                
                # Extract peak locations per column
                x_centers = tf.argmax(tf.reduce_max(x_coords[batch_idx], axis=0), axis=1)
                y_top_peaks = tf.argmax(tf.reduce_max(y_tops[batch_idx], axis=1), axis=1)
                height_peaks = tf.reduce_max(heights[batch_idx], axis=[0, 2])  # [W]
                hh_all = tf.cast(tf.maximum(10.0, height_peaks * h), tf.int32) # [W
                
                # Create empty mask
                mask = tf.zeros((h, w), tf.float32)
                
                # Render bars at top predictions per column
                for col in tf.range(w):
                    xc = tf.cast(x_centers[col], tf.int32)
                    yt = tf.cast(y_top_peaks[col], tf.int32)
                    hh = hh_all[col]  # height specific to this column
                    
                    if xc < w - bar_width:
                        mask = tf.tensor_scatter_nd_update(
                        mask,
                        indices=tf.reshape(
                            tf.stack(
                                [tf.range(yt, tf.minimum(h, yt + hh)),  # ys
                                tf.fill([hh], xc)],                    # xs
                                axis=1,
                            ),
                            [-1, 2],
                        ),
                        updates=tf.ones(hh, tf.float32),
                    )
                
                return mask
            
            masks = tf.map_fn(render_single, tf.range(batch_size), fn_output_signature=tf.TensorSpec((h, w), tf.float32))
            return tf.expand_dims(masks, -1)  # [B,H,W,1]

    # List unpacking: [x_coords, y_tops, heights]
    x_true, y_true_map, h_true = y_true
    x_pred, y_pred_map, h_pred = y_pred
    
    pixel = (
        tf.keras.losses.Huber()(x_true, x_pred) +
        tf.keras.losses.Huber()(h_true, h_pred) +
        tf.keras.losses.Huber()(y_true_map, y_pred_map)
    )

    img_shape = tf.shape(x_true)[1:3]
    pred_bars = render_bars_from_predictions(x_pred, y_pred_map, h_pred, img_shape)
    true_bars = render_bars_from_predictions(x_true, y_true_map, h_true, img_shape)
    dice = 1 - dice_coef(true_bars, pred_bars)

    return pixel + 0.5 * dice

@register_keras_serializable()
def make_soft_cldice_loss(k=10):
    def loss(y_true, y_pred):
        pred = y_pred
        target = y_true
        cl_pred = soft_skeletonize(pred, thresh_width=k)
        target_skeleton = soft_skeletonize(target, thresh_width=k)
        iflat = norm_intersection(cl_pred, target)
        tflat = norm_intersection(target_skeleton, pred)
        intersection = iflat * tflat
        return 1.0 - ((2.0 * intersection) / (iflat + tflat))
    return loss

@register_keras_serializable()
def soft_skeletonize(x, thresh_width=10):

    minpool = (
        lambda y: tf.keras.backend.pool2d(
            y * -1,
            pool_size=(3, 3),
            strides=(1, 1),
            pool_mode="max",
            data_format="channels_last",
            padding="same",
        )
        * -1
    )
    maxpool = lambda y: tf.keras.backend.pool2d(
        y,
        pool_size=(3, 3),
        strides=(1, 1),
        pool_mode="max",
        data_format="channels_last",
        padding="same",
    )

    for _ in range(thresh_width):
        min_pool_x = minpool(x)
        contour = tf.keras.backend.relu(maxpool(min_pool_x) - min_pool_x)
        x = tf.keras.backend.relu(x - contour)
    return x

@register_keras_serializable()
def norm_intersection(center_line, vessel):
    """
    inputs shape  (batch, channel, height, width)
    intersection formalized by first ares
    x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
    """
    smooth = 1.0
    clf = tf.reshape(
        center_line, (tf.shape(center_line)[0], tf.shape(center_line)[1], -1)
    )
    vf = tf.reshape(vessel, (tf.shape(vessel)[0], tf.shape(vessel)[1], -1))
    intersection = tf.keras.backend.sum(clf * vf, axis=-1)
    return (intersection + smooth) / (tf.keras.backend.sum(clf, axis=-1) + smooth)

@register_keras_serializable()
def soft_cldice_loss(target, pred, k=10):
    #pred = tf.transpose(pred, (0, 3, 1, 2))
    #target = tf.transpose(target, (0, 3, 1, 2))

    cl_pred = soft_skeletonize(pred, thresh_width=k)
    target_skeleton = soft_skeletonize(target, thresh_width=k)
    iflat = norm_intersection(cl_pred, target)
    tflat = norm_intersection(target_skeleton, pred)
    intersection = iflat * tflat
    return 1 - ((2.0 * intersection) / (iflat + tflat))

@register_keras_serializable()
def avg_pool_func(x):
    return tf.reduce_mean(x, axis=-1, keepdims=True)

@register_keras_serializable()
def resize_with_tf(tensors):
    return tf.image.resize(
        tensors[0],
        [tf.shape(tensors[1])[1], tf.shape(tensors[1])[2]],
        method=tf.image.ResizeMethod.BILINEAR,
    )

@register_keras_serializable()
def dice_bce_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return dice + tf.reduce_mean(bce)

@register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    # Squeeze last channel if present to always get (batch, H, W)
    if y_true.shape.rank == 4:
        y_true = tf.squeeze(y_true, axis=-1)
    if y_pred.shape.rank == 4:
        y_pred = tf.squeeze(y_pred, axis=-1)
    # Now shapes: (batch, H, W)
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)  # function version: returns (batch, H, W)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)      # (batch, H, W)
    focal = alpha * tf.pow(1. - pt, gamma) * bce
    return tf.reduce_mean(focal)

@register_keras_serializable()
def dice_coef(y_true, y_pred, smooth = 1.0):
    """Dice coefficient (overlap) between predicted and target masks."""
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denominator = tf.reduce_sum(y_true_f + y_pred_f, axis=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return tf.reduce_mean(dice)

@register_keras_serializable()
def weighted_bce(y_true, y_pred, pos_weight=10.0, neg_weight=1.0):
    weights = tf.where(tf.equal(y_true, 1), pos_weight, neg_weight)
    weights = tf.squeeze(weights, axis=-1)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    loss = tf.reduce_mean(bce * weights)
    return loss

@register_keras_serializable()
def dice_loss(y_true, y_pred):
    """Dice loss encourages overlap between predicted and true masks."""
    return 1.0 - dice_coef(y_true, y_pred)

@register_keras_serializable()
def bce_dice_loss(y_true, y_pred):
    """Combined Binary Cross-Entropy + Dice loss for stable training."""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)

@register_keras_serializable()
def iou_score(y_true, y_pred, smooth = 1.0):
    """Intersection-over-Union metric (Jaccard index)."""
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    total = tf.reduce_sum(y_true_f + y_pred_f, axis=1)
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)