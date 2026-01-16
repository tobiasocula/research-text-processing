import tensorflow as tf
from pathlib import Path
from model_constructing_code.loss_and_score_funcs import *

model_path = Path.cwd() / "constructed_models" / "xdata_models"

for dir in model_path.iterdir():

    if dir.name == "info.json":
        continue
    
    print(model_path / dir.name)
    model = tf.keras.models.load_model(model_path / dir.name, safe_mode=False
    )

    model.summary()