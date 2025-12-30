import json
from pathlib import Path

dir = Path.cwd() / "constructed_models" / "xdata_models"
json_data = {}
json_data[f"model_0"] = {
    "params": {
        "pct_train_data": 0.9,
        "base_filters": 8,
        "depth": 3,
        "dropout": 0.3,
        "lr": 1e-3,
        "epochs": 10,
        "batch_size": 2
    },
    "training_loss_per_epoch": "None",
    "validation_loss_per_epoch": "None",
    "saved_model_path": "model_0.keras",
}

with open(dir / "info.json", "w") as f:
    json.dump(json_data, f)