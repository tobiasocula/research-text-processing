import plotly.graph_objects as go
from pathlib import Path
import json

json_dir = Path.cwd() / "constructed_models" / "xdata_models" / "info.json"
with open(json_dir) as f:
    data = json.load(f)

idx = [0, 1, 3, 6, 10, 15, 21, 28]
models = [f"model_{i}" for i in idx]
entries = [data[m] for m in models]

validation_losses = go.Figure()
training_losses = go.Figure()

for i, entry in zip(idx, entries):
    valid_loss = entry["validation_loss_per_epoch"]
    training_loss = entry["training_loss_per_epoch"]
    validation_losses.add_trace(go.Scatter(x=list(range(10)), y=valid_loss, name=str(i)))
    training_losses.add_trace(go.Scatter(x=list(range(10)), y=training_loss, name=str(i)))

validation_losses.show()
training_losses.show()