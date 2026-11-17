# Remote GPU racket-pose training bundle

This archive contains the Good-Badminton source code, the prepared
`datasets/racket_pose_v1` training data, and `weights/yolo11n-pose.pt`.
It intentionally excludes Git history, local virtual environments, generated
outputs, raw videos/annotations, logs, and local secrets.

## On the GPU server

Extract the archive, then work from the extracted `Good-Badminton` directory.

```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
python3 -c "import ultralytics; print(ultralytics.__version__)"
```

The expected PyTorch check is `True` for CUDA. Do not run
`pip install -r requirements.txt`: that file deliberately pins a CPU PyTorch
build for the Windows local workflow. Preserve the server's CUDA-compatible
PyTorch installation. If Ultralytics is missing, install a version compatible
with that already-working environment before continuing.

Then run:

```bash
cd Good-Badminton
export YOLO_CONFIG_DIR="$PWD/.yolo_config"
yolo pose train \
  model=weights/yolo11n-pose.pt \
  data=datasets/racket_pose_v1/racket-pose.yaml \
  epochs=100 imgsz=960 batch=-1 patience=30 \
  device=0 workers=4 \
  project=runs/racket_pose name=baseline_01
```

The dataset YAML has no host-specific `path` value. Its directory is the
dataset root, so it remains valid after the archive is extracted anywhere.
Review `datasets/racket_pose_v1/skipped_groups.csv` before treating a model as
final: those entries were excluded because the corresponding racket Group did
not contain exactly five keypoints.
