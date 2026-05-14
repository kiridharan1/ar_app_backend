# SOP: Train YOLO Locally on macOS and Deploy to VM via SSH

## 1. Short Answer

Yes. You can train locally on your Mac, extract the trained `.pt` weights, and deploy this backend to a VM.

This backend expects the model at:

- `model/final_model.pt`

## 2. Prerequisites

## 2.1 Local Mac

1. macOS with Python 3.10+
2. Terminal access
3. Dataset in YOLO format
4. Enough disk space for training runs

Optional:

1. Apple Silicon Mac (M1/M2/M3) for faster training with MPS

## 2.2 VM

1. Ubuntu 22.04+ (or similar Linux)
2. SSH access with sudo
3. Open inbound backend port (default 3000)

## 3. Local Training on macOS

## 3.1 Create training environment

```bash
cd /path/to/ar_app_backend
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install --upgrade pip
pip install ultralytics
```

## 3.2 Prepare dataset (repo `data/` folders)

Use this repo layout:

```text
ar_app_backend/
  data/
    images/
    labels/
```

If your files are zipped, extract into `data/` so image and label files are paired by filename stem.

Example:

```bash
cd /path/to/ar_app_backend
mkdir -p data/images data/labels
unzip /path/to/images.zip -d data/images
unzip /path/to/labels.zip -d data/labels
```

Create train/val split folders from `data/images` and `data/labels`:

```bash
cd /path/to/ar_app_backend
mkdir -p data/train/images data/train/labels data/val/images data/val/labels
python3 - <<'PY'
from pathlib import Path
import random
import shutil

random.seed(42)
root = Path('data')
src_images = root / 'images'
src_labels = root / 'labels'
train_img = root / 'train' / 'images'
train_lbl = root / 'train' / 'labels'
val_img = root / 'val' / 'images'
val_lbl = root / 'val' / 'labels'

images = sorted([p for p in src_images.glob('*') if p.is_file()])
random.shuffle(images)
split = int(len(images) * 0.8)

for i, img in enumerate(images):
    stem = img.stem
    lbl = src_labels / f"{stem}.txt"
    target_img = train_img if i < split else val_img
    target_lbl = train_lbl if i < split else val_lbl
    shutil.copy2(img, target_img / img.name)
    if lbl.exists():
        shutil.copy2(lbl, target_lbl / lbl.name)

print(f"Total images: {len(images)}")
print(f"Train images: {len(list(train_img.glob('*')))}")
print(f"Val images: {len(list(val_img.glob('*')))}")
PY
```

Create `data/data.yaml`:

```yaml
path: /absolute/path/to/ar_app_backend/data
train: train/images
val: val/images
nc: 3
names: [class1, class2, class3]
```

Replace `nc` and `names` with your class count and class list.

## 3.3 Start training

Choose one model size:

1. Fast/smaller: `yolo11n.pt`
2. Better accuracy/slower: `yolo11s.pt`

### CPU training (works on all Macs)

```bash
yolo detect train \
  data=/absolute/path/to/ar_app_backend/data/data.yaml \
  model=yolo11s.pt \
  epochs=60 \
  imgsz=640 \
  device=cpu
```

### Apple Silicon (MPS) training

```bash
yolo detect train \
  data=/absolute/path/to/ar_app_backend/data/data.yaml \
  model=yolo11s.pt \
  epochs=60 \
  imgsz=640 \
  device=mps
```

### Fast iteration profile (recommended while experimenting)

Use this when you want much faster feedback and can accept a small accuracy tradeoff:

```bash
yolo detect train \
  data=/absolute/path/to/ar_app_backend/data/data.yaml \
  model=yolo11n.pt \
  epochs=25 \
  imgsz=512 \
  device=mps \
  cache=True \
  project=/absolute/path/to/ar_app_backend/runs/detect \
  name=train_fast
```

After you find good settings, run a final quality pass with `yolo11s.pt`, `imgsz=640`, and more epochs.

Training output is usually created in:

```text
runs/detect/train/
```

Best checkpoint:

```text
runs/detect/train/weights/best.pt
```

## 3.4 Validate and quick test

```bash
yolo detect val \
  model=runs/detect/train/weights/best.pt \
  data=/absolute/path/to/ar_app_backend/data/data.yaml

yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=/absolute/path/to/ar_app_backend/data/val/images \
  save=True
```

## 3.5 Extract model for backend

Copy best model into this backend format:

```bash
mkdir -p model
cp runs/detect/train/weights/best.pt model/final_model.pt
```

Optional archive for sharing/backups:

```bash
zip -r my_model_mac.zip model/final_model.pt runs/detect/train
```

## 4. Deploy to VM via SSH

## 4.1 Copy code and model to VM

### Option A: Clone repo on VM, then upload model only

```bash
ssh <user>@<vm_ip>
git clone <repo_url> ar_app_backend
exit
scp model/final_model.pt <user>@<vm_ip>:~/ar_app_backend/model/final_model.pt
```

### Option B: Rsync full local project

```bash
rsync -avz --exclude .git /path/to/ar_app_backend/ <user>@<vm_ip>:~/ar_app_backend/
```

## 4.2 VM setup

```bash
ssh <user>@<vm_ip>
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git

cd ~/ar_app_backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 4.3 Configure backend

Edit `config.py`:

1. `MODEL_PATH = "model/final_model.pt"`
2. `SERVER_HOST = "0.0.0.0"`
3. `SERVER_PORT = 3000` (or your preferred port)
4. `DEVICE = "cpu"` for CPU VM or `"cuda:0"` for GPU VM
5. Set `ACTIVE_PRESET` to one of:
   - `"HIGH_ACCURACY"`
   - `"BALANCED"`
   - `"HIGH_RECALL"`
   - `"VERY_HIGH_ACCURACY"`

## 4.4 Run backend

```bash
cd ~/ar_app_backend
source .venv/bin/activate
python server.py
```

## 4.5 Keep backend running with systemd (recommended)

```bash
sudo tee /etc/systemd/system/ar-app-backend.service > /dev/null <<'EOF'
[Unit]
Description=AR App YOLO Backend
After=network.target

[Service]
Type=simple
User=<user>
WorkingDirectory=/home/<user>/ar_app_backend
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/<user>/ar_app_backend/.venv/bin/python /home/<user>/ar_app_backend/server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ar-app-backend
sudo systemctl start ar-app-backend
sudo systemctl status ar-app-backend --no-pager
```

## 5. Verification

On VM:

```bash
ss -tulpen | grep 3000
journalctl -u ar-app-backend -f
```

Optional functional test:

```bash
cd ~/ar_app_backend
source .venv/bin/activate
python test_image.py
```

## 6. Common Issues

1. Training is very slow on Mac:

- Use `device=mps` on Apple Silicon
- Use `model=yolo11n.pt`
- Reduce `imgsz` to 512
- Reduce `epochs` for initial experiments (for example 20 to 30)
Reduce `epochs` for initial experiments (for example 20 to 30)
- Use `cache=True` to avoid repeated disk reads

1. Model file not found on VM:
   - Verify `~/ar_app_backend/model/final_model.pt` exists
   - Verify `MODEL_PATH` in `config.py`

2. Preset not applying:
   - Use exact values in `ACTIVE_PRESET`: `HIGH_ACCURACY`, `BALANCED`, `HIGH_RECALL`, `VERY_HIGH_ACCURACY`

3. Client cannot connect:
   - Confirm firewall/security group allows port 3000
   - Confirm server binds to `0.0.0.0`

## 7. Quick Command Block

```bash
# local train
cd /path/to/ar_app_backend
python3 -m venv .venv-train && source .venv-train/bin/activate
pip install -U pip ultralytics
yolo detect train data=/absolute/path/to/ar_app_backend/data/data.yaml model=yolo11s.pt epochs=60 imgsz=640 device=mps
mkdir -p model && cp runs/detect/train/weights/best.pt model/final_model.pt

# upload model
scp model/final_model.pt <user>@<vm_ip>:~/ar_app_backend/model/final_model.pt

# vm run
ssh <user>@<vm_ip>
cd ~/ar_app_backend
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
python server.py
```
