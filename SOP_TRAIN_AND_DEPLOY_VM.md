# SOP: Train YOLO Model and Deploy ar_app_backend to a VM via SSH

## 1. Purpose

This SOP explains how to:

1. Train or fine-tune a YOLO model that produces `final_model.pt`.
2. Deploy this backend to a Linux VM using SSH.
3. Run the Socket.IO detection server reliably (manual run and systemd service).

This repository is an inference backend. Training is done with the notebook `Train_YOLO_Models (1).ipynb` (Google Colab style), then the trained weights are copied into this project at `model/final_model.pt`.

## 2. Repo-Specific Facts (Read Before Starting)

1. Server entry point: `server.py`
2. Runtime configuration: `config.py`
3. Python dependencies: `requirements.txt`
4. Default model path expected by server: `model/final_model.pt`
5. Default host/port in config: `0.0.0.0:3000`
6. Transport: Socket.IO events (`frame`, `detections`, `get_stats`, etc.)

Important note:
`config.py` currently sets `ACTIVE_PRESET = "PRESET_HIGH_ACCURACY"`, but `server.py` expects one of `HIGH_ACCURACY`, `BALANCED`, `HIGH_RECALL`, or `VERY_HIGH_ACCURACY`. To ensure preset application works, use one of those exact names.

## 3. Prerequisites

## 3.1 Local machine (training or packaging)

1. Python 3.10+
2. Git
3. SSH client
4. Access to dataset in YOLO format:
   - images/train, labels/train
   - images/val, labels/val
   - dataset YAML file (example: `data.yaml`)

## 3.2 VM (deployment target)

1. Ubuntu 22.04+ (or similar Linux)
2. SSH access and sudo permissions
3. Python 3.10+ and venv support
4. Open inbound port 3000 (or your configured port)
5. Optional but recommended: reverse proxy (Nginx) and TLS

## 4. Training SOP (Notebook-Based, Colab)

If you already have a trained model, skip to section 5.

This section mirrors the notebook workflow in `Train_YOLO_Models (1).ipynb`.

## 4.1 Open notebook and verify GPU

1. Open `Train_YOLO_Models (1).ipynb` in Colab.
2. Select a GPU runtime.
3. Run:

```bash
!nvidia-smi
```

## 4.2 Upload dataset from Google Drive

The notebook expects a zip file at:

```text
/content/gdrive/MyDrive/yolo/data.zip
```

Run the notebook cells to:

1. Mount Drive.
2. Copy zip to `/content`.
3. Unzip to `/content/custom_data`.

Expected input structure inside the unzipped dataset:

```text
/content/custom_data/
  images/
  labels/
  classes.json
```

## 4.3 Split dataset into train/validation

Notebook behavior:

1. Creates:
   - `/content/data/train/images`
   - `/content/data/train/labels`
   - `/content/data/validation/images`
   - `/content/data/validation/labels`
2. Uses random split controlled by:
   - `train_percent = 0.3` (30% train, 70% validation)

If you want a more typical split, change `train_percent` before running that cell.

## 4.4 Install training dependency

```bash
!pip install ultralytics
```

## 4.5 Generate data.yaml from classes file

Notebook creates:

```text
/content/data.yaml
```

with:

1. `path: /content/data`
2. `train: train/images`
3. `val: validation/images`
4. `nc` and `names` derived from `/content/custom_data/classes.json`

## 4.6 Train model

Notebook training command:

```bash
!yolo detect train data=/content/data.yaml model=yolo11s.pt epochs=60 imgsz=640
```

Trained best model is produced at:

```text
/content/runs/detect/train/weights/best.pt
```

## 4.7 Test model in notebook

Notebook prediction command:

```bash
!yolo detect predict model=runs/detect/train/weights/best.pt source=data/validation/images save=True
```

Prediction images are saved under:

```text
/content/runs/detect/predict/
```

## 4.8 Export model artifact from Colab

Notebook packaging steps:

1. Copies best weights to:
   - `/content/my_model/my_model.pt`
2. Copies training run folder to:
   - `/content/my_model/train`
3. Creates downloadable zip:
   - `/content/my_model.zip`

Then use `files.download('/content/my_model.zip')`.

## 4.9 Convert exported model for backend

After downloading and extracting on local machine:

```bash
cp my_model/my_model.pt model/final_model.pt
```

This is the file path expected by the backend (`MODEL_PATH = "model/final_model.pt"` in `config.py`).

## 5. Deployment SOP via SSH

## 5.1 SSH into VM

```bash
ssh <user>@<vm_public_ip>
```

Optional first-time hardening:

```bash
sudo apt update && sudo apt -y upgrade
sudo apt install -y ufw
sudo ufw allow OpenSSH
sudo ufw allow 3000/tcp
sudo ufw --force enable
```

## 5.2 Install system packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```

If using GPU, install CUDA/NVIDIA drivers appropriate to your VM image before running the app.

## 5.3 Copy project to VM

Choose one method.

Method A: Git clone on VM

```bash
git clone <repo_url> ar_app_backend
cd ar_app_backend
```

Method B: Rsync from local machine

```bash
rsync -avz --exclude .git /path/to/ar_app_backend/ <user>@<vm_public_ip>:~/ar_app_backend/
```

## 5.4 Ensure model file exists

On VM, verify:

```bash
ls -lh model/final_model.pt
```

If missing, upload it:

```bash
scp /local/path/final_model.pt <user>@<vm_public_ip>:~/ar_app_backend/model/final_model.pt
```

## 5.5 Create runtime environment

```bash
cd ~/ar_app_backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 5.6 Configure app

Edit `config.py` for your VM:

1. Keep `SERVER_HOST = "0.0.0.0"`
2. Set `SERVER_PORT` (default 3000)
3. Set `MODEL_PATH = "model/final_model.pt"`
4. Set `DEVICE`:
   - CPU VM: `"cpu"`
   - GPU VM: typically `"cuda:0"`
5. Set `ACTIVE_PRESET` to one of:
   - `"HIGH_ACCURACY"`
   - `"BALANCED"`
   - `"HIGH_RECALL"`
   - `"VERY_HIGH_ACCURACY"`

## 5.7 Smoke test run (foreground)

```bash
cd ~/ar_app_backend
source .venv/bin/activate
python server.py
```

Expected startup logs:

1. Configuration validation passed
2. Model loaded successfully
3. Server listening at configured host/port

Stop with Ctrl+C after validation.

## 6. Production Run as systemd Service

## 6.1 Create service file

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
```

Replace `<user>` with the VM username.

## 6.2 Enable and start service

```bash
sudo systemctl daemon-reload
sudo systemctl enable ar-app-backend
sudo systemctl start ar-app-backend
sudo systemctl status ar-app-backend --no-pager
```

## 6.3 View logs

```bash
journalctl -u ar-app-backend -f
```

## 7. Functional Verification

## 7.1 Port check

```bash
ss -tulpen | grep 3000
```

## 7.2 Socket.IO test using included client

From VM (or a machine that can reach the VM):

```bash
cd ~/ar_app_backend
source .venv/bin/activate
python test_image.py
```

Expected:

1. Client connects successfully
2. `server_info` appears
3. `detections` events are received

## 8. Update/Release Procedure

For each model or code update:

1. Pull latest code or sync files.
2. Replace `model/final_model.pt`.
3. Reinstall dependencies if `requirements.txt` changed.
4. Restart service:

```bash
sudo systemctl restart ar-app-backend
sudo systemctl status ar-app-backend --no-pager
```

1. Run smoke test with `test_image.py`.

## 9. Rollback Procedure

1. Keep previous known-good model as `model/final_model_prev.pt`.
2. If issues occur:

```bash
cp model/final_model_prev.pt model/final_model.pt
sudo systemctl restart ar-app-backend
```

1. If issue is code-related, checkout previous git tag/commit and restart service.

## 10. Troubleshooting

1. Model load failure:
   - Check path and file permissions of `model/final_model.pt`
   - Confirm weights are valid Ultralytics `.pt`

2. CUDA not used on GPU VM:
   - Verify NVIDIA driver and CUDA runtime
   - Set `DEVICE = "cuda:0"` in `config.py`
   - Confirm PyTorch CUDA visibility in environment

3. Clients cannot connect:
   - Confirm `SERVER_HOST = "0.0.0.0"`
   - Confirm VM firewall/security group allows chosen port
   - Confirm no reverse proxy misconfiguration

4. High latency:
   - Lower `IMAGE_SIZE`
   - Increase `CONFIDENCE_THRESHOLD` to reduce detections
   - Use GPU instance if needed

## 11. Security Recommendations

1. Restrict CORS in production (do not keep `CORS_ORIGINS = "*"` if avoidable).
2. Run behind Nginx + TLS when exposed publicly.
3. Use SSH keys, disable password login where possible.
4. Keep VM and Python dependencies patched.

## 12. Quick Command Checklist

```bash
# VM setup
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git

# app setup
cd ~/ar_app_backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# run
python server.py

# service
sudo systemctl daemon-reload
sudo systemctl enable ar-app-backend
sudo systemctl start ar-app-backend
sudo systemctl status ar-app-backend --no-pager
```
