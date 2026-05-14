# SOP: Dataset Organization and Versioning

## 1. Goal

Keep image datasets organized, traceable, and reproducible so model training and deployment are reliable over time.

## 2. Directory Standard

Use one root per dataset version.

```text
datasets/
  project_name/
    v1/
      images/
        train/
        val/
      labels/
        train/
        val/
      data.yaml
      classes.txt
      README.md
    v2/
      ...
```

Notes:

1. Keep raw source images separate from training-ready datasets.
2. Do not mix raw files and processed files in the same folder.

## 3. Naming Conventions

## 3.1 Dataset versions

Use monotonic versions:

1. v1, v2, v3 for major updates
2. Optional date suffix for tracking: v3_2026_05_05

## 3.2 Files

1. Use stable, lowercase filenames.
2. Avoid spaces and special characters.
3. Use underscore separators.

Example:

```text
site_a_cam2_000123.jpg
site_a_cam2_000123.txt
```

## 4. Required Metadata per Version

Each version folder should include:

1. README.md with data source summary and labeling policy version
2. classes.txt with class list and fixed order
3. data.yaml for training
4. Changelog section inside README.md

Suggested README fields:

1. Created date
2. Data sources
3. Number of images by split
4. Number of annotations by class
5. Labeling team/tool used
6. Known limitations

## 5. Train/Validation Split Rules

1. Keep splits mutually exclusive by scene/source.
2. Prevent leakage from near-duplicate frames.
3. Keep class distribution reasonably balanced in validation.
4. Maintain a fixed validation set when comparing experiments.

## 6. Data Quality Gates

Before accepting a dataset version, pass these checks:

1. Image-label 1:1 mapping
2. No invalid class ids
3. No out-of-range YOLO coordinates
4. No corrupted images
5. Label consistency review completed

## 7. Dataset Release Process

## 7.1 Create release candidate

1. Copy prepared data into new version folder.
2. Freeze classes.txt and data.yaml.
3. Run QA checks and spot audits.

## 7.2 Approve and freeze

1. Mark version as approved in README.md.
2. Record who approved and date.
3. Do not edit files after freeze; create next version for changes.

## 8. Linking Dataset to Model Artifacts

For each training run, record:

1. Dataset version used
2. Training command and hyperparameters
3. Final model artifact location
4. Evaluation metrics

Suggested model artifact layout:

```text
artifacts/
  model_v1/
    best.pt
    train_config.txt
    metrics.json
    dataset_version.txt
```

## 9. Backup and Storage

1. Store frozen dataset versions in durable storage (cloud bucket or NAS).
2. Keep at least one off-machine backup.
3. Use checksums for large transfers when possible.

## 10. How This Fits Your Current Repo

For this project:

1. Final deployed weight should be copied to model/final_model.pt.
2. Training and deployment flow is documented in:
   - SOP_LOCAL_MAC_TRAIN_AND_DEPLOY.md
   - SOP_TRAIN_AND_DEPLOY_VM.md

Use this organization SOP before running either training SOP.
