# SOP: Image Labeling for YOLO Training

## 1. Goal

Create high-quality, consistent bounding box labels for object detection so training results are stable and reproducible.

This workflow is designed for YOLO format labels used by Ultralytics.

## 2. Recommended Labeling Tools

Pick one tool and stay consistent for the whole dataset.

1. CVAT (best for team projects and QA workflows)
2. Roboflow Annotate (easy web workflow, export to YOLO)
3. Label Studio (flexible, self-hostable)
4. LabelImg (lightweight local tool)

Requirement:
Export labels in YOLO detection format, where each image has a matching .txt annotation file.

## 3. Class Definition Rules

Before labeling, freeze the class list.

1. Create one source-of-truth class list file.
2. Keep class order fixed; index values must never change mid-project.
3. Use simple lowercase names with underscores.

Example:

```text
0 person
1 helmet
2 vest
```

If class order changes after partial labeling, old labels become invalid for training.

## 4. Labeling Policy (Must Follow)

## 4.1 Box placement

1. Draw tight boxes around the visible object.
2. Include only the target object, not extra background.
3. For partially visible objects, label only visible area.
4. Do not box reflections/shadows unless they are valid targets.

## 4.2 Small/unclear objects

1. If object is too tiny or ambiguous, skip it.
2. If object is clear but partly occluded, still label it.
3. Keep the same decision rule for all labelers.

## 4.3 Truncation and occlusion

1. Truncated by image border: label visible part.
2. Occluded by another object: label visible part if class is still certain.
3. If class is uncertain, do not label.

## 4.4 One object, one box

1. Never assign multiple boxes to the same object instance.
2. Do not merge nearby objects into one box.

## 5. File and Naming Standards

1. Image and label base names must match exactly.
2. Supported image extensions: .jpg, .jpeg, .png.
3. Labels are .txt with one line per object:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized to range 0 to 1.

Example:

```text
1 0.512 0.438 0.210 0.365
```

## 6. Dataset Folder Format (YOLO)

Use this structure before training:

```text
datasets/my_data/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

For each image:

1. images/train/img_001.jpg has labels/train/img_001.txt
2. images/val/img_010.jpg has labels/val/img_010.txt

## 7. Labeling Workflow

## 7.1 Collect and clean images

1. Remove exact duplicates and corrupt files.
2. Ensure class diversity (angle, distance, lighting, background).
3. Keep realistic production conditions.

## 7.2 Label a pilot batch first

1. Label 50 to 100 images.
2. Review for policy consistency.
3. Fix policy ambiguities before full-scale labeling.

## 7.3 Full labeling

1. Label all images using frozen class list and policy.
2. Keep a changelog of policy updates.
3. Run periodic QA every 200 to 500 images.

## 7.4 QA pass

Checklist:

1. Missing obvious objects
2. Wrong class assignments
3. Loose or oversized boxes
4. Duplicate boxes on same object
5. Empty label files for positive images
6. Label exists but image missing (or vice versa)

## 8. Validation Scripts (Recommended)

Run quick checks before training:

1. Count images and labels in each split.
2. Verify 1:1 image-label filename mapping.
3. Validate class ids are in expected range.
4. Validate YOLO coordinates are between 0 and 1.

## 9. Split Strategy

Recommended split:

1. Train 70 to 85 percent
2. Validation 15 to 30 percent

Important:
Do not put near-duplicate frames in both train and validation.

If data comes from videos, split by scene/video source, not by random frame.

## 10. Common Mistakes to Avoid

1. Inconsistent class definitions between labelers
2. Changing class order mid-project
3. Very loose boxes with large background area
4. Labeling uncertain objects as certain classes
5. Training without a QA pass

## 11. Handoff to Training

Before training, confirm:

1. Dataset folder follows YOLO structure
2. data.yaml names and nc match class list
3. Spot-check at least 100 random images with labels
4. Validation set is representative of production use cases

After this, train using your local-Mac SOP:

- SOP_LOCAL_MAC_TRAIN_AND_DEPLOY.md
