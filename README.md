# Custom YOLO Object Detection Model

A custom YOLO-style object detector implemented from scratch in PyTorch. The current release is a compact MobileNetV3-Large student trained with knowledge distillation from the strongest balanced EfficientNet-B0 version.

## Current Release

**Version:** v0.8

**Task:** 8-class object detection on a custom VOC8 split

**Framework:** PyTorch

**Deployment input:** 768 x 768
**Deployment parameters:** 4,488,263

## Current Architecture

```text
Input 768 x 768
    -> MobileNetV3-Large pretrained backbone
    -> Three-scale FPN/PAN neck: 56 / 72 / 112 channels
    -> Gold-YOLO-lite gather-distribute fusion: width 48
    -> Anchor-free decoupled heads at strides 8, 16 and 32
    -> DFL box decoding with reg_max = 16
    -> Confidence filtering and NMS
    -> Final detections
```

Each detection head has separate box, objectness and classification branches. The model predicts 8 VOC classes and uses 17 DFL bins for each box side.

### Training-only components

The distilled training graph also contains:

- Auxiliary detection heads
- A frozen v0.6 EfficientNet-B0 teacher
- Three 1 x 1 feature adapters that align student and teacher feature widths
- Feature, classification, objectness and DFL distillation losses

Parameter accounting:

| Component | Parameters |
|---|---:|
| Student model during training | 5,017,502 |
| KD feature adapters | 23,680 |
| Total trainable during distilled training | 5,041,182 |
| Frozen teacher loaded during training | 5,982,131 |
| Deployed inference model | 4,488,263 |

The teacher, KD adapters and auxiliary heads are not included in the deployed inference model.

## Dataset

The project currently uses a custom VOC8 dataset built from PASCAL VOC2007 and VOC2012.

Classes:

1. person
2. car
3. dog
4. cat
5. bus
6. train
7. bicycle
8. aeroplane

| Split | Source | Images |
|---|---|---:|
| Train | VOC2007 train + VOC2007 val + VOC2012 train | 7,777 |
| Validation | VOC2007 test | 3,665 |
| Test | VOC2012 val | 4,189 |

The dataset and trained checkpoints are not included in this repository.

## Version Progression

Accuracy below uses confidence 0.01 on the full 4,189-image test split. FPS is the recorded end-to-end batch-1 laptop result for each version.

| Version | Input | Training params | Inference params | Precision | Recall | mAP50 | mAP50-95 | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0.1 | 960 | 11,683,044 | Not separately recorded | 0.3138 | 0.8666 | 0.7735 | 0.5872 | 30.5641 |
| v0.2 | 960 | 8,456,708 | Not separately recorded | 0.3084 | 0.8566 | 0.7557 | 0.5755 | 28.9610 |
| v0.3 | 768 | 7,932,826 | Not separately recorded | 0.2874 | 0.8632 | 0.7633 | 0.4715 | 34.2784 |
| v0.4 | 768 | 7,952,986 | 7,075,483 | 0.3042 | 0.8567 | 0.7629 | 0.5782 | 37.3324 |
| v0.5 | 768 | 7,011,530 | 6,334,099 | 0.3003 | 0.8554 | 0.7570 | 0.5741 | 36.3372 |
| v0.6 | 768 | 6,659,562 | 5,982,131 | 0.2927 | 0.8565 | 0.7655 | 0.5776 | 37.7646 |
| v0.7 | 768 | 5,863,038 | 5,185,607 | 0.2819 | 0.8491 | 0.7477 | 0.5595 | 38.9152 |
| v0.8 | 768 | 5,017,502 | 4,488,263 | 0.3077 | 0.8443 | 0.7464 | 0.5516 | 37.9738 |

The official progression excludes the experimental v0.7b branch.

## Current v0.8 Results

### Confidence 0.01

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.3077 | 0.8443 | 0.7464 | 0.5516 |

### Confidence 0.10

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.8315 | 0.7081 | 0.6940 | 0.5258 |

### End-to-end speed

Measured on an NVIDIA GeForce RTX 4080 Laptop GPU, batch size 1, FP16, 768 x 768, over the full 4,189-image test split:

| Measurement | Result |
|---|---:|
| End-to-end latency | 26.3339 ms/image |
| End-to-end FPS | 37.9738 |
| Forward and decode | 23.2557 ms/image |
| Transfer | 0.8936 ms/image |
| Postprocessing | 2.1836 ms/image |

Laptop power and thermal state can affect repeated speed measurements.

## Knowledge Distillation

The v0.8 student learns from a frozen v0.6 teacher.

Feature adapters:

| Scale | Student channels | Teacher channels | Adapter parameters |
|---|---:|---:|---:|
| N3, stride 8 | 56 | 64 | 3,584 |
| N4, stride 16 | 72 | 80 | 5,760 |
| N5, stride 32 | 112 | 128 | 14,336 |
| Total |  |  | 23,680 |

KD loss weights:

| Loss | Weight |
|---|---:|
| Feature | 0.20 |
| Classification | 0.40 |
| Objectness | 0.20 |
| DFL | 0.30 |

Schedule:

- Epochs 1 to 2: supervised training only
- Epochs 3 to 17: full KD weight
- Epochs 18 to 20: half KD weight

## Main Files

| File | Purpose |
|---|---|
| `src/model.py` | Defines the v0.8 MobileNetV3-Large student model and decoding path. |
| `src/teacher_model.py` | Defines and loads the frozen v0.6 EfficientNet-B0 teacher. |
| `src/distillation.py` | Implements KD adapters and feature, classification, objectness and DFL distillation losses. |
| `src/loss.py` | Implements the normal detection losses. |
| `src/train.py` | Runs supervised and distilled training and saves checkpoints. |
| `src/eval.py` | Computes precision, recall, mAP and per-size results. |
| `src/predict.py` | Runs detection on images and saves visualized predictions. |
| `src/benchmark_speed.py` | Measures batch-1 end-to-end latency and FPS. |
| `src/export_deployment.py` | Exports an inference-only checkpoint without training-only components. |

## How to Run

Activate the environment:

```bash
conda activate yolo_scratch
```

Train:

```bash
python src/train.py
```

Evaluate:

```bash
python src/eval.py
```

Predict:

```bash
python src/predict.py
```

Benchmark:

```bash
python src/benchmark_speed.py
```

Export the deployment model:

```bash
python src/export_deployment.py
```

## Documentation

- `docs/model_progress.md`
- `docs/current_model_structure_v0_8.md`
- `CHANGELOG.md`

## Next Project Direction

The next planned branch moves the detector into an industrial barcode-localization setting using the BarBeR dataset. The VOC8 v0.8 branch remains the preserved baseline before that dataset transition.
