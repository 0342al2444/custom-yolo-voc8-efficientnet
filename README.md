# Custom YOLO Object Detection Model

This project implements a custom YOLO-style object detector from scratch in PyTorch.

The model uses an EfficientNet-B0 pretrained backbone, FPN/PAN neck, Gold-YOLO-lite feature fusion, anchor-free detection heads, DFL box regression, and auxiliary training heads.

## Version

Current release: v0.1.0

This is the first working research version. Future versions may reduce model size and improve small-object detection.

## Model Summary

- Model type: Custom YOLO-style object detector
- Backbone: EfficientNet-B0 pretrained on ImageNet
- Input size: 960 x 960
- Classes: 8
- Parameters: 11.68M
- Active inference parameters: 9.97M
- Detection scales: stride 4, 8, 16, 32

## Dataset

This project uses a custom VOC8 dataset built from PASCAL VOC2007 and VOC2012.

Classes:

1. person
2. car
3. dog
4. cat
5. bus
6. train
7. bicycle
8. aeroplane

Dataset split:

| Split | Source | Images |
|---|---|---:|
| Train | VOC2007 train + VOC2007 val + VOC2012 train | 7,777 |
| Val | VOC2007 test | 3,665 |
| Test | VOC2012 val | 4,189 |

The dataset is not included in this repository.

## Architecture

High-level structure:

    Input image
      -> EfficientNet-B0 backbone
      -> FPN top-down neck
      -> PAN bottom-up neck
      -> Gold-YOLO-lite feature fusion
      -> Four anchor-free detection heads
      -> DFL box decoding
      -> NMS
      -> Final detections

More details are available in:

- docs/current_model_structure_english.txt
- docs/current_model_structure_chinese.txt

## Results

### Custom model on VOC2012 val test split

| Confidence | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 0.01 | 0.3138 | 0.8666 | 0.7735 | 0.5872 |
| 0.10 | 0.8359 | 0.7395 | 0.7179 | 0.5580 |

### Comparison with YOLOv8n

On the VOC2012 val test split:

| Model | Confidence | mAP50 | mAP50-95 |
|---|---:|---:|---:|
| Custom EfficientNet-B0 YOLO | 0.01 | 0.7735 | 0.5872 |
| YOLOv8n | 0.01 | 0.748 | 0.431 |
| Custom EfficientNet-B0 YOLO | 0.10 | 0.7179 | 0.5580 |
| YOLOv8n | 0.10 | 0.713 | 0.416 |

The custom model is larger than YOLOv8n, but achieves stronger strict localization quality based on mAP50-95.

## Speed Benchmark

Hardware:

- NVIDIA GeForce RTX 4080 Laptop GPU

Input:

- 960 x 960
- Batch size: 1
- Test split: VOC2012 val, 4,189 images
- Confidence threshold: 0.10

| Setup | Time | FPS |
|---|---:|---:|
| Forward only | 22.72 ms/image | 44.00 FPS |
| Forward + box decode | 25.34 ms/image | 39.47 FPS |
| End-to-end with NMS | 32.72 ms/image | 30.56 FPS |

End-to-end timing includes tensor transfer, forward pass, box decoding, confidence filtering, and NMS.

## How to Run

Activate environment:

    conda activate yolo_scratch

Train:

    python src/train.py

Evaluate:

    python src/eval.py

Benchmark speed:

    python src/benchmark_speed.py --split test --batch-size 1 --max-images 0 --conf 0.10

## Current Limitations

- Small-object detection is still weak.
- Medium-object detection is weaker than large-object detection.
- The model is larger than YOLOv8n.
- Dataset and model weights are not included directly in the repository.

## Future Work

- Reduce model size
- Improve small-object detection
- Try lighter backbones
- Add cleaner config files
- Add more prediction examples
