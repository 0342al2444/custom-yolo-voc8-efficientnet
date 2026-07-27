# Custom YOLO Model Progress

This document records the official model sequence from v0.1 through v0.8. The experimental v0.7b branch is intentionally excluded from the main progression.

## Evaluation setup

- Test split: VOC2012 validation, 4,189 images
- Classes: person, car, dog, cat, bus, train, bicycle and aeroplane
- Accuracy table: confidence threshold 0.01
- Speed: batch 1, FP16, end-to-end laptop measurements
- v0.1 and v0.2 input: 960 x 960
- v0.3 onward input: 768 x 768

## Results

| Version | Training params | Inference params | Precision | Recall | mAP50 | mAP50-95 | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| v0.1 | 11,683,044 | Not separately recorded | 0.3138 | 0.8666 | 0.7735 | 0.5872 | 30.5641 |
| v0.2 | 8,456,708 | Not separately recorded | 0.3084 | 0.8566 | 0.7557 | 0.5755 | 28.9610 |
| v0.3 | 7,932,826 | Not separately recorded | 0.2874 | 0.8632 | 0.7633 | 0.4715 | 34.2784 |
| v0.4 | 7,952,986 | 7,075,483 | 0.3042 | 0.8567 | 0.7629 | 0.5782 | 37.3324 |
| v0.5 | 7,011,530 | 6,334,099 | 0.3003 | 0.8554 | 0.7570 | 0.5741 | 36.3372 |
| v0.6 | 6,659,562 | 5,982,131 | 0.2927 | 0.8565 | 0.7655 | 0.5776 | 37.7646 |
| v0.7 | 5,863,038 | 5,185,607 | 0.2819 | 0.8491 | 0.7477 | 0.5595 | 38.9152 |
| v0.8 | 5,017,502 | 4,488,263 | 0.3077 | 0.8443 | 0.7464 | 0.5516 | 37.9738 |

## v0.1: original four-scale model

Main design:

- EfficientNet-B0 pretrained backbone
- 960 x 960 input
- Four FPN/PAN outputs at strides 4, 8, 16 and 32
- Neck widths 64, 96, 128 and 192
- Gold-YOLO-lite fusion width 96
- Anchor-free decoupled detection heads
- Objectness, classification and DFL regression branches
- `reg_max = 16`, producing 17 bins per box side
- Training-only auxiliary heads

Outcome:

- Strongest strict localization result in the official sequence: 0.5872 mAP50-95
- Largest model and slowest early architecture
- P2 preserved small features but created a high computational cost at 960 x 960

## v0.2: width-reduced four-scale model

Changes from v0.1:

- Retained EfficientNet-B0, 960 x 960 and all four detection scales
- Reduced neck widths from 64/96/128/192 to 48/72/96/144
- Reduced Gold fusion width from 96 to 72
- Reduced head and auxiliary-branch parameters through narrower inputs

Outcome:

- Training parameters fell by about 27.6 percent
- Accuracy decreased moderately
- End-to-end speed did not improve in the recorded laptop run, showing that fewer parameters alone did not remove the expensive high-resolution P2 computation

## v0.3: three-scale and lower-resolution experiment

Changes from v0.2:

- Input reduced from 960 to 768
- Removed the stride-4 P2 neck path and P2 detection head
- Used only P3, P4 and P5 at strides 8, 16 and 32
- Reduced `reg_max` from 16 to 8

Outcome:

- Speed improved to 34.2784 FPS
- mAP50 remained competitive at 0.7633
- mAP50-95 dropped sharply to 0.4715
- The result showed that the smaller DFL range damaged precise localization more than expected

## v0.4: restored DFL localization capacity

Changes from v0.3:

- Kept the efficient 768 x 768, three-scale design
- Restored `reg_max` from 8 to 16
- Returned to 68 box-regression channels per prediction point
- Standardized the separation between training parameters and active inference parameters

Outcome:

- mAP50-95 recovered from 0.4715 to 0.5782
- End-to-end speed improved to 37.3324 FPS
- Demonstrated that the P2 head could remain removed while recovering most of the strict localization quality

## v0.5: width-slimmed three-scale model

Changes from v0.4:

- Reduced the three neck and head widths
- Used the compact 64/80/128 feature widths
- Reduced Gold-YOLO-lite fusion width to 56
- Preserved the three-scale prediction design and `reg_max = 16`

Outcome:

- Inference parameters fell to 6,334,099
- mAP50-95 remained close to v0.4 at 0.5741
- Recorded FPS was 36.3372 in the same-session comparison
- Width slimming reduced size effectively but did not guarantee a speed increase on the laptop GPU

## v0.6: depth-slimmed balanced model

Changes from v0.5:

- Retained EfficientNet-B0 and widths 64/80/128 with fusion width 56
- Reduced selected C2f and refinement depth
- Preserved the parts most important to three-scale feature fusion and localization

Outcome:

- Inference parameters fell to 5,982,131
- Reached 0.7655 mAP50 and 0.5776 mAP50-95
- Reached 37.7646 FPS in the post-restart two-run average
- Became the strongest overall accuracy, size and speed balance in the EfficientNet sequence
- Selected as the frozen teacher for v0.8

## v0.7: late-backbone slimming

Changes from v0.6:

- Reduced the EfficientNet 112-channel stage repeats from 3 to 2
- Reduced the EfficientNet 192-channel stage repeats from 4 to 3
- Kept the three-scale neck, fusion and head design

Outcome:

- Inference parameters fell to 5,185,607
- Speed increased to 38.9152 FPS
- mAP50-95 decreased to 0.5595
- Showed that late backbone depth still contributed meaningful detection quality

## v0.8: distilled MobileNetV3-Large student

Changes from v0.7:

- Replaced EfficientNet-B0 with a pretrained MobileNetV3-Large backbone
- Used neck widths 56/72/112
- Reduced Gold-YOLO-lite fusion width to 48
- Used a minimum hidden head width of 48
- Preserved strides 8, 16 and 32
- Preserved `reg_max = 16`
- Added knowledge distillation from the frozen v0.6 teacher
- Added three training-only 1 x 1 feature adapters
- Added feature, classification, objectness and DFL KD terms

Parameter accounting:

| Component | Parameters |
|---|---:|
| Student model during training | 5,017,502 |
| KD feature adapters | 23,680 |
| Total trainable during distilled training | 5,041,182 |
| Frozen teacher | 5,982,131 |
| Deployed model | 4,488,263 |

KD configuration:

- Feature weight: 0.20
- Classification weight: 0.40
- Objectness weight: 0.20
- DFL weight: 0.30
- Epochs 1 to 2: no KD
- Epochs 3 to 17: full KD
- Epochs 18 to 20: half KD

Outcome:

- Smallest official deployed model at 4,488,263 parameters
- 61.58 percent fewer deployment parameters than the v0.1 training-parameter baseline
- 0.7464 mAP50 and 0.5516 mAP50-95
- 37.9738 FPS
- Smaller than v0.7, but not faster in the stable benchmark
- Accuracy remained below v0.6, showing that the backbone replacement created a larger capacity tradeoff than depth slimming alone

## Main conclusions

1. Removing P2 and reducing input resolution produced the largest early speed improvement.
2. Reducing `reg_max` to 8 damaged strict localization and was reversed.
3. Width slimming reduced parameters more consistently than it improved measured GPU speed.
4. v0.6 is the best balanced custom version.
5. v0.7 is the fastest official EfficientNet version.
6. v0.8 is the smallest deployed version.
7. Knowledge distillation helped preserve useful accuracy after the MobileNetV3-Large backbone change, but did not fully recover v0.6 performance.
