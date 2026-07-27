# Changelog

## v0.8

- Replaced the EfficientNet-B0 deployment backbone with MobileNetV3-Large.
- Set the three-scale neck widths to 56, 72 and 112.
- Reduced Gold-YOLO-lite fusion width to 48.
- Set the minimum hidden detection-head width to 48.
- Preserved P3, P4 and P5 prediction scales at strides 8, 16 and 32.
- Preserved DFL regression with `reg_max = 16`.
- Added a frozen v0.6 teacher for knowledge distillation.
- Added training-only feature adapters for 56 to 64, 72 to 80 and 112 to 128 channel alignment.
- Added feature, classification, objectness and DFL distillation losses.
- Added inference-only deployment export.
- Added updated evaluation, prediction and end-to-end benchmarking scripts.
- Reached 4,488,263 deployed parameters, 0.7464 mAP50, 0.5516 mAP50-95 and 37.9738 FPS at confidence 0.01.

## v0.7

- Slimmed the late EfficientNet-B0 backbone stages.
- Reduced 112-channel stage repeats from 3 to 2.
- Reduced 192-channel stage repeats from 4 to 3.
- Reached 5,185,607 inference parameters and 38.9152 FPS.

## v0.6

- Reduced selected neck, fusion and head depth from v0.5.
- Preserved neck widths 64, 80 and 128 and fusion width 56.
- Reached 5,982,131 inference parameters.
- Produced the strongest overall optimized accuracy and balance.
- Selected as the teacher for v0.8.

## v0.5

- Reduced three-scale neck and head widths.
- Used widths 64, 80 and 128 with fusion width 56.
- Reduced the deployment model to 6,334,099 parameters.

## v0.4

- Preserved the 768 x 768 three-scale design.
- Restored `reg_max` from 8 to 16.
- Recovered strict localization quality to 0.5782 mAP50-95.
- Recorded separate training and inference parameter counts.

## v0.3

- Reduced input from 960 x 960 to 768 x 768.
- Removed the stride-4 P2 path and detection head.
- Switched from four prediction scales to three.
- Reduced `reg_max` from 16 to 8.

## v0.2

- Reduced four-scale neck widths from 64/96/128/192 to 48/72/96/144.
- Reduced Gold-YOLO-lite fusion width from 96 to 72.
- Reduced training parameters by about 27.6 percent from v0.1.

## v0.1

- Initial custom YOLO-style detector.
- EfficientNet-B0 backbone.
- Four-scale FPN/PAN and Gold-YOLO-lite fusion.
- Anchor-free decoupled heads with objectness and DFL regression.
- Training-only auxiliary heads.
