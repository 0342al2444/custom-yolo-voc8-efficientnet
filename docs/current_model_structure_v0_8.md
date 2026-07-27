# Current Model Structure: v0.8

## Inference graph

```text
Input image: 768 x 768
    |
    v
MobileNetV3-Large pretrained backbone
    |
    +-- stride 8 feature
    +-- stride 16 feature
    +-- stride 32 feature
    |
    v
Three-scale FPN/PAN neck
    |
    +-- N3: 56 channels, 96 x 96
    +-- N4: 72 channels, 48 x 48
    +-- N5: 112 channels, 24 x 24
    |
    v
Gold-YOLO-lite gather-distribute fusion
Fusion width: 48
    |
    v
Main decoupled detection heads
    |
    +-- P3, stride 8
    +-- P4, stride 16
    +-- P5, stride 32
    |
    v
DFL decoding, confidence filtering and NMS
    |
    v
Final detections
```

The deployment model has 4,488,263 parameters.

## Detection heads

| Head | Input | Extra refinement | Used in inference |
|---|---|---:|---|
| Main P3 | 56 x 96 x 96 | 3 C2f blocks | Yes |
| Main P4 | 72 x 48 x 48 | 2 C2f blocks | Yes |
| Main P5 | 112 x 24 x 24 | 2 C2f blocks | Yes |
| Auxiliary P3 | 56 x 96 x 96 | 2 C2f blocks | No |
| Auxiliary P4 | 72 x 48 x 48 | None | No |
| Auxiliary P5 | 112 x 24 x 24 | None | No |

Each head has three branches:

- Box branch: two 3 x 3 Conv-BN-SiLU layers, then a 1 x 1 output convolution
- Objectness branch: two depthwise-separable blocks, then one output channel
- Classification branch: two depthwise-separable blocks, then eight output channels

Minimum hidden width: 48.

## Prediction layout

`reg_max = 16`, so each side uses 17 DFL bins.

| Scale | Stride | Grid | Locations | Raw values per location |
|---|---:|---:|---:|---:|
| P3 | 8 | 96 x 96 | 9,216 | 68 box + 1 objectness + 8 class |
| P4 | 16 | 48 x 48 | 2,304 | 68 box + 1 objectness + 8 class |
| P5 | 32 | 24 x 24 | 576 | 68 box + 1 objectness + 8 class |
| Total |  |  | 12,096 |  |

DFL decoding:

1. Reshape the 68 box channels into four sides with 17 bins each.
2. Apply softmax over the bins.
3. Calculate the expected distance for each side.
4. Multiply the distance by the scale stride.
5. Convert left, top, right and bottom distances into box coordinates.

## Training graph

```text
                         +-----------------------------+
                         | Frozen v0.6 teacher         |
Input 768 x 768 -------->| EfficientNet-B0             |
    |                    | Features: 64 / 80 / 128     |
    |                    +--------------+--------------+
    |                                   |
    v                                   | teacher features and logits
MobileNetV3-Large student               |
    |                                   |
    v                                   |
Student neck: 56 / 72 / 112             |
    |                                   |
    +-- auxiliary heads                 |
    |                                   |
    +-- 1 x 1 KD adapters --------------+
    |      56 -> 64
    |      72 -> 80
    |      112 -> 128
    |
    v
Gold-YOLO-lite fusion and main heads
    |
    +-- supervised detection losses
    +-- feature KD loss
    +-- classification KD loss
    +-- objectness KD loss
    +-- DFL KD loss
    |
    v
Total training loss
```

## KD adapter parameters

The adapters use 1 x 1 convolutions without bias.

| Scale | Mapping | Parameters |
|---|---:|---:|
| N3 | 56 to 64 | 3,584 |
| N4 | 72 to 80 | 5,760 |
| N5 | 112 to 128 | 14,336 |
| Total |  | 23,680 |

## Training and inference differences

| Training | Inference |
|---|---|
| Main heads and auxiliary heads | Main heads only |
| Frozen v0.6 teacher loaded | No teacher |
| KD feature adapters | No adapters |
| Supervised and KD losses | No loss modules |
| Returns main, auxiliary and feature tensors | Returns main logits or decoded detections |
| 5,041,182 trainable parameters including adapters | 4,488,263 deployed parameters |

The frozen teacher contributes to training memory and computation, but it receives no gradient and is not deployed.
