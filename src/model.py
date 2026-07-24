from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
except ImportError as error:
    raise ImportError(
        "torchvision is required for the pretrained MobileNetV3-Large backbone. "
        "Install it with: pip install torchvision"
    ) from error


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DWConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BottleneckLite(nn.Module):
    def __init__(self, channels: int, shortcut: bool = True):
        super().__init__()

        self.conv1 = ConvBNAct(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvBNAct(channels, channels, kernel_size=3, stride=1)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))

        if self.shortcut:
            return x + y

        return y


class C2fLite(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        expansion: float = 0.5,
    ):
        super().__init__()

        hidden_channels = int(out_channels * expansion)

        self.conv_in = ConvBNAct(
            in_channels,
            hidden_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.blocks = nn.ModuleList(
            [BottleneckLite(hidden_channels) for _ in range(num_blocks)]
        )

        self.conv_out = ConvBNAct(
            hidden_channels * (2 + num_blocks),
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        features = list(x.chunk(2, dim=1))

        for block in self.blocks:
            features.append(block(features[-1]))

        x = torch.cat(features, dim=1)
        return self.conv_out(x)


class SPPF(nn.Module):
    """
    SPPF block.

    It gives the deepest feature map more context without changing resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, pool_size: int = 5):
        super().__init__()

        hidden_channels = in_channels // 2

        self.conv1 = ConvBNAct(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=pool_size,
            stride=1,
            padding=pool_size // 2,
        )

        self.conv2 = ConvBNAct(
            hidden_channels * 4,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)

        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)

        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class PretrainedMobileNetV3LargeBackbone(nn.Module):
    """
    Pretrained MobileNetV3-Large feature backbone for the v0.8 student.

    For an input size divisible by 32, it returns:

        P2: stride 4,  shape [B, 24,  H/4,  W/4]
        P3: stride 8,  shape [B, 40,  H/8,  W/8]
        P4: stride 16, shape [B, 112, H/16, W/16]
        P5: stride 32, shape [B, 960, H/32, W/32]

    Torchvision MobileNetV3-Large feature groups:
        features[0..3]  -> P2
        features[4..6]  -> P3
        features[7..12] -> P4
        features[13..16] -> P5
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        features = backbone.features

        self.stage_p2 = nn.Sequential(*features[0:4])
        self.stage_p3 = nn.Sequential(*features[4:7])
        self.stage_p4 = nn.Sequential(*features[7:13])
        self.stage_p5 = nn.Sequential(*features[13:17])

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

        self.register_buffer("mean", mean.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", std.view(1, 3, 1, 1), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = (x - self.mean) / self.std

        p2 = self.stage_p2(x)
        p3 = self.stage_p3(p2)
        p4 = self.stage_p4(p3)
        p5 = self.stage_p5(p4)

        return p2, p3, p4, p5


class TinyNeck(nn.Module):
    """
    Width-slimmed three-scale FPN and PAN neck for v0.8.

    P2/stride-4 remains excluded from the neck and detection heads.
    Depth is retained while widths are reduced to 56, 72 and 112.

    Outputs:
        N3: [B, 56,  H/8,  W/8]
        N4: [B, 72,  H/16, W/16]
        N5: [B, 112, H/32, W/32]
    """

    def __init__(self):
        super().__init__()

        self.p5_reduce = ConvBNAct(960, 112, kernel_size=1, stride=1, padding=0)
        self.sppf = SPPF(in_channels=112, out_channels=112, pool_size=5)

        self.fpn4 = C2fLite(
            in_channels=112 + 112,
            out_channels=72,
            num_blocks=2,
        )

        self.p4_reduce = ConvBNAct(72, 56, kernel_size=1, stride=1, padding=0)

        self.fpn3 = C2fLite(
            in_channels=56 + 40,
            out_channels=56,
            num_blocks=3,
        )

        self.pan3_down = ConvBNAct(56, 72, kernel_size=3, stride=2)

        self.pan4 = C2fLite(
            in_channels=72 + 72,
            out_channels=72,
            num_blocks=2,
        )

        self.pan4_down = ConvBNAct(72, 112, kernel_size=3, stride=2)

        self.pan5 = C2fLite(
            in_channels=112 + 112,
            out_channels=112,
            num_blocks=2,
        )

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p5_small = self.p5_reduce(p5)
        p5_context = self.sppf(p5_small)

        p5_up = F.interpolate(p5_context, scale_factor=2, mode="nearest")
        fpn4 = self.fpn4(torch.cat([p4, p5_up], dim=1))

        fpn4_small = self.p4_reduce(fpn4)
        p4_up = F.interpolate(fpn4_small, scale_factor=2, mode="nearest")
        fpn3 = self.fpn3(torch.cat([p3, p4_up], dim=1))

        p3_down = self.pan3_down(fpn3)
        pan4 = self.pan4(torch.cat([fpn4, p3_down], dim=1))

        p4_down = self.pan4_down(pan4)
        pan5 = self.pan5(torch.cat([p5_context, p4_down], dim=1))

        return fpn3, pan4, pan5


class GoldYOLOLiteFusion(nn.Module):
    """
    Three-scale lightweight gather-distribute fusion for strides 8, 16 and 32.

    N3/N4/N5 are gathered to stride 8, fused, then redistributed to each scale.
    """

    def __init__(
        self,
        n3_channels: int = 56,
        n4_channels: int = 72,
        n5_channels: int = 112,
        fusion_channels: int = 48,
    ):
        super().__init__()

        self.n3_to_fusion = ConvBNAct(
            n3_channels,
            fusion_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.n4_to_fusion = ConvBNAct(
            n4_channels,
            fusion_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.n5_to_fusion = ConvBNAct(
            n5_channels,
            fusion_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.fuse = nn.Sequential(
            ConvBNAct(
                fusion_channels * 3,
                fusion_channels,
                kernel_size=3,
                stride=1,
            ),
            C2fLite(
                fusion_channels,
                fusion_channels,
                num_blocks=2,
            ),
        )

        self.distribute_to_n3 = ConvBNAct(
            fusion_channels,
            n3_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.distribute_to_n4 = ConvBNAct(
            fusion_channels,
            n4_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.distribute_to_n5 = ConvBNAct(
            fusion_channels,
            n5_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.refine_n3 = ConvBNAct(
            n3_channels,
            n3_channels,
            kernel_size=3,
            stride=1,
        )

        # Retained from v0.6: keep the high-resolution N3 refinement but
        # remove the extra low-resolution 3x3 refinement convolutions. The gathered
        # fusion signal is still projected and added to N4 and N5.
        self.refine_n4 = nn.Identity()
        self.refine_n5 = nn.Identity()

    def forward(
        self,
        n3: torch.Tensor,
        n4: torch.Tensor,
        n5: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_size = n3.shape[-2:]

        n3_gather = self.n3_to_fusion(n3)

        n4_gather = F.interpolate(
            self.n4_to_fusion(n4),
            size=target_size,
            mode="nearest",
        )

        n5_gather = F.interpolate(
            self.n5_to_fusion(n5),
            size=target_size,
            mode="nearest",
        )

        gathered = torch.cat(
            [n3_gather, n4_gather, n5_gather],
            dim=1,
        )

        fused = self.fuse(gathered)

        fused_to_n4 = F.interpolate(
            fused,
            size=n4.shape[-2:],
            mode="nearest",
        )

        fused_to_n5 = F.interpolate(
            fused,
            size=n5.shape[-2:],
            mode="nearest",
        )

        n3 = self.refine_n3(n3 + self.distribute_to_n3(fused))
        n4 = self.refine_n4(n4 + self.distribute_to_n4(fused_to_n4))
        n5 = self.refine_n5(n5 + self.distribute_to_n5(fused_to_n5))

        return n3, n4, n5


class DetectionHead(nn.Module):
    """
    Stronger decoupled head.

    For high-resolution features, extra_refine_blocks can add
    lightweight C2fLite refinement before the box/objectness/class branches.

    It predicts:
        box_logits: 4 * (reg_max + 1) DFL channels
        obj_logits: 1 objectness channel
        cls_logits: 8 class channels
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        reg_max: int = 16,
        extra_refine_blocks: int = 0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_bins = reg_max + 1

        box_hidden = max(48, in_channels)
        cls_hidden = max(48, in_channels)

        if extra_refine_blocks > 0:
            self.extra_refine = C2fLite(
                in_channels=in_channels,
                out_channels=in_channels,
                num_blocks=extra_refine_blocks,
            )
        else:
            self.extra_refine = nn.Identity()

        self.box_branch = nn.Sequential(
            ConvBNAct(
                in_channels,
                box_hidden,
                kernel_size=3,
                stride=1,
            ),
            ConvBNAct(
                box_hidden,
                box_hidden,
                kernel_size=3,
                stride=1,
            ),
            nn.Conv2d(
                box_hidden,
                4 * self.num_bins,
                kernel_size=1,
            ),
        )

        self.obj_branch = nn.Sequential(
            DWConvBNAct(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
            ),
            DWConvBNAct(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
            ),
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=1,
            ),
        )

        self.class_branch = nn.Sequential(
            DWConvBNAct(
                in_channels,
                cls_hidden,
                kernel_size=3,
                stride=1,
            ),
            DWConvBNAct(
                cls_hidden,
                cls_hidden,
                kernel_size=3,
                stride=1,
            ),
            nn.Conv2d(
                cls_hidden,
                num_classes,
                kernel_size=1,
            ),
        )

        self._initialize_biases()

    def _initialize_biases(self) -> None:
        obj_conv = self.obj_branch[-1]
        class_conv = self.class_branch[-1]

        if isinstance(obj_conv, nn.Conv2d):
            nn.init.constant_(obj_conv.bias, -4.5)

        if isinstance(class_conv, nn.Conv2d):
            nn.init.constant_(class_conv.bias, -4.5)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.extra_refine(x)

        box_logits = self.box_branch(x)
        obj_logits = self.obj_branch(x)
        cls_logits = self.class_branch(x)

        return {
            "box_logits": box_logits,
            "box_raw": box_logits,
            "obj_logits": obj_logits,
            "cls_logits": cls_logits,
        }


class TinyYOLOAnchorFree(nn.Module):
    """
    v0.8 compact distilled student detector.

    Architecture:
        - pretrained MobileNetV3-Large feature backbone
        - 768 x 768 input and reg_max=16
        - neck widths 56, 72 and 112
        - Gold-YOLO-lite fusion width 48
        - detection-head hidden minimum reduced from 64 to 48
        - three anchor-free heads at strides 8, 16 and 32
        - training-only auxiliary heads
    """

    def __init__(
        self,
        num_classes: int = 8,
        image_size: int = 768,
        reg_max: int = 16,
        pretrained_backbone: bool = True,
        use_auxiliary_heads: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.image_size = image_size
        self.reg_max = reg_max
        self.num_bins = reg_max + 1
        self.strides = [8, 16, 32]
        self.use_auxiliary_heads = use_auxiliary_heads

        self.backbone = PretrainedMobileNetV3LargeBackbone(
            pretrained=pretrained_backbone,
        )

        self.neck = TinyNeck()

        self.gold_fusion = GoldYOLOLiteFusion(
            n3_channels=56,
            n4_channels=72,
            n5_channels=112,
            fusion_channels=48,
        )

        self.head_p3 = DetectionHead(
            in_channels=56,
            num_classes=num_classes,
            reg_max=reg_max,
            extra_refine_blocks=3,
        )

        self.head_p4 = DetectionHead(
            in_channels=72,
            num_classes=num_classes,
            reg_max=reg_max,
            extra_refine_blocks=2,
        )

        self.head_p5 = DetectionHead(
            in_channels=112,
            num_classes=num_classes,
            reg_max=reg_max,
            extra_refine_blocks=2,
        )

        if use_auxiliary_heads:
            self.aux_head_p3 = DetectionHead(
                in_channels=56,
                num_classes=num_classes,
                reg_max=reg_max,
                extra_refine_blocks=2,
            )

            self.aux_head_p4 = DetectionHead(
                in_channels=72,
                num_classes=num_classes,
                reg_max=reg_max,
                extra_refine_blocks=0,
            )

            self.aux_head_p5 = DetectionHead(
                in_channels=112,
                num_classes=num_classes,
                reg_max=reg_max,
                extra_refine_blocks=0,
            )
        else:
            self.aux_head_p3 = None
            self.aux_head_p4 = None
            self.aux_head_p5 = None

        projection = torch.arange(self.num_bins, dtype=torch.float32)
        self.register_buffer("dfl_projection", projection, persistent=False)

    def make_main_outputs(
        self,
        n3: torch.Tensor,
        n4: torch.Tensor,
        n5: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        return [
            {**self.head_p3(n3), "stride": 8},
            {**self.head_p4(n4), "stride": 16},
            {**self.head_p5(n5), "stride": 32},
        ]

    def make_aux_outputs(
        self,
        n3: torch.Tensor,
        n4: torch.Tensor,
        n5: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        if not self.use_auxiliary_heads:
            raise RuntimeError(
                "Auxiliary outputs were requested from an inference-only model."
            )

        assert self.aux_head_p3 is not None
        assert self.aux_head_p4 is not None
        assert self.aux_head_p5 is not None

        return [
            {**self.aux_head_p3(n3), "stride": 8},
            {**self.aux_head_p4(n4), "stride": 16},
            {**self.aux_head_p5(n5), "stride": 32},
        ]

    def forward(
        self,
        x: torch.Tensor,
        decode: bool = False,
        return_aux: bool = False,
        return_features: bool = False,
    ):
        _p2, p3, p4, p5 = self.backbone(x)

        n3_base, n4_base, n5_base = self.neck(
            p3,
            p4,
            p5,
        )

        n3, n4, n5 = self.gold_fusion(
            n3_base,
            n4_base,
            n5_base,
        )

        main_outputs = self.make_main_outputs(
            n3,
            n4,
            n5,
        )

        if decode:
            return self.decode_outputs(main_outputs)

        if return_aux or return_features:
            result = {"main": main_outputs}

            if return_aux:
                if not self.use_auxiliary_heads:
                    raise RuntimeError(
                        "return_aux=True requires use_auxiliary_heads=True."
                    )

                result["aux"] = self.make_aux_outputs(
                    n3_base,
                    n4_base,
                    n5_base,
                )

            if return_features:
                result["features"] = {
                    "n3": n3,
                    "n4": n4,
                    "n5": n5,
                }

            return result

        return main_outputs

    def dfl_decode_distances(
        self,
        box_logits: torch.Tensor,
        stride: int,
    ) -> torch.Tensor:
        batch_size, _, height, width = box_logits.shape

        logits = box_logits.view(
            batch_size,
            4,
            self.num_bins,
            height,
            width,
        )

        probs = torch.softmax(logits, dim=2)

        projection = self.dfl_projection.view(
            1,
            1,
            self.num_bins,
            1,
            1,
        )

        distances = (probs * projection).sum(dim=2)
        distances = distances * stride

        return distances

    def decode_outputs(
        self,
        outputs: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        all_boxes = []
        all_scores = []

        for output in outputs:
            box_logits = output["box_logits"]
            obj_logits = output["obj_logits"]
            cls_logits = output["cls_logits"]
            stride = output["stride"]

            batch_size, _, height, width = box_logits.shape
            device = box_logits.device

            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=device),
                torch.arange(width, device=device),
                indexing="ij",
            )

            x_point = (grid_x.float() + 0.5) * stride
            y_point = (grid_y.float() + 0.5) * stride

            x_point = x_point.unsqueeze(0).expand(batch_size, -1, -1)
            y_point = y_point.unsqueeze(0).expand(batch_size, -1, -1)

            distances = self.dfl_decode_distances(
                box_logits,
                stride=stride,
            )

            left = distances[:, 0, :, :]
            top = distances[:, 1, :, :]
            right = distances[:, 2, :, :]
            bottom = distances[:, 3, :, :]

            x1 = x_point - left
            y1 = y_point - top
            x2 = x_point + right
            y2 = y_point + bottom

            boxes = torch.stack([x1, y1, x2, y2], dim=-1)

            boxes = boxes.reshape(
                batch_size,
                height * width,
                4,
            )

            boxes = boxes.clamp(
                min=0,
                max=self.image_size,
            )

            class_scores = torch.sigmoid(cls_logits)
            objectness = torch.sigmoid(obj_logits)

            scores = class_scores * objectness

            scores = scores.permute(0, 2, 3, 1).reshape(
                batch_size,
                height * width,
                self.num_classes,
            )

            all_boxes.append(boxes)
            all_scores.append(scores)

        return {
            "boxes": torch.cat(all_boxes, dim=1),
            "scores": torch.cat(all_scores, dim=1),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def count_inference_parameters(model: nn.Module) -> int:
    """Counts parameters used by inference, excluding auxiliary heads."""

    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("aux_head_")
    )


def load_inference_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    """Loads a full training state dict or an inference-only state dict."""

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith("aux_head_")
    ]

    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Inference checkpoint does not match the current architecture.\n"
            f"Missing keys: {missing_keys}\n"
            f"Unexpected non-auxiliary keys: {unexpected_keys}"
        )