from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
except ImportError as error:
    raise ImportError(
        "torchvision is required for the v0.6 EfficientNet-B0 teacher. "
        "Install it with: pip install torchvision"
    ) from error

from model import ConvBNAct, C2fLite, SPPF, GoldYOLOLiteFusion, DetectionHead


V06_EXPERIMENT_NAME = "voc8_v06_768_no_p2_regmax16_depth_slim"


class TeacherEfficientNetB0Backbone(nn.Module):
    """Exact v0.6 EfficientNet-B0 feature backbone."""

    def __init__(self, pretrained: bool = False):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        features = efficientnet_b0(weights=weights).features

        self.stage_p2 = nn.Sequential(features[0], features[1], features[2])
        self.stage_p3 = nn.Sequential(features[3])
        self.stage_p4 = nn.Sequential(features[4], features[5])
        self.stage_p5 = nn.Sequential(features[6], features[7], features[8])

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


class TeacherNeck(nn.Module):
    """Exact v0.6 64/80/128 depth-slimmed FPN and PAN neck."""

    def __init__(self):
        super().__init__()

        self.p5_reduce = ConvBNAct(1280, 128, kernel_size=1, stride=1, padding=0)
        self.sppf = SPPF(128, 128, pool_size=5)

        self.fpn4 = C2fLite(128 + 112, 80, num_blocks=2)
        self.p4_reduce = ConvBNAct(80, 64, kernel_size=1, stride=1, padding=0)
        self.fpn3 = C2fLite(64 + 40, 64, num_blocks=3)

        self.pan3_down = ConvBNAct(64, 80, kernel_size=3, stride=2)
        self.pan4 = C2fLite(80 + 80, 80, num_blocks=2)
        self.pan4_down = ConvBNAct(80, 128, kernel_size=3, stride=2)
        self.pan5 = C2fLite(128 + 128, 128, num_blocks=2)

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p5_context = self.sppf(self.p5_reduce(p5))

        p5_up = F.interpolate(p5_context, scale_factor=2, mode="nearest")
        fpn4 = self.fpn4(torch.cat([p4, p5_up], dim=1))

        p4_up = F.interpolate(
            self.p4_reduce(fpn4),
            scale_factor=2,
            mode="nearest",
        )
        fpn3 = self.fpn3(torch.cat([p3, p4_up], dim=1))

        p3_down = self.pan3_down(fpn3)
        pan4 = self.pan4(torch.cat([fpn4, p3_down], dim=1))

        p4_down = self.pan4_down(pan4)
        pan5 = self.pan5(torch.cat([p5_context, p4_down], dim=1))

        return fpn3, pan4, pan5


class YOLOv06Teacher(nn.Module):
    """Frozen v0.6 teacher used only during v0.8 training."""

    def __init__(
        self,
        num_classes: int = 8,
        image_size: int = 768,
        reg_max: int = 16,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.image_size = image_size
        self.reg_max = reg_max
        self.num_bins = reg_max + 1
        self.strides = [8, 16, 32]

        self.backbone = TeacherEfficientNetB0Backbone(pretrained=False)
        self.neck = TeacherNeck()
        self.gold_fusion = GoldYOLOLiteFusion(
            n3_channels=64,
            n4_channels=80,
            n5_channels=128,
            fusion_channels=56,
        )

        self.head_p3 = DetectionHead(
            64,
            num_classes,
            reg_max,
            extra_refine_blocks=3,
        )
        self.head_p4 = DetectionHead(
            80,
            num_classes,
            reg_max,
            extra_refine_blocks=2,
        )
        self.head_p5 = DetectionHead(
            128,
            num_classes,
            reg_max,
            extra_refine_blocks=2,
        )

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

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = True,
    ) -> Dict[str, object] | List[Dict[str, torch.Tensor]]:
        _p2, p3, p4, p5 = self.backbone(x)
        n3_base, n4_base, n5_base = self.neck(p3, p4, p5)
        n3, n4, n5 = self.gold_fusion(n3_base, n4_base, n5_base)
        outputs = self.make_main_outputs(n3, n4, n5)

        if not return_features:
            return outputs

        return {
            "main": outputs,
            "features": {
                "n3": n3,
                "n4": n4,
                "n5": n5,
            },
        }


def load_v06_teacher_checkpoint(
    teacher: YOLOv06Teacher,
    checkpoint: dict,
) -> None:
    experiment_name = checkpoint.get("experiment_name")
    if experiment_name != V06_EXPERIMENT_NAME:
        raise ValueError(
            "Teacher checkpoint does not belong to v0.6.\n"
            f"Expected: {V06_EXPERIMENT_NAME!r}\n"
            f"Found:    {experiment_name!r}"
        )

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Teacher checkpoint does not contain model_state_dict.")

    incompatible = teacher.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith("aux_head_")
    ]

    if missing or unexpected:
        raise RuntimeError(
            "v0.6 teacher state does not match the expected architecture.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected non-auxiliary keys: {unexpected}"
        )

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
