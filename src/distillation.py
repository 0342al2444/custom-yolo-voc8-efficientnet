from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionDistillationLoss(nn.Module):
    """
    Detection-specific knowledge distillation for aligned YOLO feature grids.

    Teacher and student use the same image size, strides, classes and DFL bins,
    so their prediction tensors align directly. Small 1x1 adapters align the
    narrower student feature channels with the v0.6 teacher feature channels.
    """

    def __init__(
        self,
        student_channels: tuple[int, int, int] = (56, 72, 112),
        teacher_channels: tuple[int, int, int] = (64, 80, 128),
        num_classes: int = 8,
        reg_max: int = 16,
        temperature: float = 2.0,
        teacher_conf_threshold: float = 0.05,
        feature_weight: float = 0.20,
        classification_weight: float = 0.40,
        objectness_weight: float = 0.20,
        dfl_weight: float = 0.30,
    ):
        super().__init__()

        if len(student_channels) != 3 or len(teacher_channels) != 3:
            raise ValueError("Exactly three student and teacher feature widths are required.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_bins = reg_max + 1
        self.temperature = temperature
        self.teacher_conf_threshold = teacher_conf_threshold
        self.feature_weight = feature_weight
        self.classification_weight = classification_weight
        self.objectness_weight = objectness_weight
        self.dfl_weight = dfl_weight

        self.feature_adapters = nn.ModuleDict(
            {
                "n3": nn.Conv2d(student_channels[0], teacher_channels[0], 1, bias=False),
                "n4": nn.Conv2d(student_channels[1], teacher_channels[1], 1, bias=False),
                "n5": nn.Conv2d(student_channels[2], teacher_channels[2], 1, bias=False),
            }
        )

    @staticmethod
    def _weighted_mean(
        values: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        while weights.ndim < values.ndim:
            weights = weights.unsqueeze(1)

        weighted = values * weights
        denominator = weights.expand_as(values).sum().clamp(min=1.0)
        return weighted.sum() / denominator

    @staticmethod
    def _ground_truth_mask(
        targets: List[torch.Tensor],
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.zeros(
            (len(targets), 1, height, width),
            dtype=torch.float32,
            device=device,
        )

        for batch_index, target in enumerate(targets):
            if target.numel() == 0:
                continue

            target = target.to(device=device, dtype=torch.float32)
            x_center = target[:, 1] * width
            y_center = target[:, 2] * height
            box_width = target[:, 3] * width
            box_height = target[:, 4] * height

            x1 = torch.floor(x_center - box_width / 2).long().clamp(0, width - 1)
            y1 = torch.floor(y_center - box_height / 2).long().clamp(0, height - 1)
            x2 = torch.ceil(x_center + box_width / 2).long().clamp(0, width - 1)
            y2 = torch.ceil(y_center + box_height / 2).long().clamp(0, height - 1)

            for left, top, right, bottom in zip(x1, y1, x2, y2):
                mask[
                    batch_index,
                    0,
                    int(top.item()) : int(bottom.item()) + 1,
                    int(left.item()) : int(right.item()) + 1,
                ] = 1.0

        return mask

    def _focus_weights(
        self,
        teacher_output: Dict[str, torch.Tensor],
        targets: List[torch.Tensor],
    ) -> torch.Tensor:
        teacher_obj = torch.sigmoid(teacher_output["obj_logits"].detach().float())
        teacher_cls = torch.sigmoid(teacher_output["cls_logits"].detach().float())
        teacher_quality = teacher_obj * teacher_cls.amax(dim=1, keepdim=True)

        batch_size, _, height, width = teacher_quality.shape
        if batch_size != len(targets):
            raise ValueError("Target batch size does not match teacher outputs.")

        gt_mask = self._ground_truth_mask(
            targets=targets,
            height=height,
            width=width,
            device=teacher_quality.device,
        )

        teacher_mask = (teacher_quality >= self.teacher_conf_threshold).float()
        focus = torch.maximum(gt_mask, teacher_mask)

        # Ensure every image contributes even when the teacher has no location
        # above the confidence threshold and the image contains no GT box.
        flat_focus = focus.flatten(1)
        flat_quality = teacher_quality.flatten(1)
        empty_images = flat_focus.sum(dim=1) == 0

        if empty_images.any():
            topk = min(32, flat_quality.shape[1])
            top_indices = flat_quality[empty_images].topk(topk, dim=1).indices
            replacement = torch.zeros_like(flat_quality[empty_images])
            replacement.scatter_(1, top_indices, 1.0)
            flat_focus[empty_images] = replacement
            focus = flat_focus.view_as(focus)

        return focus * (1.0 + teacher_quality)

    def _feature_loss(
        self,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
        adapter: nn.Module,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        adapted_student = adapter(student_feature)

        if adapted_student.shape != teacher_feature.shape:
            raise ValueError(
                "Feature shapes do not align after adaptation: "
                f"student {tuple(adapted_student.shape)} vs "
                f"teacher {tuple(teacher_feature.shape)}"
            )

        student_norm = F.normalize(adapted_student.float(), dim=1)
        teacher_norm = F.normalize(teacher_feature.detach().float(), dim=1)
        per_location = (student_norm - teacher_norm).pow(2).mean(dim=1, keepdim=True)
        return self._weighted_mean(per_location, weights)

    def _classification_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        temperature = self.temperature
        teacher_targets = torch.sigmoid(teacher_logits.detach().float() / temperature)
        per_element = F.binary_cross_entropy_with_logits(
            student_logits.float() / temperature,
            teacher_targets,
            reduction="none",
        )
        per_location = per_element.mean(dim=1, keepdim=True)
        return self._weighted_mean(per_location, weights) * (temperature**2)

    def _objectness_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        temperature = self.temperature
        teacher_targets = torch.sigmoid(teacher_logits.detach().float() / temperature)
        per_location = F.binary_cross_entropy_with_logits(
            student_logits.float() / temperature,
            teacher_targets,
            reduction="none",
        )
        return self._weighted_mean(per_location, weights) * (temperature**2)

    def _dfl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels, height, width = student_logits.shape
        expected_channels = 4 * self.num_bins

        if channels != expected_channels or teacher_logits.shape != student_logits.shape:
            raise ValueError(
                "Teacher and student DFL tensors must match and contain "
                f"{expected_channels} channels."
            )

        temperature = self.temperature
        student = student_logits.float().view(
            batch_size,
            4,
            self.num_bins,
            height,
            width,
        )
        teacher = teacher_logits.detach().float().view(
            batch_size,
            4,
            self.num_bins,
            height,
            width,
        )

        student_log_prob = F.log_softmax(student / temperature, dim=2)
        teacher_prob = F.softmax(teacher / temperature, dim=2)
        per_bin = F.kl_div(student_log_prob, teacher_prob, reduction="none")
        per_location = per_bin.sum(dim=2).mean(dim=1, keepdim=True)
        return self._weighted_mean(per_location, weights) * (temperature**2)

    def forward(
        self,
        student_result: Dict[str, object],
        teacher_result: Dict[str, object],
        targets: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        student_outputs = student_result["main"]
        teacher_outputs = teacher_result["main"]
        student_features = student_result["features"]
        teacher_features = teacher_result["features"]

        if len(student_outputs) != 3 or len(teacher_outputs) != 3:
            raise ValueError("Distillation requires exactly three aligned outputs.")

        feature_losses = []
        classification_losses = []
        objectness_losses = []
        dfl_losses = []

        for scale_index, feature_name in enumerate(("n3", "n4", "n5")):
            student_output = student_outputs[scale_index]
            teacher_output = teacher_outputs[scale_index]

            if student_output["stride"] != teacher_output["stride"]:
                raise ValueError("Teacher and student strides do not align.")

            weights = self._focus_weights(teacher_output, targets)

            feature_losses.append(
                self._feature_loss(
                    student_feature=student_features[feature_name],
                    teacher_feature=teacher_features[feature_name],
                    adapter=self.feature_adapters[feature_name],
                    weights=weights,
                )
            )
            classification_losses.append(
                self._classification_loss(
                    student_output["cls_logits"],
                    teacher_output["cls_logits"],
                    weights,
                )
            )
            objectness_losses.append(
                self._objectness_loss(
                    student_output["obj_logits"],
                    teacher_output["obj_logits"],
                    weights,
                )
            )
            dfl_losses.append(
                self._dfl_loss(
                    student_output["box_logits"],
                    teacher_output["box_logits"],
                    weights,
                )
            )

        feature_loss = torch.stack(feature_losses).mean()
        classification_loss = torch.stack(classification_losses).mean()
        objectness_loss = torch.stack(objectness_losses).mean()
        dfl_loss = torch.stack(dfl_losses).mean()

        total = (
            self.feature_weight * feature_loss
            + self.classification_weight * classification_loss
            + self.objectness_weight * objectness_loss
            + self.dfl_weight * dfl_loss
        )

        return {
            "loss": total,
            "feature_loss": feature_loss,
            "classification_loss": classification_loss,
            "objectness_loss": objectness_loss,
            "dfl_loss": dfl_loss,
        }
