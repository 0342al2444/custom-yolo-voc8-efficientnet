from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOAnchorFreeLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        image_size: int = 768,
        reg_max: int = 16,
        box_loss_weight: float = 5.0,
        cls_loss_weight: float = 1.0,
        obj_loss_weight: float = 1.0,
        dfl_loss_weight: float = 1.5,
        topk: int = 10,
        alpha: float = 0.5,
        beta: float = 6.0,
        center_radius: float = 2.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        objectness_quality_power: float = 1.5,
        min_quality_target: float = 0.05,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.image_size = image_size
        self.reg_max = reg_max
        self.num_bins = reg_max + 1

        self.box_loss_weight = box_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.obj_loss_weight = obj_loss_weight
        self.dfl_loss_weight = dfl_loss_weight

        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.center_radius = center_radius

        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.objectness_quality_power = objectness_quality_power
        self.min_quality_target = min_quality_target

        projection = torch.arange(self.num_bins, dtype=torch.float32)
        self.register_buffer("dfl_projection", projection, persistent=False)

    def focal_bce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        prob = torch.sigmoid(logits)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        p_t = targets * prob + (1.0 - targets) * (1.0 - prob)
        focal_weight = (1.0 - p_t).pow(self.focal_gamma)

        alpha_factor = targets * self.focal_alpha + (1.0 - targets) * (
            1.0 - self.focal_alpha
        )

        return bce * focal_weight * alpha_factor

    def targets_to_xyxy(
        self,
        targets: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if targets.numel() == 0:
            empty_classes = torch.zeros((0,), dtype=torch.long, device=device)
            empty_boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            return empty_classes, empty_boxes

        targets = targets.to(device)

        class_ids = targets[:, 0].long()

        x_center = targets[:, 1] * self.image_size
        y_center = targets[:, 2] * self.image_size
        width = targets[:, 3] * self.image_size
        height = targets[:, 4] * self.image_size

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        boxes = boxes.clamp(min=0, max=self.image_size)

        return class_ids, boxes

    def box_area(self, boxes: torch.Tensor) -> torch.Tensor:
        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        return widths * heights

    def box_iou_matrix(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return torch.zeros(
                (boxes1.shape[0], boxes2.shape[0]),
                device=boxes1.device,
            )

        area1 = self.box_area(boxes1)
        area2 = self.box_area(boxes2)

        inter_x1 = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        inter_y1 = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        inter_x2 = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        inter_y2 = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        union = area1[:, None] + area2[None, :] - inter_area

        return inter_area / (union + eps)

    def bbox_ciou(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(dim=1)
        tgt_x1, tgt_y1, tgt_x2, tgt_y2 = target_boxes.unbind(dim=1)

        inter_x1 = torch.maximum(pred_x1, tgt_x1)
        inter_y1 = torch.maximum(pred_y1, tgt_y1)
        inter_x2 = torch.minimum(pred_x2, tgt_x2)
        inter_y2 = torch.minimum(pred_y2, tgt_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (
            pred_y2 - pred_y1
        ).clamp(min=0)

        tgt_area = (tgt_x2 - tgt_x1).clamp(min=0) * (
            tgt_y2 - tgt_y1
        ).clamp(min=0)

        union = pred_area + tgt_area - inter_area
        iou = inter_area / (union + eps)

        pred_cx = (pred_x1 + pred_x2) / 2
        pred_cy = (pred_y1 + pred_y2) / 2
        tgt_cx = (tgt_x1 + tgt_x2) / 2
        tgt_cy = (tgt_y1 + tgt_y2) / 2

        center_distance = (pred_cx - tgt_cx).pow(2) + (pred_cy - tgt_cy).pow(2)

        enc_x1 = torch.minimum(pred_x1, tgt_x1)
        enc_y1 = torch.minimum(pred_y1, tgt_y1)
        enc_x2 = torch.maximum(pred_x2, tgt_x2)
        enc_y2 = torch.maximum(pred_y2, tgt_y2)

        enc_diagonal = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2)

        pred_w = (pred_x2 - pred_x1).clamp(min=eps)
        pred_h = (pred_y2 - pred_y1).clamp(min=eps)
        tgt_w = (tgt_x2 - tgt_x1).clamp(min=eps)
        tgt_h = (tgt_y2 - tgt_y1).clamp(min=eps)

        v = (4.0 / (torch.pi**2)) * (
            torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h)
        ).pow(2)

        with torch.no_grad():
            alpha = v / (1.0 - iou + v + eps)

        ciou = iou - center_distance / (enc_diagonal + eps) - alpha * v

        return ciou.clamp(min=-1.0, max=1.0)

    def flatten_outputs(
        self,
        outputs: List[Dict[str, torch.Tensor]],
    ):
        all_box_logits = []
        all_obj_logits = []
        all_cls_logits = []
        all_points = []
        all_strides = []

        for output in outputs:
            box_logits = output["box_logits"]
            obj_logits = output["obj_logits"]
            cls_logits = output["cls_logits"]
            stride = output["stride"]

            batch_size, _, height, width = box_logits.shape
            device = box_logits.device

            box_logits = box_logits.permute(0, 2, 3, 1).contiguous()
            box_logits = box_logits.view(
                batch_size,
                height * width,
                4,
                self.num_bins,
            )

            obj_logits = obj_logits.permute(0, 2, 3, 1).contiguous()
            obj_logits = obj_logits.view(batch_size, height * width)

            cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
            cls_logits = cls_logits.view(
                batch_size,
                height * width,
                self.num_classes,
            )

            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=device),
                torch.arange(width, device=device),
                indexing="ij",
            )

            points = torch.stack(
                [
                    (grid_x.float() + 0.5) * stride,
                    (grid_y.float() + 0.5) * stride,
                ],
                dim=-1,
            ).view(-1, 2)

            strides = torch.full(
                (height * width,),
                float(stride),
                dtype=torch.float32,
                device=device,
            )

            all_box_logits.append(box_logits)
            all_obj_logits.append(obj_logits)
            all_cls_logits.append(cls_logits)
            all_points.append(points)
            all_strides.append(strides)

        box_logits = torch.cat(all_box_logits, dim=1)
        obj_logits = torch.cat(all_obj_logits, dim=1)
        cls_logits = torch.cat(all_cls_logits, dim=1)
        points = torch.cat(all_points, dim=0)
        strides = torch.cat(all_strides, dim=0)

        decoded_boxes = self.decode_boxes_from_logits(
            box_logits=box_logits,
            points=points,
            strides=strides,
        )

        return box_logits, obj_logits, cls_logits, decoded_boxes, points, strides

    def decode_boxes_from_logits(
        self,
        box_logits: torch.Tensor,
        points: torch.Tensor,
        strides: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.softmax(box_logits, dim=-1)

        projection = self.dfl_projection.view(1, 1, 1, self.num_bins)

        distances = (probs * projection).sum(dim=-1)
        distances = distances * strides.view(1, -1, 1)

        x_center = points[:, 0].view(1, -1)
        y_center = points[:, 1].view(1, -1)

        left = distances[:, :, 0]
        top = distances[:, :, 1]
        right = distances[:, :, 2]
        bottom = distances[:, :, 3]

        x1 = x_center - left
        y1 = y_center - top
        x2 = x_center + right
        y2 = y_center + bottom

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        boxes = boxes.clamp(min=0, max=self.image_size)

        return boxes

    @torch.no_grad()
    def assign_targets_single_image(
        self,
        pred_boxes: torch.Tensor,
        cls_logits: torch.Tensor,
        obj_logits: torch.Tensor,
        points: torch.Tensor,
        strides: torch.Tensor,
        targets: torch.Tensor,
    ):
        device = pred_boxes.device
        num_points = pred_boxes.shape[0]

        target_scores = torch.zeros(
            (num_points, self.num_classes),
            dtype=torch.float32,
            device=device,
        )

        target_obj = torch.zeros(
            (num_points,),
            dtype=torch.float32,
            device=device,
        )

        target_boxes = torch.zeros(
            (num_points, 4),
            dtype=torch.float32,
            device=device,
        )

        foreground_mask = torch.zeros(
            (num_points,),
            dtype=torch.bool,
            device=device,
        )

        gt_classes, gt_boxes = self.targets_to_xyxy(
            targets=targets,
            device=device,
        )

        num_gt = gt_boxes.shape[0]

        if num_gt == 0:
            return target_scores, target_obj, target_boxes, foreground_mask

        ious = self.box_iou_matrix(
            boxes1=pred_boxes,
            boxes2=gt_boxes,
        ).clamp(min=0)

        class_prob = torch.sigmoid(cls_logits)
        obj_prob = torch.sigmoid(obj_logits).unsqueeze(1)

        matched_class_prob = class_prob[:, gt_classes]
        matched_score = matched_class_prob * obj_prob

        point_x = points[:, 0].unsqueeze(1)
        point_y = points[:, 1].unsqueeze(1)

        gt_x1 = gt_boxes[:, 0].unsqueeze(0)
        gt_y1 = gt_boxes[:, 1].unsqueeze(0)
        gt_x2 = gt_boxes[:, 2].unsqueeze(0)
        gt_y2 = gt_boxes[:, 3].unsqueeze(0)

        in_gt_box = (
            (point_x >= gt_x1)
            & (point_x <= gt_x2)
            & (point_y >= gt_y1)
            & (point_y <= gt_y2)
        )

        gt_cx = ((gt_boxes[:, 0] + gt_boxes[:, 2]) / 2).unsqueeze(0)
        gt_cy = ((gt_boxes[:, 1] + gt_boxes[:, 3]) / 2).unsqueeze(0)

        radius = self.center_radius * strides.unsqueeze(1)

        in_center = (
            (point_x >= gt_cx - radius)
            & (point_x <= gt_cx + radius)
            & (point_y >= gt_cy - radius)
            & (point_y <= gt_cy + radius)
        )

        candidate_mask = in_gt_box | in_center

        alignment_metric = (
            matched_score.clamp(min=1e-9).pow(self.alpha)
            * ious.clamp(min=1e-9).pow(self.beta)
        )

        alignment_metric = alignment_metric * candidate_mask.float()

        positive_mask = torch.zeros_like(candidate_mask)

        for gt_idx in range(num_gt):
            candidate_metric = alignment_metric[:, gt_idx]

            if candidate_metric.max() <= 0:
                continue

            topk = min(self.topk, candidate_metric.numel())

            topk_values, topk_indices = torch.topk(
                candidate_metric,
                k=topk,
                largest=True,
            )

            valid_topk = topk_values > 0

            if valid_topk.sum() == 0:
                continue

            positive_mask[topk_indices[valid_topk], gt_idx] = True

        if positive_mask.sum() == 0:
            return target_scores, target_obj, target_boxes, foreground_mask

        multiple_match = positive_mask.sum(dim=1) > 1

        if multiple_match.any():
            matched_ious = ious[multiple_match]
            best_gt_for_multi = matched_ious.argmax(dim=1)

            positive_mask[multiple_match] = False
            positive_mask[multiple_match, best_gt_for_multi] = True

        foreground_mask = positive_mask.sum(dim=1) > 0

        assigned_gt_indices = positive_mask[foreground_mask].float().argmax(dim=1)
        assigned_gt_boxes = gt_boxes[assigned_gt_indices]
        assigned_gt_classes = gt_classes[assigned_gt_indices]

        assigned_ious = ious[foreground_mask, assigned_gt_indices]
        assigned_ious = assigned_ious.clamp(min=0.0, max=1.0)

        # Quality-aware confidence target:
        # higher-quality boxes receive higher objectness/class targets,
        # lower-quality positives are pushed down instead of being treated
        # like equally good object predictions.
        quality_targets = assigned_ious.pow(self.objectness_quality_power)
        quality_targets = quality_targets.clamp(
            min=self.min_quality_target,
            max=1.0,
        )

        target_boxes[foreground_mask] = assigned_gt_boxes
        target_obj[foreground_mask] = quality_targets

        target_scores[
            foreground_mask,
            assigned_gt_classes,
        ] = quality_targets

        return target_scores, target_obj, target_boxes, foreground_mask

    def bbox_to_dfl_targets(
        self,
        points: torch.Tensor,
        strides: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        point_x = points[:, 0]
        point_y = points[:, 1]

        left = (point_x - target_boxes[:, 0]) / strides
        top = (point_y - target_boxes[:, 1]) / strides
        right = (target_boxes[:, 2] - point_x) / strides
        bottom = (target_boxes[:, 3] - point_y) / strides

        distances = torch.stack([left, top, right, bottom], dim=1)

        distances = distances.clamp(
            min=0,
            max=self.reg_max - 1e-4,
        )

        return distances

    def dfl_loss(
        self,
        pred_logits: torch.Tensor,
        target_distances: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        left_bins = target_distances.floor().long()
        right_bins = left_bins + 1

        right_bins = right_bins.clamp(max=self.reg_max)

        weight_right = target_distances - left_bins.float()
        weight_left = 1.0 - weight_right

        pred_logits_flat = pred_logits.reshape(-1, self.num_bins)

        left_loss = F.cross_entropy(
            pred_logits_flat,
            left_bins.reshape(-1),
            reduction="none",
        )

        right_loss = F.cross_entropy(
            pred_logits_flat,
            right_bins.reshape(-1),
            reduction="none",
        )

        loss = left_loss * weight_left.reshape(-1) + right_loss * weight_right.reshape(-1)
        loss = loss.view(-1, 4).mean(dim=1)

        weighted_loss = (loss * weights).sum() / weights.sum().clamp(min=1.0)

        return weighted_loss

    def build_positive_diagnostics(
        self,
        foreground_mask: torch.Tensor,
        target_boxes: torch.Tensor,
        strides: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = foreground_mask.device
        batch_size = foreground_mask.shape[0]

        expanded_strides = strides.unsqueeze(0).expand(batch_size, -1)

        positive_strides = expanded_strides[foreground_mask]
        positive_boxes = target_boxes[foreground_mask]

        zero = torch.tensor(0.0, dtype=torch.float32, device=device)

        diagnostics = {
            "num_positive_stride_8": zero.clone(),
            "num_positive_stride_16": zero.clone(),
            "num_positive_stride_32": zero.clone(),
            "num_positive_small": zero.clone(),
            "num_positive_medium": zero.clone(),
            "num_positive_large": zero.clone(),
        }

        if positive_strides.numel() == 0:
            return diagnostics

        diagnostics["num_positive_stride_8"] = (positive_strides == 8).sum().float()
        diagnostics["num_positive_stride_16"] = (positive_strides == 16).sum().float()
        diagnostics["num_positive_stride_32"] = (positive_strides == 32).sum().float()

        areas = self.box_area(positive_boxes)

        diagnostics["num_positive_small"] = (areas < 32.0 * 32.0).sum().float()
        diagnostics["num_positive_medium"] = (
            (areas >= 32.0 * 32.0)
            & (areas < 96.0 * 96.0)
        ).sum().float()
        diagnostics["num_positive_large"] = (areas >= 96.0 * 96.0).sum().float()

        return diagnostics

    def forward(
        self,
        outputs,
        targets: List[torch.Tensor],
    ):
        if isinstance(outputs, dict) and "main" in outputs:
            outputs = outputs["main"]

        (
            box_logits,
            obj_logits,
            cls_logits,
            decoded_boxes,
            points,
            strides,
        ) = self.flatten_outputs(outputs)

        batch_size, num_points, _, _ = box_logits.shape
        device = box_logits.device

        target_scores_list = []
        target_obj_list = []
        target_boxes_list = []
        foreground_mask_list = []

        for batch_idx in range(batch_size):
            target_scores, target_obj, target_boxes, foreground_mask = (
                self.assign_targets_single_image(
                    pred_boxes=decoded_boxes[batch_idx].detach(),
                    cls_logits=cls_logits[batch_idx].detach(),
                    obj_logits=obj_logits[batch_idx].detach(),
                    points=points,
                    strides=strides,
                    targets=targets[batch_idx],
                )
            )

            target_scores_list.append(target_scores)
            target_obj_list.append(target_obj)
            target_boxes_list.append(target_boxes)
            foreground_mask_list.append(foreground_mask)

        target_scores = torch.stack(target_scores_list, dim=0)
        target_obj = torch.stack(target_obj_list, dim=0)
        target_boxes = torch.stack(target_boxes_list, dim=0)
        foreground_mask = torch.stack(foreground_mask_list, dim=0)

        num_positive_points = foreground_mask.sum().float().clamp(min=1.0)

        cls_loss_raw = self.focal_bce_loss(
            logits=cls_logits,
            targets=target_scores,
        )

        cls_loss = cls_loss_raw.sum() / num_positive_points

        obj_loss_raw = self.focal_bce_loss(
            logits=obj_logits,
            targets=target_obj,
        )

        obj_loss = obj_loss_raw.sum() / num_positive_points

        if foreground_mask.any():
            positive_pred_boxes = decoded_boxes[foreground_mask]
            positive_target_boxes = target_boxes[foreground_mask]
            positive_weights = target_obj[foreground_mask].detach().clamp(min=0.05)

            ciou = self.bbox_ciou(
                pred_boxes=positive_pred_boxes,
                target_boxes=positive_target_boxes,
            )

            box_loss = ((1.0 - ciou) * positive_weights).sum()
            box_loss = box_loss / positive_weights.sum().clamp(min=1.0)

            positive_points = points.unsqueeze(0).expand(batch_size, -1, -1)[
                foreground_mask
            ]

            positive_strides = strides.unsqueeze(0).expand(batch_size, -1)[
                foreground_mask
            ]

            target_distances = self.bbox_to_dfl_targets(
                points=positive_points,
                strides=positive_strides,
                target_boxes=positive_target_boxes,
            )

            positive_box_logits = box_logits[foreground_mask]

            dfl_loss = self.dfl_loss(
                pred_logits=positive_box_logits,
                target_distances=target_distances,
                weights=positive_weights,
            )
        else:
            box_loss = decoded_boxes.sum() * 0.0
            dfl_loss = box_logits.sum() * 0.0

        total_loss = (
            self.cls_loss_weight * cls_loss
            + self.obj_loss_weight * obj_loss
            + self.box_loss_weight * box_loss
            + self.dfl_loss_weight * dfl_loss
        )

        positive_diagnostics = self.build_positive_diagnostics(
            foreground_mask=foreground_mask,
            target_boxes=target_boxes,
            strides=strides,
        )

        return {
            "loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "obj_loss": obj_loss.detach(),
            "box_loss": box_loss.detach(),
            "dfl_loss": dfl_loss.detach(),
            "num_positive_points": foreground_mask.sum().detach(),
            "num_positive_stride_8": positive_diagnostics["num_positive_stride_8"].detach(),
            "num_positive_stride_16": positive_diagnostics["num_positive_stride_16"].detach(),
            "num_positive_stride_32": positive_diagnostics["num_positive_stride_32"].detach(),
            "num_positive_small": positive_diagnostics["num_positive_small"].detach(),
            "num_positive_medium": positive_diagnostics["num_positive_medium"].detach(),
            "num_positive_large": positive_diagnostics["num_positive_large"].detach(),
        }
