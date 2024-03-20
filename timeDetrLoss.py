import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_area
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)


        # Compute the giou cost betwen boxes TODO : fix to angle_box
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]



class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

class TimeSeriesDetrLoss(nn.Module):
    def __init__(self, num_classes: int, weight_set: float = 1.0, weight_bbox: float = 1.0, weight_class: float = 1.0):
        super(TimeSeriesDetrLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_set = weight_set
        self.weight_bbox = weight_bbox
        self.weight_class = weight_class

    def forward(self, pred_bbox: Tensor, target_bbox: Tensor, class_logits: Tensor, target_classes: Tensor):
        """
        Computes the combined loss for the time series object detection task.

        Args:
            pred_bbox (Tensor): Predicted angles of shape (batch_size, num_obj, 2).
            target_bbox (Tensor): Target angles of shape (batch_size, num_obj, 2).
            class_logits (Tensor): Predicted class logits of shape (batch_size, num_obj, num_classes).
            target_classes (Tensor): Target classes of shape (batch_size, num_obj, num_classe).

        Returns:
            Tensor: Combined loss value.
        """
        batch_size = pred_bbox.size(0)
        num_angles = pred_bbox.size(1)

        # Match predicted angles with target angles
        indices = self.matcher(pred_bbox, target_bbox)

        # Compute set loss
        set_loss_value = self.set_loss(pred_bbox, target_bbox)

        # Compute bounding box loss
        bbox_loss_value = self.bbox_loss(pred_bbox, target_bbox, indices)

        # Compute classification loss
        class_loss_value = self.classification_loss(class_logits, target_classes, indices)

        # Combine losses
        combined_loss = self.weight_set * set_loss_value + self.weight_bbox * bbox_loss_value + self.weight_class * class_loss_value

        return combined_loss

    def matcher(self, pred_angles: Tensor, target_angles: Tensor) -> Tensor:
        """
        Matches predicted angles with target angles using the Hungarian algorithm.

        Args:
            pred_angles (Tensor): Predicted angles of shape (batch_size, num_angles, 2).
            target_angles (Tensor): Target angles of shape (batch_size, num_angles, 2).

        Returns:
            Tensor: Indices of matched predictions and targets of shape (batch_size, num_angles).
        """
        batch_size = pred_angles.size(0)
        num_angles = pred_angles.size(1)

        # Compute the pairwise distances between predicted and target angles
        dist_matrix = torch.cdist(pred_angles.view(-1, 2), target_angles.view(-1, 2), p=2)
        dist_matrix = dist_matrix.view(batch_size, num_angles, -1)

        # Use the Hungarian algorithm to find the optimal matching
        indices = []
        for batch_idx in range(batch_size):
            row_ind, col_ind = linear_sum_assignment(dist_matrix[batch_idx].cpu().numpy())
            indices.append(col_ind)

        indices = torch.as_tensor(indices, device=pred_angles.device)

        return indices

    def set_loss(self, pred_angles: Tensor, target_angles: Tensor) -> Tensor:
        """
        Computes the set loss for predicted and target angles.

        Args:
            pred_angles (Tensor): Predicted angles of shape (batch_size, num_angles, 2).
            target_angles (Tensor): Target angles of shape (batch_size, num_angles, 2).

        Returns:
            Tensor: Set loss value.
        """
        batch_size = pred_angles.size(0)
        num_angles = pred_angles.size(1)

        # Compute the pairwise distances between predicted and target angles
        dist_matrix = torch.cdist(pred_angles.view(-1, 2), target_angles.view(-1, 2), p=2)
        dist_matrix = dist_matrix.view(batch_size, num_angles, -1)

        # Sum the minimum distances over all target angles
        loss = torch.sum(dist_matrix.min(dim=2)[0])

        # Normalize the loss by the total number of target angles
        loss = loss / (batch_size * num_angles)

        return loss

    def bbox_loss(self, pred_angles: Tensor, target_angles: Tensor, indices: Tensor) -> Tensor:
        """
        Computes the bounding box loss for matched predicted and target angles.

        Args:
            pred_angles (Tensor): Predicted angles of shape (batch_size, num_angles, 2).
            target_angles (Tensor): Target angles of shape (batch_size, num_angles, 2).
            indices (Tensor): Indices for matched predictions and targets of shape (batch_size, num_angles).

        Returns:
            Tensor: Bounding box loss value.
        """
        batch_size = pred_angles.size(0)
        num_angles = pred_angles.size(1)

        # Filter out unmatched angles
        matched_pred_angles = pred_angles[indices != -1]
        matched_target_angles = target_angles[indices != -1]

        # Compute the L1 loss between matched predicted and target angles
        bbox_loss = F.l1_loss(matched_pred_angles, matched_target_angles, reduction='sum')

        # Normalize the loss by the total number of matched angles
        bbox_loss = bbox_loss / (batch_size * num_angles)

        return bbox_loss

    def classification_loss(self, class_logits: Tensor, target_classes: Tensor, indices: Tensor) -> Tensor:
        """
        Computes the classification loss for matched predicted and target classes.

        Args:
            class_logits (Tensor): Predicted class logits of shape (batch_size, num_angles, num_classes).
            target_classes (Tensor): Target classes of shape (batch_size, num_angles).
            indices (Tensor): Indices for matched predictions and targets of shape (batch_size, num_angles).

        Returns:
            Tensor: Classification loss value.
        """
        batch_size = class_logits.size(0)
        num_angles = class_logits.size(1)

        # Filter out unmatched angles
        matched_class_logits = class_logits[indices != -1]
        matched_target_classes = target_classes[indices != -1]

        # Compute the binary cross-entropy loss
        class_loss = F.cross_entropy(matched_class_logits.view(-1, self.num_classes), matched_target_classes.view(-1))

        # Normalize the loss by the total number of matched angles
        class_loss = class_loss / (batch_size * num_angles)

        return class_loss