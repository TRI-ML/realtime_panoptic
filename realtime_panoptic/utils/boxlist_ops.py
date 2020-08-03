# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from maskrcnn-benchmark
# https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/structures/boxlist_ops.py
import torch
from torchvision.ops.boxes import nms as _box_nms

from .bounding_box import BoxList


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """Performs non-maximum suppression on a boxlist.
    The ranking scores are specified in a boxlist field via score_field.

    Parameters
    ----------
    boxlist : BoxList
        Original boxlist

    nms_thresh : float
        NMS threshold

    max_proposals :  int
        If > 0, then only the top max_proposals are kept after non-maximum suppression

    score_field : str
        Boxlist field to use during NMS score ranking. Field value needs to be numeric.
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[:max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """Only keep boxes with both sides >= min_size

    Parameters
    ----------
    boxlist : Boxlist
        Original boxlist

    min_size : int
        Max edge dimension of boxes to be kept.
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2, optimize_memory=False):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Parameters
    ----------
    box1: BoxList
        Bounding boxes, sized [N,4].
    box2: BoxList
        Bounding boxes, sized [M,4].


    Returns
    -------
    iou : tensor
        IoU of input boxes in matrix form, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError("boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area2 = boxlist2.area()

    if not optimize_memory:

        # If not optimizing memory, then following original ``maskrcnn-benchmark`` implementation

        area1 = boxlist1.area()

        box1, box2 = boxlist1.bbox, boxlist2.bbox

        lt = torch.max(box1[:, None, :2], box2[:, :2])  # shape: (N, M, 2)
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # shape: (N, M, 2)

        TO_REMOVE = 1

        wh = (rb - lt + TO_REMOVE).clamp(min=0)  # shape: (N, M, 2)
        inter = wh[:, :, 0] * wh[:, :, 1]  # shape: (N, M)

        iou = inter / (area1[:, None] + area2 - inter)

    else:

        # If optimizing memory, construct IoU matrix one box1 entry at a time
        # (in current usage this means one GT at a time)

        # Entry i of ious will hold the IoU between the ith box in boxlist1 and all boxes
        # in boxlist2
        ious = []

        box2 = boxlist2.bbox

        for i in range(N):
            area1 = boxlist1.area(i)

            box1 = boxlist1.bbox[i].unsqueeze(0)

            lt = torch.max(box1[:, None, :2], box2[:, :2])  # shape: (1, M, 2)
            rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # shape: (1, M, 2)

            TO_REMOVE = 1
            wh = (rb - lt + TO_REMOVE).clamp(min=0)  # shape: (1, M, 2)

            inter = wh[:, :, 0] * wh[:, :, 1]  # shape: (1, M)

            iou = inter / (area1 + area2 - inter)

            ious.append(iou)

        iou = torch.cat(ious)  # shape: (N, M)

    return iou


def cat_boxlist(bboxes):
    """Concatenates a list of BoxList  into a single BoxList
    image sizes needs to be same in this operation.

    Parameters
    ----------
    bboxes : list[BoxList]
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(torch.cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = torch.cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes


def pair_boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Parameters
    ----------
    box1 : BoxList
        Bounding boxes, sized [N,4].
    box2 : BoxList
        Bounding boxes, sized [N,4].
    
    Returns
    -------
    iou : tensor,
        Tensor of iou between the input pair of boxes. sized [N].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError("boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    assert len(boxlist1) == len(boxlist2), "Two boxlists should have same length"
    N = len(boxlist1)

    area2 = boxlist2.area()
    area1 = boxlist1.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox
    lt = torch.max(box1[:, :2], box2[:, :2])  # shape: (N, 2)
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # shape: (N, 2)
    TO_REMOVE = 1
    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # shape: (N, 2)
    inter = wh[:, 0] * wh[:, 1]  # shape: (N, 1)
    iou = inter / (area1 + area2 - inter)
    return iou
