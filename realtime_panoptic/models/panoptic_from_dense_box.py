# Copyright 2020 Toyota Research Institute.  All rights reserved.
import torch
import torch.nn.functional as F
from realtime_panoptic.utils.bounding_box import BoxList
from realtime_panoptic.utils.boxlist_ops import (boxlist_nms, cat_boxlist, remove_small_boxes)

class PanopticFromDenseBox:
    """Performs post-processing on the outputs of the RTPanonet.

    Parameters
    ----------
    pre_nms_thresh: float
        Acceptance class probability threshold for bounding box candidates before NMS.

    pre_nms_top_n: int
        Maximum number of accepted bounding box candidates before NMS.

    nms_thresh: float
        NMS threshold.

    fpn_post_nms_top_n: int
        Maximum number of detected object per image.

    min_size: int
        Minimum dimension of accepted detection.

    num_classes: int
        Number of total semantic classes (stuff and things).

    mask_thresh: float
        Bounding box IoU threshold to determined 'similar bounding box' in mask reconstruction.

    instance_id_range: list of int
        [min_id, max_id] defines the range of id in 1:num_classes that corresponding to thing classes.

    is_training: bool
        Whether the current process is during training process.
    """

    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        mask_thresh,
        instance_id_range,
        is_training
    ):
        super(PanopticFromDenseBox, self).__init__()
        # assign parameters
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.mask_thresh = mask_thresh
        self.instance_id_range = instance_id_range
        self.is_training = is_training

    def process(
        self, locations, box_cls, box_regression, centerness, levelness_logits, semantic_logits, image_sizes
    ):
        """ Reconstruct panoptic segmentation result from raw predictions.

        This function conduct post processing of panoptic head raw prediction, including bounding box
        prediction, semantic segmentation and levelness to reconstruct instance segmentation results.

        Parameters
        ----------
        locations: list of torch.Tensor
            Corresponding pixel locations of each FPN predictions.

        box_cls: list of torch.Tensor
            Predicted bounding box class from each FPN layers.

        box_regression: list of torch.Tensor
            Predicted bounding box offsets from each FPN layers.

        centerness: list of torch.Tensor
            Predicted object centerness from each FPN layers.

        levelness_logits:
            Global prediction of best source FPN layer for each pixel location.

        semantic_logits:
            Global prediction of semantic segmentation.

        image_sizes: list of [int,int]
            Image sizes.

        Returns:
        --------
        boxlists: list of BoxList
            reconstructed instances with masks.
        """
        num_locs_per_level = [len(loc_per_level) for loc_per_level in locations]

        sampled_boxes = []
        for i, (l, o, b, c) in enumerate(zip(locations[:-1], box_cls, box_regression, centerness)):
            if self.is_training:
                layer_boxes = self.forward_for_single_feature_map(l, o, b, c, image_sizes)
                for layer_box in layer_boxes:
                    pred_indices = layer_box.get_field("indices")
                    pred_indices = pred_indices + sum(num_locs_per_level[:i])
                    layer_box.add_field("indices", pred_indices)
                sampled_boxes.append(layer_boxes)
            else:
                sampled_boxes.append(self.forward_for_single_feature_map(l, o, b, c, image_sizes))

        # sampled_boxes are a list of bbox_list per level
        # the following converts it to per image
        boxlists = list(zip(*sampled_boxes))
        # per image, concat bbox_list of different levels into one bbox_list
        # boxlists is a list of bboxlists of N images
        try:
            boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
            boxlists = self.select_over_all_levels(boxlists)
        except Exception as e:
            print(e)
            for boxlist in boxlists:
                for box in boxlist:
                    print(box, "box shape", box.bbox.shape)

        # Generate bounding box feature map at size of [H/4, W/4] with bounding box prediction as features.
        levelness_locations = locations[-1]
        _, c_semantic, _, _ = semantic_logits.shape
        N, _, h_map, w_map = levelness_logits.shape
        bounding_box_feature_map = self.generate_box_feature_map(levelness_locations, box_regression, levelness_logits)

        # process semantic raw prediction
        semantic_logits = F.interpolate(semantic_logits, size=(h_map, w_map), mode='bilinear')
        semantic_logits = semantic_logits.view(N, c_semantic, h_map, w_map).permute(0, 2, 3, 1)
        semantic_logits = semantic_logits.reshape(N, -1, c_semantic)

        # insert semantic prob into mask
        semantic_probability = F.softmax(semantic_logits, dim=2)
        semantic_probability = semantic_probability[:, :, self.instance_id_range[0]:]
        boxlists = self.mask_reconstruction(
            boxlists=boxlists,
            box_feature_map=bounding_box_feature_map,
            semantic_prob=semantic_probability,
            box_feature_map_location=levelness_locations,
            h_map=h_map,
            w_map=w_map
        )
        # resize instance masks to original image size
        if not self.is_training:
            for boxlist in boxlists:
                masks = boxlist.get_field("mask")
                # NOTE: BoxList size is the image size without padding. MASK here is a mask with padding.
                # Mask need to be interpolated into padded image size and then crop to unpadded size.
                w, h = boxlist.size
                if len(masks.shape) == 3 and masks.shape[0] != 0:
                    masks = F.interpolate(masks.unsqueeze(0), size=(h_map * 4, w_map * 4), mode='bilinear').squeeze()
                else:
                    # handle 0 shape dummy mask.
                    masks = masks.view([-1, h_map * 4, w_map * 4])
                masks = masks >= self.mask_thresh
                if len(masks.shape) < 3:
                    masks = masks.unsqueeze(0)
                masks = masks[:, 0:h, 0:w].contiguous()
                boxlist.add_field("mask", masks)
        return boxlists

    def forward_for_single_feature_map(self, locations, box_cls, box_regression, centerness, image_sizes):
        """Recover dense bounding box detection results from raw predictions for each FPN layer.

        Parameters
        ----------
        locations: torch.Tensor
            Corresponding pixel location of FPN feature map with size of (N, H * W, 2).

        box_cls: torch.Tensor
            Predicted bounding box class probability with size of (N, C, H, W).

        box_regression: torch.Tensor
            Predicted bounding box offset centered at corresponding pixel with size of (N, 4, H, W).

        centerness: torch.Tensor
            Predicted centerness of corresponding pixel with size of (N, 1, H, W).

        Note: N is the number of FPN level.

        Returns
        -------
        results: List of BoxList
            A list of dense bounding boxes from each FPN layer.
        """

        N, C, H, W = box_cls.shape
        # M = H x W is the total number of proposal for this single feature map

        # put in the same format as locations
        # from (N, C, H, W) to (N, H, W, C)
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        # from (N, H, W, C) to (N, M, C)
        # map class prob to (-1, +1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        # from (N, 4, H, W) to (N, H, W, 4) to (N, M, 4)
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        # from (N, 4, H, W) to (N, H, W, 1) to (N, M)
        # map centerness prob to (-1, +1)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()

        # before NMS, per level filter out low cls prob with threshold 0.05
        # after this candidate_inds of size (N, M, C) with values corresponding to
        # low prob predictions become 0, otherwise 1
        candidate_inds = box_cls > self.pre_nms_thresh

        # pre_nms_top_n of size (N, M * C) => (N, 1)
        # N -> batch index, 1 -> total number of bbox predictions per image
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        # total number of proposal before NMS
        # if have more than self.pre_nms_top_n (1000) clamp to 1000
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        # (N, M, C) * (N, M, 1)
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            # filer out low score candidates
            per_box_cls = box_cls[i]  #  (M, C)
            per_candidate_inds = candidate_inds[i]  #  (M, C)
            # per_box_cls of size P, P < M * C
            per_box_cls = per_box_cls[per_candidate_inds]

            # indices of seeds bounding boxes
            # 0-dim corresponding to M, location
            # 1-dim corresponding to C, class
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            # Each of the following is of size P < M * C
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            # per_box_regression of size (M, 4)
            per_box_regression = box_regression[i]
            # (M, 4) => (P, 4)
            # in P, there might be identical bbox prediction in M
            per_box_regression = per_box_regression[per_box_loc]
            # (M, 2) => (P, 2)
            # in P, there might be identical locations in M
            per_locations = locations[per_box_loc]


            # upperbound of the number of predictions for this image
            per_pre_nms_top_n = pre_nms_top_n[i]

            # if valid predictions is more than the upperbound
            # only select topK
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                if self.is_training:
                    per_box_loc = per_box_loc[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ],
                                     dim=1)

            h, w = image_sizes[i]

            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            if self.is_training:
                boxlist.add_field("indices", per_box_loc)

            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)
        return results

    def generate_box_feature_map(self, location, box_regression, levelness_logits):
        """Generate bounding box feature aggregating dense bounding box predictions.

        Parameters
        ----------
        location: torch.Tensor
            Pixel location of levelness.

        box_regression: list of torch.Tensor
            Bounding box offsets from each FPN.

        levelness_logits: torch.Tenor
            Global prediction of best source FPN layer for each pixel location.
            Predict at the resolution of (H/4, W/4).

        Returns
        -------
        bounding_box_feature_map: torch.Tensor
            Aggregated bounding box feature map.
        """
        upscaled_box_reg = []
        N, _, h_map, w_map = levelness_logits.shape
        downsampled_shape = torch.Size((h_map, w_map))
        for box_reg in box_regression:
            upscaled_box_reg.append(F.interpolate(box_reg, size=downsampled_shape, mode='bilinear').unsqueeze(1))

        # N_level, 4, h_map, w_map
        upscaled_box_reg = torch.cat(upscaled_box_reg, 1)

        max_v, level = torch.max(levelness_logits[:, 1:, :, :], dim=1)

        box_feature_map = torch.gather(
            upscaled_box_reg, dim=1, index=level.unsqueeze(1).expand([N, 4, h_map, w_map]).unsqueeze(1)
        )

        box_feature_map = box_feature_map.view(N, 4, h_map, w_map).permute(0, 2, 3, 1)
        box_feature_map = box_feature_map.reshape(N, -1, 4)
        # generate all valid bboxes from feature map
        # shape (N, M, 4)
        levelness_locations_repeat = location.repeat(N, 1, 1)
        bounding_box_feature_map = torch.stack([
            levelness_locations_repeat[:, :, 0] - box_feature_map[:, :, 0],
            levelness_locations_repeat[:, :, 1] - box_feature_map[:, :, 1],
            levelness_locations_repeat[:, :, 0] + box_feature_map[:, :, 2],
            levelness_locations_repeat[:, :, 1] + box_feature_map[:, :, 3],
        ], dim=2)
        return bounding_box_feature_map

    def mask_reconstruction(self, boxlists, box_feature_map, semantic_prob, box_feature_map_location, h_map, w_map):
        """Reconstruct instance mask from dense bounding box and semantic smoothing.

        Parameters
        ----------
        boxlists: List of Boxlist
            Object detection result after NMS.

        box_feature_map: torch.Tensor
            Aggregated bounding box feature map.

        semantic_prob: torch.Tensor
            Prediction semantic probability.

        box_feature_map_location: torch.Tensor
            Corresponding pixel location of bounding box feature map.

        h_map: int
            Height of bounding box feature map.

        w_map: int
            Width of bounding box feature map.
        """
        for i, (boxlist, per_image_bounding_box_feature_map, per_image_semantic_prob,
                box_feature_map_loc) in enumerate(zip(boxlists, box_feature_map, semantic_prob, box_feature_map_location)):

            # decode mask from bbox embedding
            if len(boxlist) > 0:
                # query_boxes is of shape (P, 4)
                # dense_detections is of shape (P', 4)
                # P' is larger than P
                query_boxes = boxlist.bbox
                propose_cls = boxlist.get_field("labels")
                # (P, 4) -> (P, 4, 1) -> (P, 4, P) -> (P, P', 4)
                propose_bbx = query_boxes.unsqueeze(2).repeat(1, 1,
                                                             per_image_bounding_box_feature_map.shape[0]).permute(0, 2, 1)
                # (P',4) -> (4, P') -> (1, 4, P') -> (P, 4, P') -> (P, P', 4)
                voting_bbx = per_image_bounding_box_feature_map.permute(1, 0).unsqueeze(0).repeat(query_boxes.shape[0], 1,
                                                                                            1).permute(0, 2, 1)
                # implementation based on IOU for bbox_correlation_map
                # 0, 1, 2, 3 => left, top, right, bottom
                proposal_area = (propose_bbx[:, :, 2] - propose_bbx[:, :, 0]) * \
                                        (propose_bbx[:, :, 3] - propose_bbx[:, :, 1])
                voting_area = (voting_bbx[:, :, 2] - voting_bbx[:, :, 0]) * \
                                        (voting_bbx[:, :, 3] - voting_bbx[:, :, 1])
                w_intersect = torch.min(voting_bbx[:, :, 2], propose_bbx[:, :, 2]) - \
                                        torch.max(voting_bbx[:, :, 0], propose_bbx[:, :, 0])
                h_intersect = torch.min(voting_bbx[:, :, 3], propose_bbx[:, :, 3]) - \
                                        torch.max(voting_bbx[:, :, 1], propose_bbx[:, :, 1])
                w_intersect = w_intersect.clamp(min=0.0)
                h_intersect = h_intersect.clamp(min=0.0)
                w_general = torch.max(voting_bbx[:, :, 2], propose_bbx[:, :, 2]) - \
                                        torch.min(voting_bbx[:, :, 0], propose_bbx[:, :, 0])
                h_general = torch.max(voting_bbx[:, :, 3], propose_bbx[:, :, 3]) - \
                                        torch.min(voting_bbx[:, :, 1], propose_bbx[:, :, 1])
                # calculate IOU
                area_intersect = w_intersect * h_intersect
                area_union = proposal_area + voting_area - area_intersect
                torch.cuda.synchronize()

                area_general = w_general * h_general + 1e-7
                bbox_correlation_map = (area_intersect + 1.0) / (area_union + 1.0) - \
                                            (area_general - area_union) / area_general

                per_image_cls_prob = per_image_semantic_prob[:, propose_cls - 1].permute(1, 0)
                # bbox_correlation_map is of size (P or per_pre_nms_top_n, P')
                bbox_correlation_map = bbox_correlation_map * per_image_cls_prob
                # query_boxes.shape[0] is the number of filtered boxes
                masks = bbox_correlation_map.view(query_boxes.shape[0], h_map, w_map)
                if len(masks.shape) < 3:
                    masks = masks.unsqueeze(0)
                boxlist.add_field("mask", masks)
            else:
                dummy_masks = torch.zeros(len(boxlist), h_map,
                                          w_map).float().to(boxlist.bbox.device).to(boxlist.bbox.dtype)
                boxlist.add_field("mask", dummy_masks)
        return boxlists

    def select_over_all_levels(self, boxlists):
        """NMS of bounding box candidates.

        Parameters
        ----------
        boxlists: list of Boxlist
            Pre-NMS bounding boxes.

        Returns
        -------
        results: list of Boxlist
            Final detection result.
        """
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            boxlist = boxlists[i]
            scores = boxlist.get_field("scores")
            labels = boxlist.get_field("labels")
            if self.is_training:
                indices = boxlist.get_field("indices")
            boxes = boxlist.bbox

            result = []
            w, h = boxlist.size
            # skip the background
            if boxes.shape[0] < 1:
                results.append(boxlist)
                continue
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)
                if len(inds) > 0:
                    scores_j = scores[inds]
                    boxes_j = boxes[inds, :].view(-1, 4)

                    boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                    boxlist_for_class.add_field("scores", scores_j)

                    if self.is_training:
                        indices_j = indices[inds]
                        boxlist_for_class.add_field("indices", indices_j)

                    boxlist_for_class = boxlist_nms(boxlist_for_class, self.nms_thresh, score_field="scores")
                    num_labels = len(boxlist_for_class)
                    boxlist_for_class.add_field(
                        "labels", torch.full((num_labels, ), j, dtype=torch.int64, device=scores.device)
                    )
                    result.append(boxlist_for_class)
            result = cat_boxlist(result)

            # global NMS
            result = boxlist_nms(result, 0.97, score_field="scores")

            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(cls_scores.cpu(), number_of_detections - self.fpn_post_nms_top_n + 1)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results
