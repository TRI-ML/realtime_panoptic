# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from maskrcnn-benchmark
# https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/structures/bounding_box.py

import math

import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1
ROTATE_90 = 2


class BoxList:
    """This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="xyxy"):
        """Initial function.

        Parameters
        ----------
        bbox: tensor
            Nx4 tensor following bounding box parameterization defined by "mode".

        image_size: list
            [W,H] Image size.

        mode: str
            Bounding box parameterization. 'xyxy' or 'xyhw'.
        """
        device = bbox.device if isinstance(
            bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError("bbox should have 2 dimensions, got {}".format(
                bbox.ndimension()), bbox)
        if bbox.size(-1) != 4:
            raise ValueError("last dimension of bbox should have a "
                             "size of 4, got {}".format(bbox.size(-1)))
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        """Add a field to boxlist.
        """
        self.extra_fields[field] = field_data

    def get_field(self, field):
        """Get a field from boxlist.
        """
        return self.extra_fields[field]

    def has_field(self, field):
        """Check if certain field exist in boxlist
        """
        return field in self.extra_fields

    def fields(self):
        """Get all available field names.
        """
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        """Copy extra fields from given boxlist to current boxlist.
        """
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        """Convert bounding box parameterization mode.
        """
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 1
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE),
                dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        """split lists of bounding box corners. 
        """
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    def resize(self, size, *args, **kwargs):
        """Returns a resized copy of this bounding box.

        Parameters
        ----------
        size: list or tuple
            The requested image size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1)
        bbox = BoxList(scaled_box, size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def transpose(self, method):
        """Transpose bounding box (flip or rotate in 90 degree steps)

        Parameters
        ----------
        method: str
            One of:py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
           :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`,:py:attr:`PIL.Image.ROTATE_90`,
           :py:attr:`PIL.Image.ROTATE_180`,:py:attr:`PIL.Image.ROTATE_270`,
           :py:attr:`PIL.Image.TRANSPOSE` or:py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, ROTATE_90):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM and ROTATE_90 implemented"
            )

        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin
        elif method == ROTATE_90:
            transposed_xmin = ymin * image_width / image_height
            transposed_xmax = ymax * image_width / image_height
            transposed_ymin = (image_width - xmax) * image_height / image_width
            transposed_ymax = (image_width - xmin) * image_height / image_width

        transposed_boxes = torch.cat((transposed_xmin, transposed_ymin,
                                      transposed_xmax, transposed_ymax),
                                     dim=-1)
        bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def translate(self, x_offset, y_offset):
        """Translate bounding box.

        Parameters
        ----------
        x_offseflt: float
            x offset
        y_offset: float
            y offset
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()

        translated_xmin = xmin + x_offset
        translated_xmax = xmax + x_offset
        translated_ymin = ymin + y_offset
        translated_ymax = ymax + y_offset

        translated_boxes = torch.cat((translated_xmin, translated_ymin,
                                      translated_xmax, translated_ymax),
                                     dim=-1)
        bbox = BoxList(translated_boxes, self.size, mode="xyxy")
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.translate(x_offset, y_offset)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """Crop a rectangular region from this bounding box.
        
        Parameters
        ----------
        box: tuple
            The box is a 4-tuple defining the left, upper, right, and lower pixel
            coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (
                cropped_ymin == cropped_ymax)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1)
        bbox = BoxList(cropped_box, (w, h), mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def augmentation_crop(self, top, left, crop_height, crop_width):
        """Random cropping of the bounding box (bbox).
        This function is created for label box to be crop at training time.

        Parameters:
        -----------
        top: int
            Top pixel position of crop area

        left: int
            left pixel position of crop area

        crop_height: int
            Height of crop area

        crop_width: int
            Width of crop area

        Returns:
        --------
        bbox_cropped: BoxList
            A BoxList object with instances after cropping. If no valid instance is left after
            cropping, return None.


        """
        # SegmentationMasks object
        masks = self.extra_fields["masks"]

        # Conduct mask level cropping and return only the valid ones left.
        masks_cropped, keep_ids = masks.augmentation_crop(
            top, left, crop_height, crop_width)

        # the return cropped mask should be in "poly" mode
        if not keep_ids:
            return None
        assert masks_cropped.mode == "poly"
        bbox_cropped = []
        labels = self.extra_fields["labels"]
        labels_cropped = [labels[idx] for idx in keep_ids]
        labels_cropped = torch.as_tensor(labels_cropped, dtype=torch.long)

        crop_box_xyxy = [
            float(left),
            float(top),
            float(left + crop_width),
            float(top + crop_height)
        ]
        # Crop bounding box.
        # Note: this function will not change "masks"
        self.extra_fields.pop("masks", None)
        new_bbox = self.crop(crop_box_xyxy).convert("xyxy")

        # Further clip the boxes according to the clipped masks.
        for mask_id, box_id in enumerate(keep_ids):
            x1, y1, x2, y2 = new_bbox.bbox[box_id].numpy()

            # only resize the box on the edge:
            if x1 > 0 and y1 > 0 and x2 < crop_width - 1 and y2 < crop_height - 1:
                bbox_cropped.append([x1, y1, x2, y2])
            else:
                # get PolygonInstance for current instance
                current_polygon_instance = masks_cropped.instances.polygons[
                    mask_id]
                x_ids = []
                y_ids = []
                for poly in current_polygon_instance.polygons:
                    p = poly.clone()
                    x_ids.extend(p[0::2])
                    y_ids.extend(p[1::2])
                bbox_cropped.append(
                    [min(x_ids),
                     min(y_ids),
                     max(x_ids),
                     max(y_ids)])
        bbox_cropped = BoxList(
            bbox_cropped, (crop_width, crop_height), mode="xyxy")
        bbox_cropped = bbox_cropped.convert(self.mode)
        bbox_cropped.add_field("masks", masks_cropped)
        bbox_cropped.add_field("labels", labels_cropped)
        return bbox_cropped

    def to(self, device):
        """Move object to torch device.
        """
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        """Get a sub-list of Boxlist as a new Boxlist
        """
        item_bbox = self.bbox[item]
        if len(item_bbox.shape) < 2:
            item_bbox.unsqueeze(0)
        bbox = BoxList(item_bbox, self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        """Clip bounding box coordinates according to the image range. 
        """
        TO_REMOVE = 1
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self, idx=None):
        """Get bounding box area.
        """
        box = self.bbox if idx is None else self.bbox[idx].unsqueeze(0)
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * \
                (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        """Provide deep copy of Boxlist with requested fields.
        """
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(
                    field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


if __name__ == "__main__":
    bbox = BoxList([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)
