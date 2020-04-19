import inspect
import cv2
import numpy as np
from numpy import random

from ..registry import PIPELINES

interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

@PIPELINES.register_module
class Resize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            self.img_scale = img_scale

        self.keep_ratio = keep_ratio

    def _imresize_aspect_ratio(self, img, new_shape, interpolation='bilinear'):
        """Resize image while keeping the aspect ratio.

        Args:
            img (ndarray): The input image.
            scale (float or tuple[int]): The scaling factor or maximum size.
                If it is a float number, then the image will be rescaled by this
                factor, else if it is a tuple of 2 integers, then the image will
                be rescaled as large as possible within the scale.
            return_scale (bool): Whether to return the scaling factor besides the
                rescaled image.
            interpolation (str): Same as :func:`resize`.

        Returns:
            ndarray: The rescaled image.
        """
        h, w = img.shape[:2]
        if isinstance(new_shape, tuple):
            max_edge = min([h, w])
            proportion = new_shape[0] / max_edge
            new_h = int(max(h * proportion, new_shape[1]))
            new_w = int(max(w * proportion, new_shape[0]))
        else:
            raise TypeError(
                'New Shape must be a number or tuple of int, but got {}'.format(
                    type(new_shape)))

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=interp_codes[interpolation])

        delta_x = (new_w - new_shape[0]) // 2
        delta_y = (new_h - new_shape[1]) // 2

        x1 = delta_x
        y1 = delta_y
        x2 = new_w - delta_x
        y2 = new_h - delta_y

        x2 = x2 if (x2 - x1) == new_shape[0] else x2 - 1
        y2 = y2 if (y2 - y1) == new_shape[1] else y2 - 1

        cropped_img = resized_img[y1:y2, x1:x2, :]

        return cropped_img

    def _resize_img(self, results):
        if self.keep_ratio:
            img = self._imresize_aspect_ratio(results['img'], self.img_scale)
        else:
            img, w_scale, h_scale = self.imresize(
                results['img'], results['scale'], return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['keep_ratio'] = self.keep_ratio

    def __call__(self, results):
        self._resize_img(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, multiscale_mode={}, ratio_range={}, '
                     'keep_ratio={})').format(self.img_scale,
                                              self.multiscale_mode,
                                              self.ratio_range,
                                              self.keep_ratio)
        return repr_str


@PIPELINES.register_module
class RandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical']

    def _imflip(self, img, direction='horizontal'):
        """Flip an image horizontally or vertically.

        Args:
            img (ndarray): Image to be flipped.
            direction (str): The flip direction, either "horizontal" or "vertical".

        Returns:
            ndarray: The flipped image.
        """
        assert direction in ['horizontal', 'vertical']
        if direction == 'horizontal':
            return np.flip(img, axis=1)
        else:
            return np.flip(img, axis=0)

    def __call__(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            results['img'] = self._imflip(
                results['img'], direction=results['flip_direction'])

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)


@PIPELINES.register_module
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _impad(self, img, shape, pad_val=0):
        """Pad an image to a certain shape.

        Args:
            img (ndarray): Image to be padded.
            shape (tuple): Expected padding shape.
            pad_val (number or sequence): Values to be filled in padding areas.

        Returns:
            ndarray: The padded image.
        """
        if not isinstance(pad_val, (int, float)):
            assert len(pad_val) == img.shape[-1]
        if len(shape) < len(img.shape):
            shape = shape + (img.shape[-1],)
        assert len(shape) == len(img.shape)
        for i in range(len(shape)):
            assert shape[i] >= img.shape[i]
        pad = np.empty(shape, dtype=img.dtype)
        pad[...] = pad_val
        pad[:img.shape[0], :img.shape[1], ...] = img
        return pad

    def _impad_to_multiple(self, img, divisor, pad_val=0):
        """Pad an image to ensure each edge to be multiple to some number.

        Args:
            img (ndarray): Image to be padded.
            divisor (int): Padded image edges will be multiple to divisor.
            pad_val (number or sequence): Same as :func:`impad`.

        Returns:
            ndarray: The padded image.
        """
        pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
        pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
        return self._impad(img, (pad_h, pad_w), pad_val)

    def _pad_img(self, results):
        if self.size is not None:
            padded_img = self._impad(results['img'], self.size)
        elif self.size_divisor is not None:
            padded_img = self._impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, size_divisor={}, pad_val={})'.format(
            self.size, self.size_divisor, self.pad_val)
        return repr_str


@PIPELINES.register_module
class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def _imnormalize(self, img, mean, std, to_rgb=True):
        img = img.astype(np.float32)
        if to_rgb:
            code = getattr(cv2, 'COLOR_{}2{}'.format('bgr'.upper(), 'rgb'.upper()))
            img = cv2.cvtColor(img, code)
        return (img - mean) / std

    def __call__(self, results):
        results['img'] = self._imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={}, to_rgb={})'.format(
            self.mean, self.std, self.to_rgb)
        return repr_str


@PIPELINES.register_module
class RandomCrop(object):
    """Random crop the image & bboxes & masks.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, results):
        img = results['img']
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        # filter out the gt bboxes that are completely cropped
        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            valid_inds = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (
                gt_bboxes[:, 3] > gt_bboxes[:, 1])
            # if no gt bbox remains after cropping, just skip this image
            if not np.any(valid_inds):
                return None
            results['gt_bboxes'] = gt_bboxes[valid_inds, :]
            if 'gt_labels' in results:
                results['gt_labels'] = results['gt_labels'][valid_inds]

            # filter and crop the masks
            if 'gt_masks' in results:
                valid_gt_masks = []
                for i in np.where(valid_inds)[0]:
                    gt_mask = results['gt_masks'][i][crop_y1:crop_y2,
                                                     crop_x1:crop_x2]
                    valid_gt_masks.append(gt_mask)
                results['gt_masks'] = valid_gt_masks

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={})'.format(
            self.crop_size)