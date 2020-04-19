import numpy as np
import tensorflow as tf

from pycocotools.coco import COCO
from tqdm import tqdm

from .base_dataset import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class DeepFashion2Dataset(BaseDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)

        if 'category_to_pairs' in self.coco.dataset:
            self.prepare_train_data()

        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def prepare_train_data(self):
        self.data_indices = []
        for key, values in self.coco.dataset['category_to_pairs'].items():
            for pair_id, (shop, user) in values.items():
                self.data_indices.append([key, pair_id, 'shop'])
                self.data_indices.append([key, pair_id, 'user'])

        return

    def __prepare_training_dataset(self, img_infos):
        pairs = []
        for cat_id, values in tqdm(self.coco.dataset['category_to_pairs'].items()):
            for pair_id, (shop, user) in values.items():
                if len(shop) == 0:
                    shuffled_user = np.asarray(user)[np.random.permutation(len(user))]
                    for i in range(len(shuffled_user) - 1):
                        user_id = [j for j in range(len(self.img_ids)) if
                                   img_infos[j]['id'] == shuffled_user[i]['image_id']]
                        assert len(user_id) == 1
                        user_id = user_id[0]

                        user_id2 = [j for j in range(len(self.img_ids)) if
                                   img_infos[j]['id'] == shuffled_user[i + 1]['image_id']]
                        assert len(user_id2) == 1
                        user_id2 = user_id2[0]

                        pairs.append([user_id, user_id2])
                elif len(user) == 0:
                    shuffled_shop = np.asarray(shop)[np.random.permutation(len(shop))]
                    for i in range(len(shuffled_shop) - 1):
                        shop_id = [j for j in range(len(self.img_ids)) if
                                   img_infos[j]['id'] == shuffled_shop[i]['image_id']]
                        assert len(shop_id) == 1
                        shop_id = shop_id[0]

                        shop_id2 = [j for j in range(len(self.img_ids)) if
                                    img_infos[j]['id'] == shuffled_shop[i + 1]['image_id']]
                        assert len(shop_id2) == 1
                        shop_id2 = shop_id2[0]

                        pairs.append([shop_id, shop_id2])
                else:
                    shuffled_shop = np.asarray(shop)[np.random.permutation(len(shop))]
                    shuffled_user = np.asarray(user)[np.random.permutation(len(user))]

                    for i in range(len(shuffled_shop)):
                        shop_id = [j for j in range(len(self.img_ids)) if img_infos[j]['id'] == shuffled_shop[i]['image_id']]
                        assert len(shop_id) == 1
                        shop_id = shop_id[0]

                        if len(shuffled_user) > i:
                            user_i = i
                        else:
                            user_i = np.random.choice(np.arange(len(shuffled_user)), 1)[0]

                        user_id = [j for j in range(len(self.img_ids)) if img_infos[j]['id'] == shuffled_user[user_i]['image_id']]
                        assert len(user_id) == 1
                        user_id = user_id[0]

                        pairs.append([shop_id, user_id])

        return pairs

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)

                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def __py_func_map(self, idx):
        cat_id = str(np.core.defchararray.decode(idx[0].numpy()))
        pair_id = str(np.core.defchararray.decode(idx[1].numpy()))
        img_type = str(np.core.defchararray.decode(idx[2].numpy()))

        if img_type == 'shop':
            img_types = self.coco.dataset['category_to_pairs'][cat_id][pair_id][0]
            if len(img_types) == 0:
                img_types = self.coco.dataset['category_to_pairs'][cat_id][pair_id][1]
        else:
            img_types = self.coco.dataset['category_to_pairs'][cat_id][pair_id][1]
            if len(img_types) == 0:
                img_types = self.coco.dataset['category_to_pairs'][cat_id][pair_id][0]

        img = np.random.choice(img_types, 1)[0]
        cur_id = self.img_ids.index(img['image_id'])
        img_info = self.img_infos[cur_id]
        ann_info = self.get_ann_info(cur_id)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[cur_id]
        self.pre_pipeline(results)
        img = self.pipeline(results)['img']

        label = '%s_%s' % (cat_id, pair_id)

        return img, label

    def prepare_train_img(self, idx):
        return tf.py_function(self.__py_func_map, [idx], [tf.float64, tf.string])

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]

        self.pre_pipeline(results)

        return self.pipeline(results)