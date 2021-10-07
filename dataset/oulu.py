import os
import cv2
import paddle.fluid as fluid
import numpy as np
import pickle
from .datasetbase import DatasetBase


class OULU(DatasetBase):
    def __init__(self,
                 img_prefix,
                 ann_file,
                 img_scale,
                 img_norm_cfg,
                 extra_aug=None,
                 test_mode=False,
                 val_mode=False):

        self.val_mode = val_mode
        super(OULU, self).__init__(
            img_prefix,
            ann_file,
            img_scale,
            img_norm_cfg,
            extra_aug,
            test_mode)

    def load_annotations(self, ann_file):
        pos_infos = []
        neg_infos = []
        img_infos = []
        self.kp_dict = dict()

        with open(ann_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
              item = line.strip().split(' ')
              if self.test_mode:
                    img_infos.append(dict(
                        img_path=item[0],
                        label=int(item[1])))
              if int(item[1]) == 0:
                    neg_infos.append(dict(
                        img_path=item[0],
                        label=int(item[1])))
              else:
                    pos_infos.append(dict(
                        img_path=item[0],
                        label=int(item[1])))

        pos_len = len(pos_infos)
        neg_len = len(neg_infos)
        print('total number of images: {} | pos: {}, neg: {}'.format(pos_len + neg_len, pos_len,
                                                                     neg_len))
        if self.test_mode or self.val_mode:
            return pos_infos + neg_infos
        else:
            if pos_len > neg_len:
                neg_infos = neg_infos * int(float(pos_len) / neg_len + 1)
                neg_infos = neg_infos[:pos_len]
            else:
                pos_infos = pos_infos * int(float(neg_len) / pos_len + 1)
                pos_infos = pos_infos[:neg_len]
            img_infos = pos_infos + neg_infos
            print('After balance total number of data: {} | pos: {}, neg: {}'.format(
                len(img_infos), len(pos_infos), len(neg_infos)))
            return img_infos

    def _get_patch(self, img, size=224):
        h, w, _ = img.shape
        center = [w/2, h/2]
        x1, x2 = max(0, int(center[0] - size/2)), min(w, int(center[0] + size/2))
        y1, y2 = max(0, int(center[1] - size/2)), min(h, int(center[1] + size/2))
        img = img[y1:y2, x1:x2, :]

        return img

    def train(self, batch_size=None):
        pos_infos = self.img_infos[:int(len(self.img_infos) / 2)]
        neg_infos = self.img_infos[int(len(self.img_infos) / 2):]
        assert len(pos_infos) == len(neg_infos)

        def reader():
            np.random.shuffle(pos_infos)
            np.random.shuffle(neg_infos)

            img_infos = []
            for i in range(len(pos_infos)):
                img_infos.append(pos_infos[i])
                img_infos.append(neg_infos[i])

            batch = []
            for img_info in img_infos:
                img_path = os.path.join(self.img_prefix,
                                        *[p for p in img_info['img_path'].split('/')[-8:]])
                label = img_info['label']
                # label = 0 if label > 1 else 1
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img = img.astype(np.float32)
                mask = np.zeros_like(img)
                img, mask, label = self.extra_aug(img, mask=mask, label=label)

                flip = True if np.random.rand() < 0.5 else False
                img, mask = self.img_transform(img, self.img_scale, mask=mask, flip=flip)

                if batch_size is None:
                    yield img, mask, label
                else:
                    batch.append([img, mask, label])
                    if len(batch) == batch_size:
                        yield batch
                        batch = []

        return reader

    def test(self):
        def reader():
            for img_info in self.img_infos:
                img_path = os.path.join(self.img_prefix,
                                        *[p for p in img_info['img_path'].split('/')[-8:]])
                label = img_info['label']
                # label = 0 if label > 1 else 1
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # img = self._get_patch(img, size=self.img_scale[0])

                img = self.img_transform(img, self.img_scale)
                yield img, label

        return reader




