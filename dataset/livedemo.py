import os
import cv2
import paddle.fluid as fluid
import numpy as np
import pickle
import random
from .datasetbase import DatasetBase
from facenet_pytorch import MTCNN
from PIL import Image


class LIVEDEMO(DatasetBase):
    def __init__(self,
                 img_prefix,
                 ann_file,
                 img_scale,
                 img_norm_cfg,
                 videoPath,
                 extra_aug=None):

        super(LIVEDEMO, self).__init__(
            img_prefix,
            ann_file,
            img_scale,
            img_norm_cfg,
            extra_aug)

        self.videoPath = videoPath

    def get_face(self,img):
        mtcnn = MTCNN(image_size=224,post_process=False, margin=0)
        # frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(img)

        face = mtcnn(frame)
        face = face.permute(1, 2, 0).int().numpy()

        return face

    def test(self):
        def reader():
            cap= cv2.VideoCapture(self.videoPath)
            p=0
            while (p<64):
              p=p+1
              ret, frame = cap.read()
              if ret:
                try:
                  face = self.get_face(frame)
                except:
                  continue
                img = self.img_transform(face, self.img_scale)
                label = 1
                yield img, label
        return reader
