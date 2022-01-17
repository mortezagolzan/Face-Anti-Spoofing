import paddle.fluid as fluid
import cv2
import numpy as np
from models.scan import SCAN
from utils.runner import Runnner
from facenet_pytorch.models.mtcnn import MTCNN
from dataset.livedemo import LIVEDEMO
import time

model_cfg = dict(
    backbone=dict(
        depth=18,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1),
    neck=dict(
        norm_cfg=dict(type='IN')),
    head=dict(
        depth=18,
        out_indices=(3,),
        norm_cfg=dict(type='BN'),
        dropout=0.5),
    train_cfg=dict(
        w_cls=5.0,
        w_tri=1.0,
        w_reg=5.0,
        with_mask=False),
    test_cfg=dict(
        thr=0.5),
    pretrained='./pretrained/resnet18-torch',
)

checkpoint_cfg = dict(
    work_dir='./work_dir/test/',
    load_from='./work_dir/ff_add_val/Best_model',
    save_interval=30000,
    eval_interval=200,
    log_interval=10,
    eval_type='acc'
)

optimizer_cfg = dict(
    lr=0.0005,
    type='Adam',
    warmup_iter=1000,
    decay_epoch=[2,3,5],
    decay=0.3,
    regularization=0.0005,
)

extra_aug = dict(
    photo_metric_distortion=dict(
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.8, 1.2),
        hue_delta=16),
    random_erasing=dict(
        probability=0.5,
        area=(0.01, 0.03),
        mean=(80, 80, 80)),
    random_cutout=dict(
        probability=0.5,
        max_edge=20),
    ramdom_rotate=dict(
        probability=0.5,
        angle=30),
    ramdom_crop=dict(
        probability=0.5,
        w_h=(0.12, 0.12))
)


test_dataset = LIVEDEMO(
    img_scale=(224, 224),
    img_norm_cfg=dict(mean=(100, 100, 100), std=(80, 80, 80)),
    img_prefix='',
    ann_file='',
    extra_aug=dict()
)

with fluid.dygraph.guard():
    model = SCAN(**model_cfg)
    
runner = Runnner(
            model,
            test_dataset,
            batch_size=1,
            checkpoint_config=checkpoint_cfg,
            optimizer_config=optimizer_cfg)

mtcnn = MTCNN(keep_all=True)


def mtcnn_detect(img: np.ndarray,color) -> np.ndarray:
    boxes, probs = mtcnn.detect(img)
    for box in boxes:
        x_left = min(box[0], box[2])
        x_right = max(box[0], box[2])
        y_left = min(box[1], box[3])
        y_right = max(box[1], box[3])
        if color =='green':
          cv2.putText(img, 'Real', 
                      (int(x_left), int(y_left) - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      4, 
                      (0, 255, 0), 3)
          img = cv2.rectangle(img, (x_left, y_left), (x_right, y_right), 
                              (0, 255, 0), 4)
        else:
          cv2.putText(img, 'Fake', 
                      (int(x_left), int(y_left) - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      4, 
                      (0, 0, 255), 3)
          img = cv2.rectangle(img, (x_left, y_left), (x_right, y_right), 
                    (0, 0, 255), 4)
    return img

t0 = time.time()
pred = runner.test_liveDemo(thr=0.7896)
t1 = time.time()

print('Total Time For Execution is: ',t1-t0)

phases = [pred[x:x+5] for x in range(0, len(pred),5)]

cap= cv2.VideoCapture('/content/rooz_roozphone_fake.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 30.0, (1280,720))
p=0
i=1
while (p<64):
  p+=1
  if i%5!=0:
    ph=0
  else:
    ph+=1
  color = 'green' if phases[ph].count(1)>=3 else 'red'
  ret, frame = cap.read()
  if ret:
    try:
      changedImg = mtcnn_detect(frame,color)
    except:
      continue
    out.write(changedImg)
    i+=1

cap.release()
out.release()
cv2.destroyAllWindows()