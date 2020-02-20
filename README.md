# fer_pytorch
Face Expression Recognition in Pytorch 
### Install
```
git clone https://github.com/lhwcv/fer_pytorch
cd fer_pytorch && python setup.py install
```

### Baseline Performance

1.**FER2013**

model| im_size| acc| recall| F1
|---|---|:---:| :---: | :---:
res50| 224| -|-|-
mobile_v2|224|-|-|-

2.**ExpW**

model| im_size| acc| recall| F1
|---|---|:---:| :---: | :---:
res50| 224| -|-|-
mobile_v2|224|-|-|-


3.**AffectNet**

model| im_size| acc| recall| F1
|---|---|:---:| :---: | :---:
res50| 224| -|-|-
mobile_v2|224|-|-|-

###  Tools
#### Face detect by MTCNN
See `examples/face_detect_mtcnn.py`
```
import cv2
import os
import torch
from fer_pytorch.face_detect import  MTCNN

mtcnn = MTCNN(
        image_size = 224,
        min_face_size = 40,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    )
bboxs, scores, landmarks = mtcnn.detect(img, landmarks=True)
for box, score, points in zip(bboxs,scores,landmarks):
        box[2] = box[2] - box[0] # w
        box[3] = box[3] - box[1] # h
        cv2.rectangle(img, tuple([int(v) for v in box.tolist()]), (255,0,0),3,16)
        for p in points:
            cv2.circle(img, tuple([int(v) for v in p.tolist()]),5, (0,255,0),3,16)
img_path = img_path[:-4]+'_det.jpg'
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite(img_path, img)
```
