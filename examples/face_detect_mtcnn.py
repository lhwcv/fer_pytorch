import cv2
import os
import torch
from fer_pytorch.face_detect import  MTCNN


def main():
    img_path = os.path.join(os.path.dirname(__file__), './data/1.jpg')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mtcnn = MTCNN(
        image_size = 224,
        min_face_size = 80,
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

if __name__ =='__main__':
    main()

