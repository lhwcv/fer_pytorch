import cv2
import os, sys
import torch
from fer_pytorch.face_detect import  MTCNN
import numpy
from IPython import embed
import pandas as pd
# from dataset.base import FER_DatasetTest
# from dataset.fer2013 import FER2013
import h5py
import torch.utils.data as data
from IPython import embed
from PIL import Image

import time

class affine_img_with_five_landmark:
    imgSize = [112, 96]           
    coord5point = [[30.2946, 51.6963],
                   [65.5318, 51.6963],
                   [48.0252, 71.7366],
                   [33.5493, 92.3655],
                   [62.7299, 92.3655]]

    def __init__(self):
        print("cuda : {}".format(torch.cuda.is_available()))
        self.OutSizeMapping()
        self.mtcnn = MTCNN(
        image_size = 224,
        min_face_size = 40,
#         device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device=torch.device( 'cpu')
    )
    
    def OutSizeMapping(self, OutSize = 255):
        sw, sh = float(OutSize) / self.imgSize[0], float(OutSize) / self.imgSize[1]
        self.newcoord5point = [ [sh * x[0],sw * x[1]] for x in self.coord5point]
        self.imgSize = [OutSize, OutSize]
        
    def transformation_from_points(self, p, q):

        pad = numpy.ones(5)
        p = numpy.insert(p, 2, values=pad, axis=1)
        q = numpy.insert(q, 2, values=pad, axis=1)

        # 最小二乘
        M1 = numpy.linalg.inv(p.T*p)
        M2 = p.T*q
        M = M1*M2
        return M.T
        
    def warp_im(self, img_im, orgi_landmarks,tar_landmarks):
        pts1 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in orgi_landmarks]))
        pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in tar_landmarks]))
        M = self.transformation_from_points(pts1, pts2)
        dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
        return dst

    def FaceAlign(self, img_im, face_landmarks):
        dst = self.warp_im(img_im, face_landmarks, self.newcoord5point)
        crop_im = dst[0:self.imgSize[0], 0:self.imgSize[1]]
        return crop_im

    def face_affiner(self, img, bbox_label=None):
        imgSize = [112, 96]
        h, w = img.shape[:2]

        if img.dtype != 'uint8': # check whether image or not 
            raise RuntimeError('dtype of numpy array is not uint8!!! check it !!!')

        bboxs, scores, landmarks = self.mtcnn.detect(img, landmarks=True)
        if bboxs is not None and len(bboxs)!=0:
            if bbox_label is not None:
                y0, x0, y1, x1 = bbox_label
                best_area = 100000000000000
                best_id = -1
                for i, box in enumerate(bboxs):
                    # 计算外接矩
                    area = (max(x1, box[2]) - min(x0, box[0]))*(max(y1, box[3]) - min(y0, box[1]))
                    if area < best_area: 
                        best_area = area
                        best_id = i
            else:
                best_id = 0
        else: # bboxs 为空，检测失败

            return img
        
        if best_id == -1:
            embed()
        img = self.FaceAlign(img,landmarks[best_id].tolist())
        return img
    
    
if __name__ =='__main__':

    affiner = affine_img_with_five_landmark()
    
    img = cv2.imread('../../exmaples/data/1.jpg')
    dst=affiner.face_affiner(img)
    cv2.imwrite('../../exmaples/data/1_affine.jpg', dst)
