import os, sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy
from IPython import embed
import time

# expect 5 landmark, top-left and bottom-right coord on imgSize [112,96]
ImgSize = [112, 96]           
Coord7Point = [[30.2946, 51.6963],
               [65.5318, 51.6963],
               [48.0252, 71.7366],
               [33.5493, 92.3655],
               [62.7299, 92.3655],
               [      0,       0],
               [    112,      96]]


def OutSizeRescale(OutSize = 255):
    global ImgSize
    global Coord7Point
    sw, sh = float(OutSize) / ImgSize[0], float(OutSize) / ImgSize[1]
    Coord7Point = [ [sh * x[0],sw * x[1]] for x in Coord7Point]
    ImgSize = [OutSize, OutSize]


def TransformationFromPoints(p, q):

    pad = numpy.ones(p.shape[0])
    p = numpy.insert(p, 2, values=pad, axis=1)
    q = numpy.insert(q, 2, values=pad, axis=1)

    # 最小二乘
    # M1 = numpy.linalg.inv(p.T*p)
    M1 = numpy.linalg.pinv(p.T*p)  # pseudo inverse 
    M2 = p.T*q
    M = M1*M2
    return M.T


def WarpIm(img_im, orgi_landmarks, tar_landmarks):
    # embed()
    pts1 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = TransformationFromPoints(pts1, pts2)
    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
    return dst


def FaceAlign(img_im, face_landmarks, outimgsize, use_bbox=True):
    if not use_bbox:
        global Coord7Point
        Coord7Point = Coord7Point[:5]
    assert len(face_landmarks) == len(Coord7Point)
    OutSizeRescale(outimgsize)
    dst = WarpIm(img_im, face_landmarks, Coord7Point)
    crop_im = dst[0:ImgSize[0], 0:ImgSize[1]]
    return crop_im


