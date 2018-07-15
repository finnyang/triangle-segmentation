import cv2
import numpy as np
import os

fid=open('/home/yang/triangle-segmentation/data/reserve.txt')
lines=fid.readlines()
for line in lines:
    line=line.strip().split()[0]
    image=cv2.imread(line,-1)
    (h,w)=image.shape
    for i in range(h):
        for j in range(w):
            if(image[i,j] == 255):
                image[i,j]=1
    path=line.split('/')[-1]
    path=os.path.join('/home/yang/triangle-segmentation/data/masks',path)
    cv2.imwrite(path,image)
