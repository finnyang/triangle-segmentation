import cv2
import numpy as np
import numpy.random as nr
import math

img=cv2.imread('/home/yang/triangle-segmentation/UNet/test_img.tif',-1)
[h,w]=img.shape
points=nr.randint(1,h-1,(2))
while(1):
    if (img[points[0]][points[1]] == 0) and(img[points[0]-1][points[1]-1]==255 or img[points[0]-1][points[1]+1]==255 or img[points[0]+1][points[1]-1] == 255 or img[points[0]+1][points[1]+1] == 255):
        break;
    else:
        points=nr.randint(1,h-1,(2))
r=nr.randint(50,int(h))

for i in range(h):
    for j in range(w):
        if math.sqrt(math.pow(points[0]-i,2)+math.pow(points[1]-j,2)<r) and img[i][j]!=255:
            img[i][j] = 128
cv2.imwrite('/home/yang/triangle-segmentation/UNet/test_img_circle.tif',img)
