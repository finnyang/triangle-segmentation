import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
from PIL import Image
import os
import time

def in_tangle(A,B,C,P):
    MA=[P[0]-A[0],P[1]-A[1]]
    MB=[P[0]-B[0],P[1]-B[1]]
    MC=[P[0]-C[0],P[1]-C[1]]
    a=MA[0]*MB[1]-MA[1]*MB[0]
    b=MB[0]*MC[1]-MB[1]*MC[0]
    c=MC[0]*MA[1]-MC[1]*MA[0]
    if(a<0 and b<0 and c<0) or (a>0 and b>0 and c>0):
        return True
    else:
        return False
def get3point(max_value):
    points=nr.randint(0,max_value,(3,2))
    return points
def randpoint(max_value,num):
    points=nr.randint(0,max_value,(num,2))
    return points
def test():
    num = 10000
    value = 10
    retangle = get3point(128)
    x = retangle[:, 0]
    x = np.append(x, x[0])
    y = retangle[:, 1]
    y = np.append(y, y[0])

    points = randpoint(128, num)
    # points=np.meshgrid(128)
    A = retangle[0, :]
    B = retangle[1, :]
    C = retangle[2, :]

    point_in_x = []
    point_in_y = []
    point_out_x = []
    point_out_y = []

    for i in range(num):
        if (in_tangle(A, B, C, points[i, :])):
            point_in_x.append(points[i, 0])
            point_in_y.append(points[i, 1])
        else:
            point_out_x.append(points[i, 0])
            point_out_y.append(points[i, 1])
    plt.figure()
    plt.scatter(point_in_x, point_in_y, s=value, c='r')
    plt.scatter(point_out_x, point_out_y, s=value, c='b')
    print len(point_out_x)
    print len(point_in_x)
    plt.plot(x, y)
    plt.show()
def create_sample(image_size,is_show=0):
    in_num=0
    out_num=0
    image=np.zeros(image_size)
    height = image_size[0]
    width=image_size[1]
    height_s=range(0,height)
    width_s=range(0,width)
    [x,y]=np.meshgrid(height_s,width_s)
    retangle = get3point(image_size[0])
    A = retangle[0, :]
    B = retangle[1, :]
    C = retangle[2, :]
    for i in height_s:
        for j in width_s:
            if (in_tangle(A, B, C, [x[i,j],y[i,j]])):
                image[i,j]=255
                in_num+=1
            else:
                image[i,j]=0
                out_num+=1
    if is_show:
        plt.figure()
        plt.imshow(image)
        plt.show()
    return image,in_num*1.0/(in_num+out_num)
def process(image_num,dir,file):
    fid=open(os.path.join(dir,file),'w')
    for i in range(image_num):
        print "process %d"%(i)
        path=os.path.join(dir,str(i)+'.tif')
        [image, rate] = create_sample([128, 128])
        image=image.astype(np.uint8)
        im=Image.fromarray(image)
        im.save(path)
        fid.write(path+' '+str(rate)+'\n')
    fid.close()
def get(num,dir,file):
    if os.path.exists(dir):
        pass
    else:
        os.mkdir(dir)
    start = time.clock()
    process(num,dir,file)
    end = time.clock()
    print end - start
if __name__=="__main__":
    get(10, '/home/yang/triangle-segmentation/temp', 'a.txt')
    pass