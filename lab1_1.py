import math

import numpy as np
from PIL import Image,ImageOps

matrix= np.zeros((2000,2000,3),dtype=np.uint8)
z_buffer=np.full((2000,2000),np.inf)
matrix[0:2000,0:2000]= (144,203,194)
'''
for i in range(600):
    for j in range(800):
        matrix[i,j]=((i+j)/135)
'''
'''
def drow_line(matrix,x0,y0,x1,y1,color):
    step=1.0/150
    for t in np.arange(0,1,step):
        x=round((1.0-t)*x0+t*x1)
        y=round((1.0-t)*y0+t*y1)
        matrix[y,x]=color[0],color[1],color[2]
def x_loop(image,x0,y0,x1,y1,color):
    for x in range(x0,x1):
        t=(x-x0)/(x1-x0)
        y=round((1.0-t)*y0+t*y1)
        image[y,x]=color


def x_loop2(image,x0,y0,x1,y1,color):
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0,x1):
        t=(x-x0)/(x1-x0)
        y=round((1.0-t)*y0+t*y1)
        image[y,x]=color
def x_loop3(image,x0,y0,x1,y1,color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0,x1):
        t=(x-x0)/(x1-x0)
        y=round((1.0-t)*y0+t*y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
def x_loop4(image,x0,y0,x1,y1,color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y=y0
    dy = abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0,x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > 0.5):
            derror -= 1.0
            y += y_update
def x_loop5(image,x0,y0,x1,y1,color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y=y0
    dy =  2.0*abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0,x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror >(x1-x0)):
            derror -= 2.0*(x1-x0)*1.0
            y += y_update
for i in range(13):
    x0=100
    y0=100
    x1=int(100+95*np.cos((i*2*np.pi)/13))
    y1=int(100+95*np.sin((i*2*np.pi)/13))
    #drow_line(matrix,x0,y0,x1,y1,(255,255,255))
    x_loop5(matrix,x0,y0,x1,y1,(139,0,255))
f= open("C:\\Users\\igort\\Downloads\\Telegram Desktop\\model_1.obj")
vec=[]
lis=[]
for s in f:
    split=s.split()
    if(split[0]=="v"):
        vec.append([float(x) for x in split[1:]])
    if (split[0]=='f'):
        lis.append([int(x.split('/')[0]) for x in split[1:]])
print(lis)
for vertex in vec:
    matrix[int(10000*vertex[1])+1000,int(10000*vertex[0])+1000]= (139,0,255)
for face in lis:
    x0=10000*vec[face[0]-1][0]+1000
    y0=10000*vec[face[0]-1][1]+1000
    x1=10000*vec[face[1]-1][0]+1000
    y1=10000*vec[face[1]-1][1]+1000
    x2=10000*vec[face[2]-1][0]+1000
    y2=10000*vec[face[2]-1][1]+1000
    x_loop5(matrix,int(x0),int(y0),int(x1),int(y1),(139,0,255))
    x_loop5(matrix, int(x1), int(y1), int(x2), int(y2), (139, 0, 255))
    x_loop5(matrix, int(x2), int(y2), int(x1), int(y1), (139, 0, 255))
image = Image.fromarray(matrix, mode="RGB")
image=ImageOps.flip(image)
image.save("image2.png")
'''
def bercent(x,y,x0,y0,x1,y1,x2,y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2))/((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2))/((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2
def draw_triange(matrix,z_buffer,x0, y0, x1, y1, x2, y2,color):
    xmin= math.floor(min(x0,x1,x2))
    if(xmin<0):
        xmin=0
    ymin= math.floor(min(y0,y1,y2))
    if(ymin<0):
        ymin=0
    xmax= math.ceil(max(x0,x1,x2))
    if(xmax>matrix.shape[1]):
        xmax=matrix.shape[1]
    ymax= math.ceil(max(y0,y1,y2))
    if(ymax>matrix.shape[0]):
        ymax=matrix.shape[0]
    for x in range(xmin,xmax):
        for y in range(ymin,ymax):
            l0,l1,l2=bercent(x,y, x0, y0, x1, y1, x2, y2)
            if(l0>=0 and l1>=0 and l2>=0):
                z=10*z0+l1*z1+l2*z2
                if z<z_buffer[y,x]:
                    z_buffer[y,x]=z
                    matrix[y,x]=color
def norm(x0,y0,z0,x1,y1,z1,x2,y2,z2):
    x=(y1-y2)*(z1-z0)-(y1-y0)*(z1-z2)
    y=((x1-x2)*(z1-z0)-(x1-x0)*(z1-z2))
    z=(x1-x2)*(y1-y0)-(x1-x0)*(y1-y2)
    return x,y,z
def ligth(x,y,z):
    cosl=z/(math.sqrt(x**2+y**2+z**2))
    return cosl
f= open("C:\\Users\\igort\\Downloads\\Telegram Desktop\\model_1.obj")
vec=[]
lis=[]
for s in f:
    split=s.split()
    if(split[0]=="v"):
        vec.append([float(x) for x in split[1:]])
    if (split[0]=='f'):
        lis.append([int(x.split('/')[0]) for x in split[1:]])
for vertex in vec:
    matrix[int(10000*vertex[1])+1000,int(10000*vertex[0])+1000]= (139,0,255)
for face in lis:
    x0=10000*vec[face[0]-1][0]+1000
    y0=10000*vec[face[0]-1][1]+1000
    z0= 10000 * vec[face[0] - 1][2] + 1000
    x1=10000*vec[face[1]-1][0]+1000
    y1=10000*vec[face[1]-1][1]+1000
    z1 = 10000 * vec[face[1] - 1][2] + 1000
    x2=10000*vec[face[2]-1][0]+1000
    y2=10000*vec[face[2]-1][1]+1000
    z2 = 10000 * vec[face[2] - 1][2] + 1000
    xn,yn,zn=norm(x0,y0,z0,x1,y1,z1,x2,y2,z2)
    cosl=ligth(xn,yn,zn)
    if cosl<0:
        draw_triange(matrix,z_buffer,x0, y0, x1, y1, x2, y2, (-255*cosl,-0*cosl,-255*cosl))
image = Image.fromarray(matrix, mode="RGB")
image=ImageOps.flip(image)
image.save('image7.png')