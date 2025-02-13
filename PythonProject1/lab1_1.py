import numpy as np
from PIL import Image,ImageOps

matrix= np.zeros((2000,2000,3),dtype=np.uint8)

matrix[0:2000,0:2000]= (144,203,194)
'''
for i in range(600):
    for j in range(800):
        matrix[i,j]=((i+j)/135)
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
'''for i in range(13):
    x0=100
    y0=100
    x1=int(100+95*np.cos((i*2*np.pi)/13))
    y1=int(100+95*np.sin((i*2*np.pi)/13))
    #drow_line(matrix,x0,y0,x1,y1,(255,255,255))
    x_loop5(matrix,x0,y0,x1,y1,(139,0,255))'''
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
