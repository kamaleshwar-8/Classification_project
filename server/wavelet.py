import numpy as np
import pywt
import cv2

def w2d(img,mode='haar',level=1):
    imarray=img
    imarray=cv2.cvtColor(imarray,cv2.COLOR_BGR2GRAY)
    imarray=np.float32(imarray)
    imarray/=255;
    coeff=pywt.wavedec2(imarray,mode,level=level)
    coeffh=list(coeff)
    coeffh[0]*=0;
    imarrayh=pywt.waverec2(coeffh,mode)
    imarrayh*=255
    imarrayh=np.uint8(imarrayh)
    return imarrayh