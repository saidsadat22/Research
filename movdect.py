#!/usr/bin/env python3


# prerequisites:  opencv matplotlib numpy

import cv2
import numpy as np
import os
import sys
import time
import argparse
import subprocess
import shutil

# a bat



cap = cv2.VideoCapture("SLBE_20230904_095801.mp4")

import cv2
import numpy as np


def equalize_hist(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply histogram equalization to the V channel
    hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])

    # Convert the HSV image back to RGB color space
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Save the image
    return rgb_image

def adjust_hsv(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjust the hue, saturation, and value of the image
    hsv_image[:, :, 0] += 59
    hsv_image[:, :, 1] *= 0
    hsv_image[:, :, 2] += 0

    # Convert the image back to RGB color space
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Save the image
    return rgb_image


def readFrame(cap):
    ret, frame = cap.read()
    for i in range(10):
        ret, frame = cap.read()
        if ret: break
    if ret:
        #frame = frame[-310:-210,250:450,:]
        frame = frame[550:,:2000,...]
        #frame[:,:,2] = cv2.absdiff(frame[:,:,1],frame[:,:,0])
        #frame[:,:,1] = cv2.equalizeHist(frame[:,:,1])
        return frame/255.0
    else:
        return None

# calculate difference of gaussian on frame differenced method
def dog(frame1, frame2, frame3, ksize=(5, 5), sigma1=1.0, sigma2=4.0):
    # frame differencing
    diff12 = cv2.absdiff(frame2, frame1)
    # gaussians of frame differencing
    god12a = cv2.GaussianBlur(diff12, ksize, sigma1)
    god12b = cv2.GaussianBlur(diff12, ksize, sigma2)
    # difference of gaussian
    dog1 = god12b-god12a
    dog2 = cv2.absdiff(god12b, god12a)
    dog=np.concatenate((frame1,diff12,dog1,dog2,frame2
    ), axis=0)
    return dog

def dog1(frame,sigma1=3,sigma2=5):
    dog = cv2.GaussianBlur(frame, (5, 5), sigma1) - cv2.GaussianBlur(frame, (5, 5), sigma2)
    return dog/dog.max()



imgs = []
imgsum = None
while True:

    img = readFrame(cap)
    if img is None: break
    imgs += [ img[100:,:500][121:256,:] ]
    if imgsum is None:
        imgsum = img
    imgsum += img



imgmedian = np.median(imgs, axis=0)


def labelimage(img, str):
    
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
      
    # org
    org = (0, 20)
    org1 = (1,21)
      
    # fontScale
    fontScale = .5
       
    # Blue color in BGR
    color = (255,255,255)
    color1 = (0,0,0)
      
    # Line thickness of 2 px
    thickness = 1

    # convert to bgr 
    if len(img.shape) == 2:
        imgout = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2BGR)
    else:
        imgout = img.copy()
    
    imgout = cv2.resize(imgout.copy(),(int(imgout.shape[1]*0.75) , int(imgout.shape[0]*0.75)))

    
    # put text on the image
    imgout = cv2.putText(imgout, str, org1, font, 
                   fontScale, color1, thickness, cv2.LINE_AA,bottomLeftOrigin = False )   
    imgout = cv2.putText(imgout, str, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA,bottomLeftOrigin = False )
    return imgout



i = 4
while True:
    imgsav = [ labelimage(imgs[i], f"Original image imgs[i] i={i}") ]

    imgmedian = np.median(imgs[i-4:i+5], axis=0)
    imgsav += [ labelimage(imgmedian[...,1],"median of 9 frames imgs[i-4:i+5]") ]
    
    
    print(imgsav)
    print(f"imgsav[-1].shape = {imgsav[-1].shape}")
    

    
    
    img1 = cv2.absdiff(imgs[i],imgmedian)
    #img1 = dog1(cv2.absdiff(imgs[i],imgmedian))
    #img2 = dog1(cv2.absdiff(imgs[i+1],imgmedian))
    #imgdiff = cv2.absdiff(img1, img2)
    imgdiff = img1
    imgdiff = np.uint8(imgdiff*255)
    imgdiff = cv2.cvtColor(imgdiff, cv2.COLOR_BGR2GRAY)
    imgsav += [ labelimage(imgdiff.copy()/imgdiff.max(), "difference between median and org") ]


    #imgdiff = cv2.threshold(imgdiff, np.percentile(imgdiff,99.9), 255, cv2.THRESH_BINARY)[1]
    imgdiff = cv2.threshold(imgdiff, 40, 255, cv2.THRESH_BINARY)[1]
    imgsav += [ labelimage(imgdiff.copy(),"thresholded") ]

    imgdiff = cv2.erode(imgdiff, None, iterations=1)
    imgsav += [labelimage(imgdiff.copy(),"eroded") ]
    imgdiff = cv2.dilate(imgdiff, None, iterations=3)
    imgsav += [ labelimage(imgdiff.copy(),"dilated") ]

    #imgdiff = cv2.medianBlur(imgdiff, 3)
    imgdiff = cv2.GaussianBlur(imgdiff, (5, 5), 3)
    imgdiff = cv2.threshold(imgdiff, np.percentile(imgdiff,90), 255, cv2.THRESH_BINARY)[1]
    imgdif2 = cv2.dilate(imgdiff, None, iterations=3)
    imgdog = cv2.absdiff(imgdiff, imgdif2)
    orgi = imgs[i]
    newi = imgs[i].copy()
    newi[imgdog>0] = [0,0,1]
    imgsav += [ labelimage(newi,"borders are | dilate(threshold(blur(img))) - threshold(blur(img)) |") ]
    imgmask = cv2.GaussianBlur(imgdif2.copy(), (11,11), 7)/255.0
    
    imgmask = np.stack( [ imgmask ]*3 , axis = 2)
    imgmean = imgs[i].copy()
    imgmean[...,0] = imgmean[...,0].mean()
    imgmean[...,1] = imgmean[...,1].mean()
    imgmean[...,2] = imgmean[...,2].mean()
    maskresult = imgs[i]*imgmask + np.mean(imgs[i])*(1.0-imgmask)
    maskresult = imgs[i].copy()
    maskresult[...,0] = imgs[i][...,0]*imgmask[...,0] #+ imgmean[...,0]*(1.0-imgmask[...,0])
    maskresult[...,1] = imgs[i][...,1]*imgmask[...,1] #+ imgmean[...,1]*(1.0-imgmask[...,1])
    maskresult[...,2] = imgs[i][...,2]*imgmask[...,2] #+ imgmean[...,2]*(1.0-imgmask[...,2])
    
    
    imgsav += [ labelimage(maskresult,"masked with mean" ) ]
    #newi[imgdiff==0] = np.mean(imgs[i], axis=(0,1))
    cv2.imshow("DOG", np.concatenate(imgsav)) #imgs[i]) #*imgdiff/255) #*imgs[i]/255) #np.concatenate((imgfrombg), axis=0))
    key = cv2.waitKey(0)
    if key in (27, ord('q')): break
    # backspace goes back one frame
    if key == 8 or key==81:
        i -= 1
        if i < 4: i = 4
    # space goes forward one frame
    if key == 32 or key==83:
        i += 1
        if i >= len(imgs)-4: i = len(imgs)-3
    
cap.release()
cv2.destroyAllWindows()


