# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:07:19 2016

@author: Saber
"""
import math
import cv2
import numpy as np
#import matplotlib
import skimage 
def transform(x,y,ang,s0,s1):
    '''
    @x:x point
    @y:y point
    @ang:angle
    @s0:size of original image
    @s1:size of target image
    '''
    x0 = x - s0[1]/2
    y0 = y - s0[0]/2
    xx = x0*math.cos(ang) - y0*math.sin(ang) + s1[1]/2
    yy = x0*math.sin(ang) + y0*math.cos(ang) + s1[0]/2
    return xx,yy
def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
def guard(x, N):
    b=[]
    for i in x:
        if i<1:i=1
        if i>N:i=N
        b.append(i)
    return b
'''
传入参数的时候应该把大小调整过
'''
def align(img, f5pt, crop_size, ec_mc_y, ec_y):
    '''
    @img:某一张图片的
    @f5pt:
    @crop_size:
    @ec_mc_y:
    @ec_y:
    '''
#    if img.shape[0]<img.shape[1]:
#        img=rotate_about_center(img,90)
#    f5pt =f5pt
    ang_tan = (f5pt[0][1]-f5pt[1][1])/float(f5pt[0][0]-f5pt[1][0])
    ang=math.atan(ang_tan)/math.pi*180
    img_rot=rotate_about_center(img,ang)
    imgh = img.shape[0]
    imgw = img.shape[1]
    #取眼睛的中点
    x=(f5pt[0][0]+f5pt[1][0])/2.0
    y=(f5pt[0][1]+f5pt[1][1])/2.0
    
    ang = -ang/180.0*math.pi
    xx, yy= transform(x, y, ang, img.shape, img_rot.shape)
    eyec=np.round([xx, yy])
#    cv2.circle(img_rot,(int(eyec[0]),int(eyec[1])),2,(0,255,255),2)
#    cv2.imshow('res',img_rot)    
#    cv2.waitKey(0)
    #嘴巴的中点
    x=(f5pt[3][0]+f5pt[4][0])/2.0
    y=(f5pt[3][1]+f5pt[4][1])/2.0
    
    ang = -ang/180.0*math.pi
    xx, yy= transform(x, y, ang, img.shape, img_rot.shape)
    
    mouthc=np.round([xx, yy])
#    cv2.circle(img_rot,(int(mouthc[0]),int(mouthc[1])),2,(255,255,255),2)
#    print img_rot.shape
#    cv2.imshow('rot',img_rot)    
#    cv2.waitKey(0)
    resize_scale = ec_mc_y/float(mouthc[1]-eyec[1])
    img_resize=cv2.resize(img_rot,(int(img_rot.shape[0]*resize_scale),int(img_rot.shape[1]*resize_scale)))
#    print img_resize.shape
#    cv2.imshow('res',img_resize)    
#    cv2.waitKey(0)
    eyec2 = (eyec -[img_rot.shape[1]/2.0,img_rot.shape[0]/2.0]) * resize_scale + [img_resize.shape[1]/2.0,img_resize.shape[0]/2.0]
    eyec2 = np.round(eyec2)
    
#    img_resize=rotate_about_center(img_resize,90)

    

    img_crop = np.zeros((crop_size, crop_size, img.shape[2]),dtype='uint8')
    crop_y = eyec2[1] - ec_y
    crop_y_end = crop_y + crop_size 
    crop_x = eyec2[0]-np.floor(crop_size/2.0)
    crop_x_end = crop_x + crop_size
    box = guard([crop_x,crop_x_end,crop_y,crop_y_end], 500)
#    print box
#    cv2.imshow('crop',img_resize[box[2]:box[3],box[0]:box[1],:])    
#    cv2.waitKey(0)
#    print img_crop.shape
    temp_row=img_resize.shape[0]
    temp_col=img_resize.shape[1]
    if img_resize.shape[0]<box[3] :
        temp_row=box[3]
    if img_resize.shape[1]<box[1]:
        temp_col=box[1]
    img_temp = np.zeros((temp_row, temp_col, img.shape[2]),dtype='uint8')
    img_temp[0:img_resize.shape[0],0:img_resize.shape[1],:]=img_resize[:,:,:]
#    print 'tempsize',img_temp.shape
    img_crop[:,:,:] = img_temp[box[2]:box[3],box[0]:box[1],:]
#    cv2.imshow('crop',img_crop)    
#    cv2.waitKey(0)
    cropped = img_crop/255.0
#    cv2.imshow('crop_2',cropped)    
#    cv2.waitKey(0)
    cropped=cropped[:,:,[2,1,0]]
    return img_resize, eyec2, cropped, resize_scale
#feature1=np.random.randn(4,1)
#feature2=np.random.randn(4,1)  
#print np.dot(feature1.T,feature2)/(np.linalg.norm(feature1)*np.linalg.norm(feature2))
#filedir=r'E:\validation\code\lab\\copyimg\130904198402170636.jpg'
##filedir=r'E:\validation\test.jpg'
#im=cv2.imread(filedir)
##im=rotate_about_center(im,90)
#print im.dtype.name
##gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
##cv2.imshow('1',im)
##cv2.waitKey(0)
##   
#points=[[139.857162476,182.602310181],[228.466384888,184.390106201],[ 182.154281616,231.61819458],[149.090164185, 284.293945312],[214.544631958,284.636962891]]
##points=[[91.6902,115.7979],[142.0693,106.4791],[103.9992,141.9467],[99.9796,171.2519],[146.5974,164.3181]]       
#crop_size=128
#ec_mc_y=48
#ec_y=40
#for i in range(1):        
#        for j in range(5):        
#            cv2.circle(im,(int(points[j][0]),int(points[j][1])),2,(0,0,255),2)
#res, eyec2, cropped, resize_scale= align(im, points, crop_size, ec_mc_y, ec_y)
##cropped=np.float32(cropped)
##print cropped.dtype.name
##gray = np.zeros(cropped.shape, np.float32)
##gray=cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
##gray=color.rgb2gray(cropped)
#cv2.imshow('a',cropped)
##cropped=cropped[:,:,[2,1,0]]
#cv2.imwrite(r'E:\validation\test1.JPG',cropped)
#cv2.waitKey(0)