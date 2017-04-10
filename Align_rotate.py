# -*- coding: utf-8 -*-
"""
Created on Fri Apr 07 13:21:11 2017

@author: Saber
"""

import math
import cv2
import numpy as np
import os
class Align_rotate:
    def __init__(self,imageSave_path,imageList,size=(0,0),imgroot=r'',imgerrorp=None):
        imlist_file=open(imageList,'r')
        imlistfile_rows=imlist_file.readlines()
        self.imnum=len(imlistfile_rows)
        self.imsavePath=[]
        self.impath=[]
        self.impoints=np.zeros((self.imnum,5,2))
        for i in range(self.imnum):
                imlistfile_subrow=imlistfile_rows[i].strip('\n')
                imlistfile_subrow=imlistfile_subrow.split(' ')
                filerootdir=imlistfile_subrow[0]
                filerootdir=filerootdir.split('\\')
                self.impath.append(imgroot+'\\'+filerootdir[-2]+'\\'+filerootdir[-1])
                self.imsavePath.append(imageSave_path+'\\'+filerootdir[-2]+'\\'+filerootdir[-1])
                save_dir=imageSave_path+'\\'+filerootdir[-2]
                if os.path.exists(save_dir)==False:
                    os.mkdir(save_dir)
                imlistfile_subrow_points=imlistfile_subrow[1:]
                p_pose=0
                for index in range(5):
                    p_pose=index*2
                    self.impoints[i][index][0]=imlistfile_subrow_points[p_pose]
                    self.impoints[i][index][1]=imlistfile_subrow_points[p_pose+1]
        if imgerrorp is not None:
            self.errorfilepath=imgerrorp
        else:
            self.errorfilepath=imgroot+'\\errorcroplist.txt'
    def cropimgfor_lightcnn(self,train=True,Grey=1):
        for i in range(self.imnum):
            try:
                print self.impath[i]
                im=cv2.imread(self.impath[i],not Grey)
                f5pt=self.impoints[i]
                if train:
                    img_resize, eyec2, cropped, resize_scale=self.alignfor_lightcnn(im, f5pt, 144, 48, 48)
                else:
                    img_resize, eyec2, cropped, resize_scale=self.alignfor_lightcnn(im, f5pt, 128, 48, 40)
                cv2.imwrite(self.imsavePath[i],cropped)
            except Exception:
                f=open(self.errorfilepath,'a')
                f.write(self.impath[i]+'\n')
                f.close()
                print 'image num:%d'%i
    def transform(self,x,y,ang,s0,s1):
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
        
    def rotate_about_center(self,src, angle, scale=1.):
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
    def guard(self,x, N):
        b=[]
        for i in x:
            if i<1:i=1
            if i>N:i=N
            b.append(i)
        return b
    '''
    传入参数的时候应该把大小调整过
    '''
    def alignfor_lightcnn(self,img, f5pt, crop_size, ec_mc_y, ec_y):
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
        img_rot=self.rotate_about_center(img,ang)
        imgh = img.shape[0]
        imgw = img.shape[1]
        #取眼睛的中点
        x=(f5pt[0][0]+f5pt[1][0])/2.0
        y=(f5pt[0][1]+f5pt[1][1])/2.0
        
        ang = -ang/180.0*math.pi
        xx, yy= self.transform(x, y, ang, img.shape, img_rot.shape)
        eyec=np.round([xx, yy])
    #    cv2.circle(img_rot,(int(eyec[0]),int(eyec[1])),2,(0,255,255),2)
    #    cv2.imshow('res',img_rot)    
    #    cv2.waitKey(0)
        #嘴巴的中点
        x=(f5pt[3][0]+f5pt[4][0])/2.0
        y=(f5pt[3][1]+f5pt[4][1])/2.0
        
        ang = -ang/180.0*math.pi
        xx, yy= self.transform(x, y, ang, img.shape, img_rot.shape)
        
        mouthc=np.round([xx, yy])
        resize_scale = ec_mc_y/float(mouthc[1]-eyec[1])
        img_resize=cv2.resize(img_rot,(int(img_rot.shape[0]*resize_scale),int(img_rot.shape[1]*resize_scale)))
        eyec2 = (eyec -[img_rot.shape[1]/2.0,img_rot.shape[0]/2.0]) * resize_scale + [img_resize.shape[1]/2.0,img_resize.shape[0]/2.0]
        eyec2 = np.round(eyec2)
        
    #    img_resize=rotate_about_center(img_resize,90)

        img_crop = np.zeros((crop_size, crop_size, img.shape[2]),dtype='uint8')
        crop_y = eyec2[1] - ec_y
        crop_y_end = crop_y + crop_size 
        crop_x = eyec2[0]-np.floor(crop_size/2.0)
        crop_x_end = crop_x + crop_size
        box = self.guard([crop_x,crop_x_end,crop_y,crop_y_end], 500)
        temp_row=img_resize.shape[0]
        temp_col=img_resize.shape[1]
        if img_resize.shape[0]<box[3] :
            temp_row=box[3]
        if img_resize.shape[1]<box[1]:
            temp_col=box[1]
        img_temp = np.zeros((temp_row, temp_col, img.shape[2]),dtype='uint8')
        img_temp[0:img_resize.shape[0],0:img_resize.shape[1],:]=img_resize[:,:,:]
        img_crop[:,:,:] = img_temp[box[2]:box[3],box[0]:box[1],:]
        cropped = img_crop
        cropped=cropped
        return img_resize, eyec2, cropped, resize_scale
        
if __name__=='__main__':
    a=Align_rotate(r'E:\Temp',r'E:\Temp\test.txt',size=(128,128),imgroot=r'E:\database\CASIA-WebFace')
    a.cropimgfor_lightcnn(train=True,Grey=0)