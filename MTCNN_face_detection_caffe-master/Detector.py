
import cv2
import numpy as np
from MtcnnDetector import FaceDetector

def detectface2_5p_relative(filelist,AB2plist,AB5plist,errorlist,size,imgroot=''):
    from MtcnnDetector import FaceDetector
    fp1 = open(AB2plist,'w')
    fp2 = open(AB5plist,'w')
    fp3 = open(errorlist,'w')
    fid = open(filelist)
    lines = fid.readlines()
    for line in lines:
        line = line.strip().split(' ')[0]  
        img = cv2.imread(imgroot+line)
        
#        img=cv2.resize(img,size)
        try:
            detector = FaceDetector(minsize = 20, gpuid = 0, fastresize = False) 
            total_boxes,points,numbox = detector.detectface(img)
            x1 = float(total_boxes[0][0])
            y1 = float(total_boxes[0][1])
            x2 = float(total_boxes[0][2])
            y2 = float(total_boxes[0][3])
            a1 = float(points[0,0])
            b1 = float(points[5,0])
            a2 = float(points[1,0])
            b2 = float(points[6,0])
            a3 = float(points[2,0])
            b3 = float(points[7,0])
            a4 = float(points[3,0])
            b4 = float(points[8,0])
            a5 = float(points[4,0])
            b5 = float(points[9,0])
            fp1.write(line + ' ' + str(x1) + ' ' + str(x2) + ' ' + str(y1)+ ' ' + str(y2) +'\n')
            fp2.write(line + ' ' + str(a1) + ' ' + str(b1) + ' ' + str(a2)+ ' ' + str(b2) + ' ' + str(a3) + ' ' + str(b3) + ' ' + str(a4) + ' ' + str(b4) + ' ' + str(a5) + ' ' + str(b5) +'\n')
        except:
            print 'error!'
            fp3.write(line + '\n')
            continue
    fp1.close()
    fp2.close()
    fp3.close()
def detectface2_5p_returnpoints(imagepath):
    from MtcnnDetector import FaceDetector
    flag=True
    try:
        img = cv2.imread(imagepath)
        detector = FaceDetector(minsize = 20, gpuid = 0, fastresize = False) 
        total_boxes,points,numbox = detector.detectface(img)
#        x1 = float(total_boxes[0][0])
#        y1 = float(total_boxes[0][1])
#        x2 = float(total_boxes[0][2])
#        y2 = float(total_boxes[0][3])
#        a1 = float(points[0,0])
#        b1 = float(points[5,0])
#        a2 = float(points[1,0])
#        b2 = float(points[6,0])
#        a3 = float(points[2,0])
#        b3 = float(points[7,0])
#        a4 = float(points[3,0])
#        b4 = float(points[8,0])
#        a5 = float(points[4,0])
#        b5 = float(points[9,0])
    except:
        flag=False
#        print 'error!'
    return points,flag
def detectface2_5p(filelist,AB2plist,AB5plist,errorlist,size=(224,224)):
    from MtcnnDetector import FaceDetector
    fp1 = open(AB2plist,'w')
    fp2 = open(AB5plist,'w')
    fp3 = open(errorlist,'w')
    fid = open(filelist)
    lines = fid.readlines()
    for line in lines:
        line = line.strip()  
        img = cv2.imread(line)
#        img=cv2.resize(img,size)
        try:
            detector = FaceDetector(minsize = 20, gpuid = 0, fastresize = False) 
            total_boxes,points,numbox = detector.detectface(img)
            x1 = float(total_boxes[0][0])
            y1 = float(total_boxes[0][1])
            x2 = float(total_boxes[0][2])
            y2 = float(total_boxes[0][3])
            a1 = float(points[0,0])
            b1 = float(points[5,0])
            a2 = float(points[1,0])
            b2 = float(points[6,0])
            a3 = float(points[2,0])
            b3 = float(points[7,0])
            a4 = float(points[3,0])
            b4 = float(points[8,0])
            a5 = float(points[4,0])
            b5 = float(points[9,0])
            fp1.write(line + ' ' + str(x1) + ' ' + str(x2) + ' ' + str(y1)+ ' ' + str(y2) +'\n')
            fp2.write(line + ' ' + str(a1) + ' ' + str(b1) + ' ' + str(a2)+ ' ' + str(b2) + ' ' + str(a3) + ' ' + str(b3) + ' ' + str(a4) + ' ' + str(b4) + ' ' + str(a5) + ' ' + str(b5) +'\n')
        except:
            fp3.write(line + '\n')
            continue
    fp1.close()
    fp2.close()
    fp3.close()
if __name__ == '__main__':   
    
    filelist=r'G:\file25\list.txt'
    fp1 = open('G:/FaceTools-release/testlist_2p.txt','w')
    fp2 = open('G:/FaceTools-release/testlist_5p.txt','w')
    fp3 = open('G:/FaceTools-release/testlist_error.txt','w')
    fid = open(filelist)
    lines = fid.readlines()
    for line in lines:
        line = line.strip()  
        img = cv2.imread(line)
        try:
            detector = FaceDetector(minsize = 20, gpuid = 0, fastresize = False) 
            total_boxes,points,numbox = detector.detectface(img)
            x1 = float(total_boxes[0][0])
            y1 = float(total_boxes[0][1])
            x2 = float(total_boxes[0][2])
            y2 = float(total_boxes[0][3])
            a1 = float(points[0,0])
            b1 = float(points[5,0])
            a2 = float(points[1,0])
            b2 = float(points[6,0])
            a3 = float(points[2,0])
            b3 = float(points[7,0])
            a4 = float(points[3,0])
            b4 = float(points[8,0])
            a5 = float(points[4,0])
            b5 = float(points[9,0])
            fp1.write(line + ' ' + str(x1) + ' ' + str(x2) + ' ' + str(y1)+ ' ' + str(y2) +'\n')
            fp2.write(line + ' ' + str(a1) + ' ' + str(b1) + ' ' + str(a2)+ ' ' + str(b2) + ' ' + str(a3) + ' ' + str(b3) + ' ' + str(a4) + ' ' + str(b4) + ' ' + str(a5) + ' ' + str(b5) +'\n')
        except:
            fp3.write(line + '\n')
            continue
    fp1.close()
    fp2.close()
    fp3.close()
        
#        for i in range(numbox):
#            cv2.rectangle(img,(int(total_boxes[i][0]),int(total_boxes[i][1])),(int(total_boxes[i][2]),int(total_boxes[i][3])),(0,255,0),2)        
#            for j in range(5):        
#                cv2.circle(img,(int(points[j,i]),int(points[j+5,i])),2,(0,0,255),2)
#    
#        cv2.imwrite( '1_out.jpg',img )
#        print 'Done.'