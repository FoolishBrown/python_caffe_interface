# -*- coding: utf-8 -*-
"""
Created on Fri Apr 07 14:05:43 2017

@author: Saber
"""

import os

class CreateFileList:
    def __init__(self,savepath,impath):
        self.savepath=savepath
        self.impath=impath
        
    def mkimglist(self,listname):
        '''
        @brief 获得目录下文件列表
        '''    
        txtfile=None
        if os.path.isfile(listname):
            txtfile=open(listname,'w')
        else:
            txtfile=open(self.savepath+'\\'+listname,'w')
        def GetFileList(imagepath, fileList):
            newDir = imagepath 
            if os.path.isfile(imagepath):
                fileList.append(imagepath)
            elif os.path.isdir(imagepath):
                for s in os.listdir(imagepath):
                    newDir = os.path.join(imagepath,s)
                    GetFileList(newDir,fileList)
            return fileList
        list = GetFileList(self.impath,[])
        for i in list:
            txtfile.write(i)
            txtfile.write('\n')
        txtfile.close()
    def GetFileList_relative(self,imagepath,fileList,i=0,savepath='\\'):
            newDir = imagepath 
            if os.path.isfile(imagepath):
                fileList.append(savepath+' '+str(i))
            elif os.path.isdir(imagepath):
                i=i+1
                for s in os.listdir(imagepath):
                    savepath=os.path.join(savepath,s)
                    newDir = os.path.join(imagepath,s)
                    GetFileList(newDir,fileList,i,savepath)
            return fileList
    def maketrainlist(self,listname):
        '''
        @brief将数据:在指定目录中获取训练数据列表
        '''
        txtfile=None
        if os.path.isfile(listname):
            txtfile=open(listname,'w')
        else:
            txtfile=open(self.savepath+'\\'+listname,'w')
        filelist=self.GetFileList_relative(self.impath,[],i=0,savepath='\\')
        for i in filelist:
            txtfile.write(i+'\n')
        txtfile.close()
    def div_data(ratio,filedir,savedir=''):
        '''
        @brief: 将数据集分为训练集和测试集,保存在prjpath下
        '''
        trainfile=None
        if os.path.isfile(listname):
            trainfile=open(listname,'r')
        else:
            trainfile=open(self.savepath+'\\'+listname,'r')
        data_train=os.path.dirname(savedir)+'\\'+'train.txt'
        data_test=os.path.dirname(savedir)+'\\'+'test.txt'
        fid_train=open(data_train,'w')
        fid_test=open(data_test,'w')
        
        num=0
        data_list=trainfile.readlines()
        for  sublist in data_list:
            if num%ratio!=0:
                fid_train.write(sublist)
            else:
                fid_test.write(sublist)
            num=num+1
        fid_train.close()
        fid_test.close()  
        trainfile.close()
        
    #按照目录下图片的数目来来筛选图片列表
    def selectimglistLimitByCount(self,img_path,context,img_listpath,limitNum,flag='w',iftrain=True):
        fl=None
        if os.path.isfile(listname):
            fl=open(listname,flag)
        else:
            fl=open(self.savepath+'\\'+listname,flag)
        self.impath=self.impath+context
        filenamelist=[]
        classcount=0
        index=-1
        if iftrain:
            for a,b,c in os.walk(self.impath):
                if(len(filenamelist)==0):
                    filenamelist=b
                else:
                    index=index+1
                if(len(c)>=limitNum and limitNum!=0 and index>=0):
                    for iterm in range(limitNum):
                        fl.write(context+'\\'+filenamelist[index]+'\\'+c[iterm]+' '+str(classcount))
                        fl.write('\n')
                    classcount=classcount+1
                elif(limitNum==0 and index>=0):
                    for iterm in range(len(c)):
                        fl.write(context+'\\'+filenamelist[index]+'\\'+c[iterm]+' '+str(classcount))
                        fl.write('\n')
                    classcount=classcount+1
        else:
            for a,b,c in os.walk(self.impath):
                if(len(filenamelist)==0):
                    filenamelist=b
                else:
                    index=index+1
                if(len(c)>=limitNum and limitNum!=0 and index>=0):
                    for iterm in range(limitNum):
                        fl.write(context+'\\'+filenamelist[index]+'\\'+c[iterm])
                        fl.write('\n')
                    classcount=classcount+1
                elif(limitNum==0 and index>=0):
                    for iterm in range(len(c)):
                        fl.write(context+'\\'+filenamelist[index]+'\\'+c[iterm])
                        fl.write('\n')
                    classcount=classcount+1
        fl.close()
    #获取YTB下部分人脸
    def selectimglistLimitByCount_forYTB(self,img_listpath,limitNum,flag='w'):
        fl=None
        if os.path.isfile(img_listpath):
            fl=open(img_listpath,flag)
        else:
            fl=open(self.savepath+'\\'+img_listpath,flag)
        import random
        for a,b,c in os.walk(self.impath):
            if(len(b)==0):
                random.shuffle(c)
                for iterm in range(limitNum):
                    fl.write(a+'\\'+c[iterm])
                    fl.write('\n')
        fl.close()
    def fetchsubdir_relative(self,sourcetxt,targettxt):
        '''
        @brief：将全路径变为相对路径
        '''    
    
        source=open(sourcetxt,'r')
        target=open(targettxt,'w')
        sourcelines=source.readlines()
        i=0
        for sourcesubline in sourcelines:
            word=sourcesubline.split('\\')
            target.write(word[-3]+'\\'+word[-2]+'\\'+word[-1])
            i=i+1
        source.close()
        target.close()
    def generatetestfileForLFWtest(self,num):
        '''
        @brief:按照指定的数量制造测试集
        通常来说是正例反例各一半
        '''
        import random
        lfw_left=open(self.savepath+'\\'+'leftlist.txt','w')
        lfw_right=open(self.savepath+'\\'+'rightlist.txt','w')
        assert(num%2==0)
        label=open(self.savepath+'\\'+'labels.txt','w')
        for i in range(num):
            if i>(num/2.0):
                l=0
            else:
                l=1
            label.write(str(l)+'\n') 
        label.close()
        leftlist=[]
        rightlist=[]
        #先生成同类
        temppair=[]
        diffpair=[]
        selectnum=0
        print self.impath
        for a,b,c in os.walk(self.impath):
            bnum=len(b)
            cnum=len(c)
            if cnum==0 and bnum<10 :
                if selectnum==num/2.0:
                    break
                if len(temppair)>=2:
                    selectnum+=1
                    print selectnum
                    random.shuffle(temppair)
                    leftlist.append(temppair[0])
                    rightlist.append(temppair[1])
                    diffpair.append(temppair[-1])
                    
                temppair=[]
            if cnum>=2 and bnum==0:
                for f in c:
                    temppair.append(os.path.join(a,f))   
        if len(temppair)>=2:
            random.shuffle(temppair)
            leftlist.append(temppair[0])
            rightlist.append(temppair[1])
            diffpair.append(temppair[-1])
        print len(temppair),len(diffpair)
        ld=len(diffpair)
        print ld
        for i in range((num/2)-1):
            left_index=random.randint(i+1,ld-1)
            leftlist.append(diffpair[left_index])
            rightlist.append(diffpair[i])    
        assert(len(leftlist)==len(rightlist))
        for i in range((num/2)-1):
            lfw_left.write(leftlist[i]+'\n')
            lfw_right.write(rightlist[i]+'\n')
        lfw_left.close()
        lfw_right.close()
if __name__=='__main__':
    a=CreateFileList(r'E:\database\Lab144_lightcnn\3-23\128\YTF_3',r'E:\database\frame_images_DB')
#    a.mkimglist('ytblist.txt')
    a.selectimglistLimitByCount_forYTB('testYTB10.txt',20)
#    a.generatetestfileForLFWtest(3000)