# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 14:16:36 2017

@author: Saber
"""

import readConf
import caffe
import numpy as np
import os
import scipy.io as scio
import sklearn.metrics as skm
from collections import OrderedDict
caffe.set_mode_gpu()
#自己的算法
import Calculate_acc
import ReadimgList
import CompareFeature

class LFWtest:
    def __init__(self,section):
        #模型信息读取
        self.dataroot=readConf.getConfig(section,"imgroot")
        self.modelroot=readConf.getConfig(section,"root")
        self.caffemodel=readConf.getConfig(section,"model")
        self.deploy=readConf.getConfig(section,"deploy")
        self.meanfile=readConf.getConfig(section,"meanproto")
        self.inputlayer=readConf.getConfig(section,"datalayer")
        self.outlayer=readConf.getConfig(section,"outputlayer")
        self.scale=readConf.getConfig(section,"inputscale")
        if os.path.isfile(self.meanfile):
            blob = caffe.proto.caffe_pb2.BlobProto()
            data = open(self.meanfile,'rb').read()
            blob.ParseFromString(data)
            array=np.array((caffe.io.blobproto_to_array(blob)))
            mean=array[0]
            print mean
            self.mean=mean.mean(1).mean(1)
            print self.mean
        #LFW读取
        self.left=readConf.getConfig('LFW_test',"left")
        self.right=readConf.getConfig('LFW_test','right')
        self.label=readConf.getConfig('LFW_test',"label")
        labelfile=open(self.label,'r')
        label_rows=labelfile.readlines()
        self.labels=[]    
        for label_sub in label_rows:
            self.labels.append(int(label_sub.strip('\n')))
    def setdata2model_Manual(self,func,save=False):
        '''
        @brief:func:可选read_imagelist_3channel或者read_imagelist_1channel
        对应RGB和Grey图像
        '''

        net=caffe.Classifier(self.modelroot+'\\'+self.deploy,self.modelroot+'\\'+self.caffemodel,input_scale=1)
        net.transformer.set_input_scale(self.inputlayer,int( self.scale))
        batch_size = net.blobs[self.inputlayer].data.shape[0]
        
        size=net.blobs[self.inputlayer].data.shape[2]
        fea_len=net.blobs[self.outlayer].data.shape[1]
        X,X_num=func(self.left,size=[size,size],root=self.dataroot)
        feature_left=np.zeros((X_num,fea_len),dtype='float32')
        img_batch = []
        for i in range(X_num):
            img_batch.append(X[i])
            if len(img_batch) == batch_size or i==X_num-1:
                scores = net.predict(img_batch, oversample=False)
                blobs = OrderedDict( [(k, v.data) for k, v in net.blobs.items()])
                print '%d images processed' % (i+1,)
                feature_left[i-len(img_batch)+1:i+1, :] = blobs[self.outlayer][0:len(img_batch),:].copy()
                img_batch = []
#            net.blobs[self.inputlayer].data[...]= X[i]
#            out=net.forward()
#            feature_left[i,:]=out[self.outlayer]
        feature_left=np.asarray(feature_left,dtype='float32')
        
        X2,X_num=func(self.right,size=[size,size],root=self.dataroot)
        feature_right=np.zeros((X_num,fea_len),dtype='float32')
        for i in range(X_num):
            img_batch.append(X2[i])
            if len(img_batch) == batch_size or i==X_num-1:
                scores = net.predict(img_batch, oversample=False)
                blobs = OrderedDict( [(k, v.data) for k, v in net.blobs.items()])
                print '%d images processed' % (i+1,)
                feature_right[i-len(img_batch)+1:i+1, :] = blobs[self.outlayer][0:len(img_batch),:].copy()
                img_batch = []
        feature_right=np.asarray(feature_right,dtype='float32')
        assert(feature_left.shape==feature_right.shape)
        if save:
            self.savemat(feature_left,feature_right)
        return feature_left,feature_right,X_num
    def savemat(self,feature_left,feature_right):
        scio.savemat(self.modelroot+'\\'+os.path.splitext(self.caffemodel)[0]+'_lfw.mat', 
                     {'features_left':feature_left,'features_right':feature_right,'labels':self.labels})
       
    def readmat(self):
#        LFWdata=scio.loadmat(r'E:\validation\face_verification_experiment-master\results\LFW_l.mat')
#        
                
#        LFWdata1=scio.loadmat(r'E:\validation\face_verification_experiment-master\results\LFW_r.mat')
#        return LFWdata['features'],LFWdata1['features'],len(LFWdata['features'])

        LFWdata=scio.loadmat(self.modelroot+'\\'+os.path.splitext(self.caffemodel)[0]+'_lfw.mat')
        return LFWdata['features_left'],LFWdata['features_right'],len(LFWdata['labels'][0])
    def comAccuracy(self,Grey=False,save=False,notexist=1):
        '''
        @brief：
        flag=0特征已经存在，可以直接读取
        save 是否保存
        grey 是不是灰度图像
        '''
        feature1=[]
        feature2=[]
        num=0
        if notexist:
            if Grey:
                feature1,feature2,num=self.setdata2model_Manual(ReadimgList.read_imagelist_1channel,save=save)
            else:
                feature1,feature2,num=self.setdata2model_Manual(ReadimgList.read_imagelist_3channel,save=save)
        else:
            feature1,feature2,num=self.readmat()
        predicts=np.zeros((num,1))
#        for i in range(num):
#            predicts[i]=np.dot(feature1[i],feature2[i].T)/(np.linalg.norm(feature1[i])*np.linalg.norm(feature2[i]))
        predict=CompareFeature.compare_pair_features(feature1,feature2,flag=0,metric='cosine')
        for i in range(len(predict)):
            predicts[i]=predict[i]
        print predict
        print feature1[0],feature2[0]
        pre=[]
        acc=Calculate_acc.calculate_accuracy_pairs(predicts,self.labels,num,threshold=[0,0.89],step=0.01)
        pre.append(np.max(acc))
        return pre,predicts
    def draw(self,predicts,title):
        fpr, tpr, thresholds=skm.roc_curve(self.labels,predicts)
        '''
        画ROC曲线图
        '''
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic using: '+title)
        plt.legend(loc="lower right")
    #    plt.show()
        plt.savefig(self.modelroot+'\\'+os.path.splitext(self.caffemodel)[0]+title+'.png')        
        
if __name__=='__main__':
    section="deepid64_lfw"
    lfwtestdemo=LFWtest(section)
    pre,_=lfwtestdemo.comAccuracy(save=True,notexist=0,Grey=True)
    print pre
        
        
        
        
        
        
        
        
        
        
        
        
        
        