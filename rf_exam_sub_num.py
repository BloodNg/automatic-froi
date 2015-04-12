#!/usr/bin/env python
"""
=======================================================================
Random forest subject number test.
=======================================================================
"""

import numpy as np
import nibabel as nib
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split,KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import cross_validation
from sklearn.metrics import average_precision_score,roc_auc_score,\
                            precision_score,recall_score,f1_score

def read_sess_list(sess):
    """
    Load subject Id list.
    Tested
    """
    sf = open(sess,'r')
    sess = sf.readlines()
    sess = [line.rstrip('\r\n') for line in sess]
    sess = np.array(sess)
    print sess
    return sess

def load_train(sess):
    dict = {}
    for sid in sess:
        dict[sid]=np.loadtxt('dataset/feat_%s'%sid)
    return dict

def load_test(sess):
    dict = {}
    for sid in sess:
        dict[sid]=np.loadtxt('dataset/ab_feat_%s'%sid)
    return dict


def combin_sample(sess,dict):
    sam = np.array([])
    flag = 0
    for i in sess:
        se = dict[i]
        if flag==0:
            sam = se
            flag = 1
        else:
            sam= np.vstack((sam,se))
 #   print sam.shape
    return sam

def read_test_sample(id):
    se = np.loadtxt('dataset/ab_feat_%s'%id)
    return se

def testset_predict(sess,clf,mean,std,masks,fold,img):
    print "Predicting subject :",sess
    flag = 0
    metric_all = np.array([])
    for sub in sess:
        metric = np.ones(20)
        test_set = read_test_sample(sub)
        TE = test_set
        #2. split x and y
        X_cv = TE[:,masks]
        y_cv = TE[:,-4]
        coor = TE[:,-3:]
        #3. Standardize
        X_true = (X_cv - mean)/std
        y_true = y_cv
        
        y_pred = clf.predict(X_true)
        
        list = [0,1,2,3,4]
        for i,l in enumerate(list):
            P = y_pred== l
            T = y_true== l
            di = dice(P,T)
            if np.isnan(di):
                metric[i] = 1
            else:
                metric[i] = di
        
        #4. Compute metrics
        precision = precision_score(y_true, y_pred,average=None)
        recall = recall_score(y_true, y_pred,average=None)
        f1 = f1_score(y_true,y_pred,average=None)
        
        if len(precision)==len(list):
            metric[5:10] = precision
        if len(recall)==len(list):
            metric[10:15] = recall
        if len(f1)==len(list):
            metric[15:20] = f1
        
        #print metric
        if flag==0:
            metric_all = metric
            flag = 1
        else:
            metric_all = np.vstack((metric_all,metric))

        #construct the predicted image.
        tmp = np.zeros_like(img.get_data())

        for j,c in enumerate(coor):
            tmp[tuple(c)] = y_pred[j]
        img._data = tmp
        nib.save(img,"pred_img_flod_%d/pred_img_%s.nii.gz"%(fold,sub))
        print 'Saving pred_img_%s done!'%sub

    print metric_all
    filename = "test_metrics_sub_level_fold_%d.txt"%fold
    np.savetxt(filename,metric_all,fmt='%1.8e',delimiter=',',newline='\n')
    print "metric score for test set sub level have been saved in the %s"%filename

def dice(r,p):
    return 2.0*np.sum(r*p)/(np.sum(r)+np.sum(p))

def main():
    #read train and cv part sess list
    sa = time.time()
    sess_train = read_sess_list('./sess_norm')
    sess_test = read_sess_list('./sess_abnorm')
    sample_train = load_train(sess_train)
    sample_test = load_test(sess_test)

    TE_sample = combin_sample(sess_test, sample_test)
    print 'load data use:',time.time()-sa

    #read img template
    img = nib.load('MNI_brain.nii.gz')
    data = img.get_data()

    print "total subject number:",len(sess_train)+len(sess_test)
    
    sa = time.time()
    #initial the parameter Depth and Tree num
    #total 423 feats
    feat_mask = np.array(range(0,423))
    print "feat_mask : ",feat_mask,feat_mask.shape

    #outer layer cv for test
    
    ratoi = [0.9375,0.875,0.75,0.625,0.5,0.375,0.25,0.125,0]

    for r in ratoi:
        st = time.time()
        ss = cross_validation.ShuffleSplit(len(sess_train), n_iter=3,\
            test_size = r,random_state=13)
        fflag = 0
        k = 0
        test_metrics = []
        for train_m, cv_m in ss:
            print train_m,sess_test,len(train_m),len(sess_test)
            print 'the %s round'%k
            #-----------------------------------------test part------------------------------------------
            #compute dice for every single test subject. compute the score and save the prediction as .nii
            #using the optimized paramter train and predict the testset.
            ts = time.time()
            #print("%s %s" % (train_m, cv_m))
            TR_sample = combin_sample(sess_train[train_m], sample_train)
            TR = TR_sample
            TE = TE_sample

            #feature select
            X = TR[:,feat_mask]
            y = TR[:,-4]
            X_t = TE[:,feat_mask]
            y_t = TE[:,-4]

            print X.shape, y.shape, X_t.shape, y_t.shape
            print y, np.unique(y), np.unique(y_t)

            #shuffle
            idx = np.arange(X.shape[0])
            np.random.seed(13)
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]

            #standardize
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            X_train = (X - mean)/std
            y_train = y
            X_test = (X_t - mean)/std
            y_test = y_t
        
            depth = 15
            ntree = 50

            clf = RandomForestClassifier(bootstrap=True,criterion='entropy',max_depth=depth,
                max_features='sqrt',n_estimators=ntree,min_samples_split=8, n_jobs=100)

            tf = time.time()
            clf.fit(X_train,y_train)
            print "predict once used:%s"%(time.time()-tf)
        
            print "feature importances:", clf.feature_importances_
            filename = "sub_num_exam/feat_importance_round_%s_%d.txt"%(r,k)
            np.savetxt(filename,clf.feature_importances_,fmt='%1.8e',delimiter=',',newline='\n')

            tp = time.time()
            y_pred = clf.predict(X_test)
            print "predict once used:%s"%(time.time()-tp)
        
            y_true = y_test
            print(classification_report(y_true, y_pred))

            #compute dice
            metric = []
            labels = [0,1,2,3,4]

            precision = precision_score(y_true, y_pred,average=None)
            recall = recall_score(y_true, y_pred,average=None)
            f1 = f1_score(y_true,y_pred,average=None)

            for j in labels:
                P = y_pred==j
                T = y_true==j
                di = dice(P,T)
                print 'Dice for label:',j,di
                metric.append(di)

            metric = np.array(metric)
            metric = np.hstack((metric,precision))
            metric = np.hstack((metric,recall))
            metric = np.hstack((metric,f1))
            print 'Dice,precision,recall,f1',metric
            
            if fflag==0:
                test_metrics = metric
                fflag = 1
            else:
                test_metrics = np.vstack((test_metrics,metric))

            #predict single subject.
            #outer level cv
            k+=1
            print "one fold test time used:%s"%(time.time()-ts)
    
        print 'test metrics sample level : ', test_metrics
        filename = "sub_num_exam/test_metric_sample_level_%s.txt"%r
        np.savetxt(filename,test_metrics,fmt='%1.8e',delimiter=',',newline='\n')
        print "metrics score for testset sample level have been saved in the %s"%filename

    print "all time used:%s"%(time.time()-sa)
    return

if __name__=="__main__":
    main()
