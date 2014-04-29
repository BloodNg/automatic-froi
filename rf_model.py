#!/usr/bin/env python
"""
=======================================================================
Random forest trainning and cv for paramter selection.

Input  : a sess list with train and cv.
Output : a pair of best parameter forest classifier.

TODO:
    1. Cross-splite the sess list for train and cv by combining the sample parts
       trainning part and cv part.
    2. feature scaling and setting.
    3. fit the training dataset to get clf, using DICE score for evaluation.
    4. average the scores with fold and different classes.
    5. save the best parameter of clf and write scores of different fold and classes.
    
    Note : manual change the parameter for grid search.
Paramter: the number of the forest N and the maximum depth of the tree.
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
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier)
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import cross_validation

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
        dict[sid]=np.loadtxt('sample_repo/sample_%s'%sid)
    return dict

def load_test(sess):
    dict = {}
    for sid in sess:
        dict[sid]=np.loadtxt('sample_repo/sample_%s'%sid)
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
    print sam.shape
    return sam

def read_test_sample(id):
    se = np.loadtxt('repo/test/sample_%s'%id)
    return se

def dice_per_sub(sess,clf,mean,std):
    print "dice session:",sess
    dice_all = np.array([])
    flag = 0
    for sub in sess:
        dice_one = []
        test_set = read_test_sample(sub)
        #1. mask z>2.3
        mask_c = test_set[:,3]>=2.3
        TE = test_set[mask_c]
        #2. split x and y
        X_c = TE[:,:50]
        y_c = TE[:,50]
        #3. Standardize
        X_t = (X_c - mean)/std
        y_t = y_c
        y_p = clf.predict(X_t)
        #4. Compute 4 dice
        list = [0,1,3,5]
        for i in list:
            print y_p
            print y_t
            A= y_p==i
            B= y_t==i
            print "A:",A
            print "B:",B
            di=dice(A,B)
            print di
            dice_one.append(di)
        dice_one = np.array(dice_one)
        if flag==0:
            dice_all = dice_one
            flag = 1
        else:
            dice_all = np.vstack((dice_all,dice_one))
    
    print dice_all.shape
    print "dice_all:",dice_all
    print "mean dice:",np.mean(dice_all,axis=0)
    print "dice std:",np.std(dice_all,axis=0)
    filename = "t_dice_score_%d.txt"%len(sess)
    np.savetxt(filename,dice_all,fmt='%1.4e',delimiter=',',newline='\n')
    print "dice have been saved in the %s"%filename

def dice(r,p):
    return 2.0*np.sum(r*p)/(np.sum(r)+np.sum(p))

def main():
    #read train and cv part sess list
    sess = read_sess_list('./sess')
    sample = load_train(sess)

    coor = np.loadtxt('./coordinates')
    print coor,coor.shape
    #rean img template
    img = nib.load('MNI_brain.nii.gz')
    data = img.get_data()

    print "train+cv number:",len(sess)
    sa = time.time()
    feat_mask = [[0,1,2,3],
              [0,1,2,3,6,7,8,12,13,14,15,16,17,18,19,20,21,22,
               23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,
               72,73,74,75,76,77,78,79,80,81],
              [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
               23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,
               43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,
               63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,
               83,84,85,86,87,88,89,90,91]]
    masks = feat_mask[2]
    #initial the parameter
    param = []
    T = [5,10,50,100,150,200,250,300]
    D = [5,10,15,20,25,30,35,40]
    param = [[t,d] for t in T for d in D]
    #print param
    #for a pair of parameter,do kFold cross validation and compute the DICE
    Dices = np.zeros((64,3)) # average all fold and classes.
    for i,p in enumerate(param):
        print 'parameter:',p
        Dices[i][0] = p[0]
        Dices[i][1] = p[1]
        dice_all = np.array([]) # record the Dice detail of specific parameter.
        flag = 0
        st = time.time()
        kf = KFold(len(sess), n_folds=3, indices=False)
        for train_m, cv_m in kf:
            sp = time.time()
            #print("%s %s" % (train_m, cv_m))
            TR_sample = combin_sample(sess[train_m],sample)
            CV_sample = combin_sample(sess[cv_m],sample)
            #intensity
            mask_t = TR_sample[:,3]>=2.3
            mask_c = CV_sample[:,3]>=2.3
            TR = TR_sample[mask_t]
            CV = CV_sample[mask_c]
            #feature select
            X = TR[:,masks]
            y = TR[:,-1]
            X_c = CV[:,masks]
            y_c = CV[:,-1]
            #print X.shape,y.shape,X_c.shape,y_c.shape
            
            # Shuffle
            idx = np.arange(X.shape[0])
            np.random.seed(10)
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]

            # Standardize
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            X_train = (X - mean)/std
            X_cv = (X_c - mean)/std
            y_train = y
            y_cv = y_c
         
            print "time prepare sample once used:%s"%(time.time()-sp)
            # training model and save model.
            clf = RandomForestClassifier(bootstrap=True,criterion='entropy',max_depth=p[1], 
                max_features='sqrt',n_estimators=p[0],min_samples_split=8, n_jobs=100)
            
            tt = time.time()
            clf.fit(X_train,y_train)
            print "feature importances:",clf.feature_importances_
            print "fit once used:%s"%(time.time()-tt)
            
            tt = time.time()
            y_pred = clf.predict(X_cv)
            print "predict once used:%s"%(time.time()-tt)
            #y_pp = clf.predict_proba(X_test)
            
            y_true = y_cv
            print(classification_report(y_true, y_pred))
            
            #compute dice
            tt = time.time()
            dice_one = []
            labels = [0,1,3,5]
            for j in labels:
                #print y_p.shape
                #print y_t.shape
                A= y_pred==j
                B= y_true==j
                #print "A:",A
                #print "B:",B
                di = dice(A,B)
                #print di
                dice_one.append(di)
            dice_one = np.array(dice_one)
            if flag==0:
                dice_all = dice_one
                flag = 1
            else:
                dice_all = np.vstack((dice_all,dice_one))
            print "time compute DICE once used:%s"%(time.time()-tt)

        print 'Dice score for parameter:',p,dice_all
        class_mean = np.mean(dice_all,axis=0)
        fold_mean = np.mean(dice_all,axis=1)
        para_mean = np.mean(class_mean)
        para_std = np.std(fold_mean)
        print 'Class Dice mean:', class_mean
        print 'Fold Dice mean:', fold_mean
        print 'Parameter Dice mean:', para_mean
        print 'Parameter Dice std:', para_std
        
        Dices[i][2] = para_mean
        #print Dices
        filename = "Dice_of_%d_%d.txt"%(p[0],p[1])
        np.savetxt(filename,dice_all,fmt='%1.4e',delimiter=',',newline='\n')
        print "dice have been saved in the %s"%filename
    #compute dice for every single subject. save the result.
    #dice_per_sub(test,clf,mean,std)
        print "time used:%s"%(time.time()-st)
    filename = "Mean_Dice.txt"
    np.savetxt(filename,Dices,fmt='%1.4e',delimiter=',',newline='\n')
    print "all time used:%s"%(time.time()-sa)
    return

if __name__=="__main__":
    main()
