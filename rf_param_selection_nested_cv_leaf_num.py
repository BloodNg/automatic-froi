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
        dict[sid]=np.loadtxt('dataset/feat_%s'%sid)
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
    se = np.loadtxt('dataset/feat_%s'%id)
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
    sess = read_sess_list('./sess_norm')
    #sess = read_sess_list('./sess')
    sample = load_train(sess)
    print 'load data use:',time.time()-sa

    coor = np.loadtxt('./coordinates')
    print "space coordinate:",coor.shape
    
    #read img template
    img = nib.load('MNI_brain.nii.gz')
    data = img.get_data()

    print "total subject number:",len(sess)
    
    sa = time.time()

    #initial the parameter Depth and Tree num
    TreeN = 50
    DepthM = 15
    LeafM = [1,2,3,4,5,6,7,8,9,10]
    
    print "Parameter set: ",TreeN,DepthM,LeafM
    #total 423 feats
    feat_mask = np.array(range(0,423))
    print "feat_mask : ",feat_mask,feat_mask.shape

    #outer layer cv for test
    kf = KFold(len(sess), n_folds=3)
    k = 0

    test_metrics = []

    fflag = 0
    for train, test in kf:
        print 'outer layer %s-fold'%k
        #for a pair of parameter,do 3-Fold cross validation and compute the DICE
        metric_set = np.zeros((10,9)) #average all fold and classes. mean and std.
        sess_train = sess[train]
        sess_test  = sess[test]
        sess_train_f = "sess_train_fold_%d"%k
        sess_test_f = "sess_test_fold_%d"%k
        np.savetxt(sess_train_f,sess_train,fmt='%s',delimiter=',',newline='\n')
        np.savetxt(sess_test_f,sess_test,fmt='%s',delimiter=',',newline='\n')
        print "Subjects for outer training : ",sess_train
        print "Subjects for outer testing : ",sess_test
        #-------------------------------parameter selection part-----------------------------
        sk = time.time()
        for i,leaf_min in enumerate(LeafM):
            print 'Parameter: Tree_Num : 50 ,Depth_Max : 15 ,Leaf Sample Number Min:%s'%leaf_min
            metric_set[i][0] = leaf_min
            metric_param = np.array([]) #record the classes Dice detail of specific parameter.
            flag = 0
            #inner layer 3-fold cv for parameter selection. 

            sf = time.time()
            kfn = KFold(len(sess_train), n_folds=3)
            for train_m, cv_m in kfn:
                sp = time.time()
                #print("%s %s" % (train_m, cv_m))
                TR = combin_sample(sess_train[train_m],sample)
                CV = combin_sample(sess_train[cv_m],sample)
                print TR.shape,CV.shape
                #feature select
                X = TR[:,feat_mask]
                y = TR[:,-4]
                X_c = CV[:,feat_mask]
                y_c = CV[:,-4]

                print X.shape,y.shape,X_c.shape,y_c.shape
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
                X_cv = (X_c - mean)/std
                y_train = y
                y_cv = y_c
                 
                print "time prepare sample once used:%s"%(time.time()-sp)
                    
                #training model and save model.
                clf = RandomForestClassifier(bootstrap=True,criterion='entropy',max_depth=DepthM,
                        max_features='sqrt',n_estimators=TreeN,min_samples_split=leaf_min, n_jobs=100)
                
                tr = time.time()
                clf.fit(X_train,y_train)
                #print "feature importances:",clf.feature_importances_
                print "fit model once used:%s"%(time.time()-tr)
                
                tp = time.time()
                y_pred = clf.predict(X_cv)
                print "validate predict once used:%s"%(time.time()-tp)
                
                y_true = y_cv
                #print(classification_report(y_true, y_pred))
                
                #compute dice,precision,recall,f1,score
                tm = time.time()
                metric = []
                labels = [0,1,2,3,4]
                
                precision = precision_score(y_true, y_pred,average=None)
                recall = recall_score(y_true, y_pred,average=None)
                f1 = f1_score(y_true,y_pred,average=None)
                
                print 'precision,recall,f1',precision,recall,f1

                for j in labels:
                    P = y_pred==j
                    T = y_true==j
                    di = dice(P,T)
                    #print 'Dice for label:',j,di
                    metric.append(di)

                metric = np.array(metric)
                metric = np.hstack((metric,precision))
                metric = np.hstack((metric,recall))
                metric = np.hstack((metric,f1))
                #print 'Dice,precision,recall,f1',metric

                if flag==0:
                    metric_param = metric
                    flag = 1
                else:
                    metric_param = np.vstack((metric_param,metric))
                
                print "time compute one parameter metrics once used:%s"%(time.time()-tm)

            #print 'Metrics score for a parameter pair:',p,metric_param
            
            #static the metrics and save the score results.
            
            #average cross the classes
            Dice_mean = np.mean(metric_param[:,0:5],axis=1)
            Precision_mean = np.mean(metric_param[:,5:10],axis=1)
            Recall_mean = np.mean(metric_param[:,10:15],axis=1)
            F1_mean = np.mean(metric_param[:,15:20],axis=1)

            #print Dice_mean
            #print Precision_mean
            #print Recall_mean
            #print F1_mean
                
            #average and std cross the k-fold
            Dice_mean = np.mean(Dice_mean)
            Dice_std = np.std(Dice_mean)
            Precision_mean = np.mean(Precision_mean)
            Precision_std = np.std(Precision_mean)
            Recall_mean = np.mean(Recall_mean)
            Recall_std = np.std(Recall_mean)
            F1_mean = np.mean(F1_mean)
            F1_std = np.std(F1_mean)
                
            #print 'Dice mean:', Dice_mean
            #print 'Dice std:', Dice_std
            #print 'Precision mean:', Precision_mean
            #print 'Precision std:', Precision_std
            #print 'Recall mean:', Recall_mean
            #print 'Recall std:', Recall_std
            #print 'F1 score mean:', F1_mean
            #print 'F1 score std:', F1_std
                
            metric_set[i][1] = Dice_mean
            metric_set[i][2] = Dice_std
            metric_set[i][3] = Precision_mean
            metric_set[i][4] = Precision_std
            metric_set[i][5] = Recall_mean
            metric_set[i][6] = Recall_std
            metric_set[i][7] = F1_mean
            metric_set[i][8] = F1_std

            filename = "param_select_score/outer_fold_%d_TreeN_50_DepthM_15_LeafM_%d.txt"%(k,leaf_min)
            np.savetxt(filename,metric_param,fmt='%1.8e',delimiter=',',newline='\n')
            print "evaluation score under leaf sample min for every class and fold have been saved in the %s"%filename
            
            print "first level cv once, time used:%s"%(time.time()-sf)
            
        print "try all parameter pair for one outer level fold, time used:%s"%(time.time()-sk)
        print "evalutaion statistic score: ", metric_set

        filename = "param_select_score/eval_stat_fold_%s_TreeN_50_DepthM_15_LeafM.txt"%k
        np.savetxt(filename,metric_set,fmt='%1.8e',delimiter=',',newline='\n')
        print "evaluation static score under leaf sample min have been saved in the %s"%filename
        k += 1
    print "all time used:%s"%(time.time()-sa)
    return

if __name__=="__main__":
    main()
