#!/usr/bin/env python
"""
=======================================================================
Random forest trainning and test module.

Input : train raw samples within trainning list.
Output : a forest classifier.

TODO:
    1. Cross-splite the train sess list for combining the sample parts
       trainning part/validation part
    2. feature scaling?
    3. for trainning k clf and scoring
    4. average the scores (Dice)
       clf evaluation. write the parameter log and scores
    5. save the clf.

    manual change the parameter for grid search.
=======================================================================
"""

import numpy as np
import pylab as pl
import nibabel as nib
import time 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split,KFold
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model, datasets
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier)
from sklearn.metrics import roc_curve, auc, roc_auc_score
import  matplotlib.pyplot as plt
import cPickle as pickle

def save_clf(clf):
    '''
    Save the clf
    '''
    with open('rf_act.clf', 'wb') as outfile:
        pickle.dump(clf, outfile, pickle.HIGHEST_PROTOCOL)
    print 'save the model done.'
    return 0

def load_clf():
    '''
    Load the clf
    '''
    with open('rf_act.clf', 'rb') as infile:
        rf_clf = pickle.load(infile)
    return rf_clf

def load_data(filename):
    raw_data = pd.read_csv(filename)
    dataset = raw_data.values
    return dataset

def sigmoid(z):
    theta = 1
    return 1.0/(1+np.exp(-theta*z))


def score_fun(beh):
    '''
    User behavior modeling.
    Scoring range (1,10)
    '''
    beh = np.array(beh)

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

def combin_sample(sess,f):
    sam = np.array([])
    flag = 0
    for i in sess:
        if f ==0:
            se = np.loadtxt('repo/train/sample_%s'%i)
        else:
            se = np.loadtxt('repo/test/sample_%s'%i)
        
        if flag==0:
            sam = se
            flag = 1
        else:
            sam= np.vstack((sam,se))
    return sam

def read_test_sample(id):
    se = np.loadtxt('repo/test/sample_%s'%id)
    return se

def dice_per_sub(sess,clf,mean,std,coor,masks,img):
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
        X_c = TE[:,masks]
        y_c = TE[:,-1]
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
    
        co = coor[mask_c]
        tmp = np.zeros_like(img.get_data())

        for j,c in enumerate(co):
            tmp[tuple(c)] = y_p[j]
        img._data = tmp
        nib.save(img,"./predict_context/predicted_%s.nii.gz"%sub)
        print 'saving predicted_%s done!'%sub
    
    print dice_all.shape
    print "dice_all:",dice_all
    print "mean dice:",np.mean(dice_all,axis=0)
    print "dice std:",np.std(dice_all,axis=0)
    filename = "dice_score_%d:"%len(sess)
    np.savetxt(filename,dice_all,fmt='%1.4e',delimiter=',',newline='\n')
    print "dice have been saved in the %s"%filename

def dice(r,p):
    return 2.0*np.sum(r*p)/(np.sum(r)+np.sum(p)+0.0000001)

def main():
    #read train part sess list
    train = read_sess_list('./train_split.sess')
    #read test part sess list
    test = read_sess_list('./test_split.sess')
    #read mask indexs 
    coor = np.loadtxt('./coordinates')
    print coor,coor.shape
    #rean img template
    img = nib.load('MNI_brain.nii.gz')
    data = img.get_data()
    TR_sample = combin_sample(train,0)
    TE_sample = combin_sample(test,1)
    print TR_sample.shape
    print TE_sample.shape
    
    #print TR_sample[4000]
    mask_t = TR_sample[:,3]>=2.3
    mask_c = TE_sample[:,3]>=2.3
    TR = TR_sample[mask_t]
    TE = TE_sample[mask_c]
    print TR.shape
    print TE.shape
    '''
    masks = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
               23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,
               43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,
               63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,
               83,84,85,86,87,88,89,90,91]
    masks = [0,1,2,3]
    '''
    masks = [0,1,2,3,6,7,8,12,13,14,15,16,17,18,19,20,21,22,
               23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,
               72,73,74,75,76,77,78,79,80,81]

    X = TR[:,masks]
    y = TR[:,-1]
    X_c = TE[:,masks]
    y_c = TE[:,-1]
    print X.shape,y.shape,X_c.shape,y_c.shape
    
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
    X_test = (X_c - mean)/std
    y_train = y
    y_test = y_c
    
    # training model and save model.
    clf = RandomForestClassifier(bootstrap=True,criterion='entropy',max_depth=30, 
                                 max_features='sqrt',min_samples_split=8,
                                 n_estimators=100, n_jobs=100, 
                                 oob_score=True)
    st = time.time()
    clf.fit(X_train,y_train)
    print "time used:%s"%(time.time()-st)
    save_clf(clf)
    print "save the clf done!"
    '''
    print "fi:",clf.feature_importances_
    print "oob:",clf.oob_score_
    print "mean accuracy:",clf.score(X_test, y_test)
    
    st = time.time()
    y_p  = clf.predict(X_test)
    #y_pp = clf.predict_proba(X_test)
    print "time used:%s"%(time.time()-st)
    

    y_t = y_test
    print(classification_report(y_t, y_p))
    print 'prediction:',y_p[300:600]
    print 'true:',y_t[300:600]
    dice_one = []
    list = [0,1,3,5]
    for i in list:
        A = y_p==i
        B = y_t==i
        di = dice(A,B)
        print di
        dice_one.append(di)
    dice_one = np.array(dice_one)
    print 'dice_one',dice_one
    print "dice mean[0,1,3,5]:",np.mean(dice_all,axis=0)
    print "dice std [0,1,3,5]",np.std(dice_all,axis=0)
    '''

    #prediction real 
    rf_clf = load_clf()
    dice_per_sub(test,rf_clf,mean,std,coor,masks,img)
    print "time used:%s"%(time.time()-st)
    '''
    #predict one subject
    for i in range(42):
        print i
        sub = np.loadtxt("repo/test/sample_%s"%test[i])
        print sub[:,3]
        mk = sub[:,3] >= 2.3
        print sub.shape
        sub_md = sub[mk]
        print sub_md[:,3]

        X_t = sub_md[:,masks]
        y_t = sub_md[:,-1]
        co = coor[mk]

        tmp = np.zeros_like(data)
        y_p = clf.predict(X_t)
        
        print 'Prediction:',y_p
        print 'True:',y_t
        for j,c in enumerate(co):
            tmp[tuple(c)] = y_p[j]
        img._data = tmp
        nib.save(img,"./predict/predicted_%s.nii.gz"%test[i])
    
    ra = roc_auc_score(y_test,y_p)
    print "auc score:",ra
    di = dice(y_test,y_p)
    print di
    fpr, tpr, thresholds = roc_curve(y_test, y_pp[:, 1],pos_label=1)
    plt.plot(fpr, tpr, 'b-', label='rf')
    '''
    '''
    #LR
    clf = linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.00001, C=1, fit_intercept=True, intercept_scaling=1, class_weight='auto')
    clf.fit(X_train,y_train)
    #print "fi:",clf.feature_importances_
    #print "oob:",clf.oob_score_
    #print "param:",clf.get_params(True)

    print "time used:%s"%(time.time()-st)
    plt.plot(fpr, tpr, 'y-', label='svc')
    '''
    #plt.legend()
   # plt.show()
    '''
    
    s1 = 'precision' 
    s2 = 'recall'
    s3 = 'f1'

    s1 = 'precision' 
    s2 = 'recall'
    s3 = 'f1'
    s4 = 'accuracy'
    s5 = 'roc_auc'
    
    param = [{'C':[0.001,0.01,0.1,1,20,100]}]
    print("# parameters for %s" %s1)
    model = linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1, class_weight=None)
    clf = GridSearchCV(model, param, cv=3, n_jobs=1, scoring=s1)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")      
    print(clf.best_estimator_)
    print("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"%(mean_score, scores.std() / 2, params))
    print("Detailed classification report:")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    print("# parameters for %s" %s2)
    clf = GridSearchCV(model, param, cv=10,n_jobs=1, scoring=s2)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")      
    print(clf.best_estimator_)
    print("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"%(mean_score, scores.std() / 2, params))
    print("Detailed classification report:")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    print("# parameters for %s" %s3)
    clf = GridSearchCV(model, param, cv=10,n_jobs=1 ,scoring=s3)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")      
    print(clf.best_estimator_)
    print("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"%(mean_score, scores.std() / 2, params))
    print("Detailed classification report:")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    print("# parameters for %s" %s4)
    clf = GridSearchCV(model, param, cv=10,n_jobs=1,scoring=s4)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")      
    print(clf.best_estimator_)
    print("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"%(mean_score, scores.std() / 2, params))
    print("Detailed classification report:")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    print("# parameters for %s" %s5)
    clf = GridSearchCV(model, param, cv=10,n_jobs=1, scoring=s5)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")      
    print(clf.best_estimator_)
    print("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"%(mean_score, scores.std() / 2, params))
    print("Detailed classification report:")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, clf.predict(X_test)

    print "time used:%s"%(time.time()-st)
    '''
    return 

if __name__=="__main__":
    main()
