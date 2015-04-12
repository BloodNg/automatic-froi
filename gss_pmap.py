#!/usr/bin/env python
'''
The Group-Constrained Subject-Specific(GSS) method for defining fROIs
'''

import os
import sys
import re
import time
import nibabel as nib
import numpy as np
from segment import watershed
from sklearn.cross_validation import train_test_split,KFold

def read_sess_list(sess):
    """
    Load subject Id list.
    Tested
    """
    sf = open(sess,'r')
    sess = sf.readlines()
    sess = [line.rstrip('\r\n') for line in sess]
    sess = np.array(sess)
    print sess,len(sess)
    return sess

def gss():
    #1.data reposition.
    database = '../automatic-froi/raw_data'
    contrast = 'face-object'
    roi_mask = nib.load('./brain_mask.nii.gz')
    mask_data = roi_mask.get_data()
    result = roi_mask
    result_img = np.zeros_like(mask_data)
    result._data = result_img

    #2.read  session list, 3-fold.
    print "read session list."
    sess_list = read_sess_list('./sess_norm')
    k=0
    kf = KFold(len(sess_list), n_folds=3, indices=False)
    for train, test in kf:
        p = raw_input('input the paramters.(e.g sigma thr_p thr_w)>>>')
        tmp = p.split()
        sigma = float(tmp[0])
        thr_p = float(tmp[1])
        thr_w = float(tmp[2])
        print sigma,thr_p,thr_w

        train_list = sess_list[train]
        test_list  = sess_list[test]
        print len(train_list)
        print len(test_list)

        train_num = len(train_list)
        print train_num
        #3.load z-map and overlap all.
        print "1.creating prob z-map."
        for i,sub in enumerate(train_list):
            sub_dir =  os.path.join(database,sub)
            sub_ct_dir = os.path.join(sub_dir,contrast)
            f_list = os.listdir(sub_ct_dir)
            for file in f_list:
                if re.search('zstat1.nii.gz',file):
                    x_img =  os.path.join(sub_ct_dir,file)
                    print x_img
            #if re.search('_ff.nii.gz',file): 
            #    y_img =  os.path.join(sub_ct_dir,file)
            #    print y_img
            sub_img = nib.load(x_img)
            sub_data = sub_img.get_data()
            sub_bin = sub_data>=2.3
            sub_img._data = sub_bin
            nib.save(sub_img,'%s_bin.nii.gz'%sub)
            print i,sub,sub_bin.shape

            result_img += sub_bin
    
        #4.save the p overlap map.
        print "2.save prob-map."
        p_map = result_img/(train_num*1.0)
        result._data = p_map
        nib.save(result,'../froi_work/gss/p_map_raw_%d.nii.gz'%k)
        #5.watershed segmentation, smooth,thresh the image
        #p_img = nib.load('p_map_raw.nii.gz')
        #p_map = p_img.get_data()
        regions = watershed(p_map,sigma,thr_p)
        #regions = np.array([[[ int(k) for k in j] for j in i] for i in regions])
        #result._data = regions
        nib.save(result,'../froi_work/gss/waters_%d.nii.gz'%k)
        #6.select the the parches with x% in it.
        labels = np.unique(regions)
        labels = labels[1:]
        labels = [int(i) for i in labels]
        print labels

        for l in labels:
            mask = regions==l
            if np.max(p_map[mask])<thr_w:
                regions[mask]=0

        result._data = regions
        nib.save(result,'../froi_work/gss/p_map_m_%d.nii.gz'%k)

        #7.create a totall mask.
        total_mask = regions > 0
        result._data = total_mask
        nib.save(result,'../froi_work/gss/total_mask_%d.nii.gz'%k)

        #8.intersect to define roi.
        #re_img = nib.load('waters.nii.gz')
        #regions = re_img.get_data()
        
        print "8.read test session list."
        test_sub_num = len(test_list)
        print test_sub_num
        labels = np.unique(regions)
        labels = labels[1:]
        labels = [i for i in labels]
        print labels
        

        line = raw_input('input the roi id for four target.(e.g 18 20, 19, 14 32,)>>>')
        target = line.split(',')
        target_list = [ l.split() for l in target]

        target = [[int(i) for i in l] for l in target_list]
        print  'target:',target
        
        for l in target:
            for i in l:
                mask = regions==i
                regions[mask] = l[0]
        
        result._data = regions
        nib.save(result,'../froi_work/gss/p_map_m_combined_%d.nii.gz'%k)
        
        
        target_new = [l[0] for l in target]
        print target_new
        
        final_t = np.zeros_like(regions)
        for l in target_new:
            mask = regions==l
            final_t[mask] = l

        result._data = final_t
        nib.save(result,'../froi_work/gss/final_template_%d.nii.gz'%k)

        dice_all = []
        flag = 0
        for i,sub in enumerate(test_list):
            sub_dir =  os.path.join(database,sub)
            sub_ct_dir = os.path.join(sub_dir,contrast)
            f_list = os.listdir(sub_ct_dir)
            for file in f_list:
                if re.search('zstat1.nii.gz',file):
                    x_img =  os.path.join(sub_ct_dir,file)
                    print x_img
                if re.search('_ff.nii.gz',file): 
                    y_img =  os.path.join(sub_ct_dir,file)
                    print y_img
            
            sub_img = nib.load(x_img)
            sub_data = sub_img.get_data()
            #ground truth.
            label_img = nib.load(y_img)
            true_data = label_img.get_data()

            pred_data = np.zeros_like(sub_data)
            #target = [18,19,14]
            
            roi = [1,2,3,4]
            
            for l in target_new:
                mask = np.logical_and(regions==l,sub_data>=2.3)
                if l == target_new[0]:
                    print l
                    pred_data[mask] = roi[0]
                elif l == target_new[1] :
                    print l
                    pred_data[mask] = roi[1]
                elif l == target_new[2]:
                    print l
                    pred_data[mask] = roi[2]
                elif l == target_new[3] :
                    print l
                    pred_data[mask] = roi[3]

            #9.evaluation the dice score.
            dice_sub = []
            for r in roi:
                int_roi = np.logical_and(true_data== r,pred_data == r)
                Hit = np.sum(int_roi)
                T = np.sum(true_data == r)
                P = np.sum(pred_data == r)
                dice_r = 2*Hit/((T+P)*1.0) #dice score for label r
                dice_sub.append(dice_r)
            
            dice_sub = np.array(dice_sub)
            print dice_sub
            if flag == 0:
                dice_all = dice_sub
                flag = 1
            else:
                dice_all = np.vstack((dice_all,dice_sub))
            sub_img._data = pred_data
            nib.save(sub_img,'../froi_work/gss/%s_predicted.nii.gz'%sub)
            
        #10.write paramter and score.
        np.savetxt('../froi_work/gss/dice_score_fold_%d'%k,dice_all,fmt='%10.8f',delimiter='  ',newline='\n')
        k+=1
    
    return

def main():
    gss()
    #gss(0.5,0.2,0.3)
    '''
    param = [(1,1),(2,2)]
    for p in param:
        dice = gss(p)
        if best<dice:
            best=dice
            best_p = p
    '''
    return 0

if __name__=='__main__':
    main()

