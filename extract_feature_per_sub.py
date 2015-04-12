#!/usr/bin/env python
#Filename: feature_ext.py
"""
======================================================================
Feature extraction module.
           
Input:     subject zstat map and prior brain image
           zstate.nii.gz and MNI_brain_standard.nii.gz

Output:    sample dataset file per Subject  FFA OFA (left and right)
           train_subject_id.txt
           test_subject_id.txt


Author: dangxiaobin@gmail.com
Date:   2014.12.22
=======================================================================
"""
print (__doc__)

import os
import sys
import re
import time
import nibabel as nib
import numpy as np
import multiprocessing as mul
import functools

def main():
    #data reposition and mask constraint
    database = '/nfs/j3/userhome/dangxiaobin/git/automatic-froi/raw_data'
    contrast = 'face-object'
    sess_list = read_sess_list('./sess')
    print sess_list
    # contain left and right  ffa ofa
    roi_mask = nib.load('./pre_mask_4_roi.nii.gz')
    MNI = nib.load("./MNI_brain.nii.gz")
    mni_data = MNI.get_data()
    mask_data = roi_mask.get_data()

    # get cubiods mask by max/min the spatical range.
    non = mask_data.nonzero()
    coor_r = np.zeros(6)
    coor_r[0] = np.min(non[0])
    coor_r[1] = np.max(non[0])
    coor_r[2] = np.min(non[1])
    coor_r[3] = np.max(non[1])
    coor_r[4] = np.min(non[2])
    coor_r[5] = np.max(non[2])

    #mask_index, mask_img = get_mask(coor_r,mask_data)
    cx = np.median(non[0])
    cy = np.median(non[1])
    cz = np.median(non[2])
    
    cn = [cx,cy,cz]

    mask_index = np.transpose(mask_data.nonzero())
    print mask_index.shape, mask_index, cn

    '''
    brain_m = nib.load("brain_mask.nii.gz")
    data = brain_m.get_data()
    mask_f = data*mask_img
    mask_index = np.transpose(mask_f.nonzero())
    print mask_index.shape
    '''

    #final_mask = roi_mask
    #final_mask._data = mask_img
    #nib.save(final_mask, 'final_mask.nii.gz')

    #output the mask coordinate indexs
    writeout(mask_index)

    st = time.time()
    subject_num = len(sess_list)
    sample_num = len(mask_index) # per subject
    
    sub_stat = {}

    #get neighbor offset 1,2,3 radiud cubois.
    of_1 = get_neighbor_offset(1)
    of_2 = get_neighbor_offset(2)

    #print offset_1.shape, offset_2.shape, offset_3.shape
    vecs = np.array([[1,0,0],
                     [-1,0,0],
                     [0,1,0],
                     [0,-1,0],
                     [0,0,1],
                     [0,0,-1],
                     [2,0,0],
                     [-2,0,0],
                     [0,2,0],
                     [0,-2,0],
                     [0,0,2],
                     [0,0,-2],
                     [3,0,0],
                     [-3,0,0],
                     [0,3,0],
                     [0,-3,0],
                     [0,0,3],
                     [0,0,-3]])
    print vecs,vecs.shape

    # random picked 54 pairs of offset coordinate
    r_os = np.zeros((51,2))
    for i in range(0,51):
        r_os[i][0] = np.random.randint(0,18)
        tmp= np.random.randint(0,18)
        while tmp==r_os[i][0]:
            tmp= np.random.randint(0,18)
        r_os[i][1] = tmp
    print r_os

    ##functions test
    #img = nib.load("MNI_brain.nii.gz")
    #data = img.get_data()
    #m = [[44,22,33],[44,23,33],[44,23,22]]
    #u = get_mean(data,m)
    #print m,u
    #offset = get_neighbor_offset(1)
    #print offset,offset.shape
    #offset = get_neighbor_offset(3)
    #print offset,offset.shape
    #########################################################################

    #feature extraction on train set.
    #save the every subject sample per file named sample_sid.txt

    pool = mul.Pool(30)
    result = pool.map(functools.partial(ext,database=database,mask_index=mask_index,mni=mni_data,of_1=of_1,of_2=of_2,of_vec=vecs,r_os=r_os), sess_list)
    
    print result

    sf = open('sub_filt','wb')
    for s in result:
        if s!= 'F':
            sf.write(s+'\n')
    sf.close()

    print "Feature extraction total time:%s"%(time.time()-st)
    return

def ext(sub=None,database=None,mask_index=None,mni=None,of_1=None,of_2=None,of_vec=None,r_os=None):
    """
    Wrapper function.
    Tested
    """
    print "Starting extraction: %s"%sub
    contrast = 'face-object'
    sub_dir =  os.path.join(database,sub)
    sub_ct_dir = os.path.join(sub_dir,contrast)
    f_list = os.listdir(sub_ct_dir)
    for file in f_list:
        if re.search('zstat1.nii.gz',file):
            x_img =  os.path.join(sub_ct_dir,file)
           # print x_img
        if re.search('_ff.nii.gz',file): 
            y_img =  os.path.join(sub_ct_dir,file)
           # print y_img
    
    #initial the feature array.
    #feature list 
    
    feat_buff = np.zeros(427)
    samples = feature_ext(x_img,y_img,mask_index,mni,of_1,of_2,of_vec,r_os,feat_buff)
    samples = np.array(samples)
    #check the result
    #mask = samples[:,92]!=0
    #print samples[mask,92]
    #output samples as a fiile
    
    #static the sample set
    #print samples
    
    mask1 = samples[:,-4]==1
    mask2 = samples[:,-4]==2
    mask3 = samples[:,-4]==3
    mask4 = samples[:,-4]==4
    
    #print mask1
    #print mask2
    #print mask3
    #print mask4
    #return sub
    
    ofa_l = np.sum(mask1)
    ofa_r = np.sum(mask2)
    ffa_l = np.sum(mask3)
    ffa_r = np.sum(mask4)
    
    roi_stat = "%s\t%s\t%s\t%s\t%s\n"%(sub,ofa_l,ofa_r,ffa_l,ffa_r)
    if ofa_l >= 8 and ofa_r >= 8 and ffa_l >= 8 and ffa_r >= 8:
        print "OFA_l:",ofa_l,"OFA_r:",ofa_r,"FFA_l:",ffa_l,"FFA_r:",ffa_r
        np.savetxt('dataset/feat_%s'%sub,samples,fmt='%10.5f',delimiter='\t',newline='\n')
        with open('roi_static.norm','a') as sf_n:
            sf_n.write(roi_stat)
        return sub
    else:
        print "OFA_l:",ofa_l,"OFA_r:",ofa_r,"FFA_l:",ffa_l,"FFA_r:",ffa_r
        np.savetxt('dataset/ab_feat_%s'%sub,samples,fmt='%10.5f',delimiter='\t',newline='\n')
        with open('roi_static.abnorm','a') as sf_an:
            sf_an.write(roi_stat)
        return 'F'

def feature_ext(sub_x,sub_y,mask_index,mni,os1,os2,of_vec,r_os,feat):
    """
    Feature extraction fucntion.
    """
    #offset vectors
    vecs = of_vec
    #load the zstat image, labeled image and standard brain image
    x = nib.load(sub_x)
    y = nib.load(sub_y)

    x_data = x.get_data()
    y_data = y.get_data()
    s_data = mni
    #center of the region.
    c = [31, 27, 27]
    feat_all = []
    flag = 0
    #one coordinate one feature vec
    st = time.time()

    for i,coor in enumerate(mask_index):
        #print i,coor
        #record the voxel coordinate for reconstruct the prediction.
        feat[424] = coor[0]
        feat[425] = coor[1]
        feat[426] = coor[2]
        #1.local f (0-3:v,x,y,z)
        I = x_data[tuple(coor)]
        #z stat value upper than 2.3
        if I < 2.3:
            continue
        
        feat[0] = I
        x = coor[0] - c[0]
        y = coor[1] - c[1]
        z = coor[2] - c[2]
        r,t,p = spherical_coordinates(x,y,z)

        feat[1] = r
        feat[2] = t
        feat[3] = p

        S = s_data[tuple(coor)]
        feat[213] = S
        
        #context feature:1.neighord smooth
        neigh1 = coor + os1
        neigh2 = coor + os2
        
        mean1 = get_mean(x_data,neigh1)
        mean2 = get_mean(x_data,neigh2)
        feat[4] = mean1
        feat[5] = mean2
        
        mean_s1 = get_mean(s_data,neigh1)
        mean_s2 = get_mean(s_data,neigh2)
        feat[214] = mean_s1
        feat[215] = mean_s2
        
        #2,context feature: compare with offset region
        for j,v in enumerate(vecs):
            p = v + coor
            pn1 = p + os1
            pn2 = p + os2
            

            feat[6+j] = I - get_mean(x_data,pn1)
            feat[24+j] =  mean1 - get_mean(x_data,pn1)
            feat[42+j] =  mean2 - get_mean(x_data,pn2)
            
            feat[216+j]  =  S - get_mean(s_data,pn1)
            feat[234+j] =  mean_s1 - get_mean(s_data,pn1)
            feat[252+j] =  mean_s2 - get_mean(s_data,pn2)

        #3,context feature: compare the two offset regions
        for j,rand in enumerate(r_os):
            vs = vecs[rand[0]]
            vt = vecs[rand[1]]
            p1 = vs + coor
            p2 = vt + coor
            # get 1 and 2 offset neigbor
            pn1 = p1 + os1
            pn2 = p2 + os1
            pnn1 = p1 + os2
            pnn2 = p2 + os2
            
            #print p1,p2
            #print pn1,pn2
            #print pnn1,pnn2

            feat[60+j]  = x_data[tuple(p1)] - x_data[tuple(p2)]
            feat[270+j] = s_data[tuple(p1)] - s_data[tuple(p2)]

            feat[111+j] = get_mean(x_data,pn1) - get_mean(x_data,pn2)
            feat[321+j] = get_mean(s_data,pn1) - get_mean(s_data,pn2)
        
            feat[162+j] = get_mean(x_data,pnn1) - get_mean(x_data,pnn2)
            feat[372+j] = get_mean(s_data,pnn1) - get_mean(s_data,pnn2)
        
        #Label 0/1/2/3/4
        label = y_data[tuple(coor)]
        if label == 1 or label == 2 or label == 3 or label == 4:
            feat[423] = label
            #print i ,'label:',label
        else:
            feat[423] = 0

        if flag == 0:
            feat_all = feat
            flag = 1
        else:
            feat_all = np.vstack((feat_all,feat))
    return  feat_all

def writeout(mask_index):
    """
    Write mask coordinates out.
    Tested
    """
    coorf = open('coordinates','w')
    for c in mask_index:
        coorf.write("%d\t%d\t%d\n"%(c[0],c[1],c[2]))
    coorf.close()
    return 0

def get_neighbor_offset(radius):
    """
    get neighbor offset for generating cubiods.
    Tested
    """
    offsets = [] 
    for x in np.arange(-radius,radius+1):
        for y in np.arange(-radius,radius+1):
            for z in np.arange(-radius,radius+1):
                offsets.append([x,y,z])
    offsets = np.array(offsets)
    return offsets


def output_sess_list(sess,list):
    """
    Output session list for data split
    Tested
    """
    file = open(sess,'w')
    for l in list:
        file.write("%s\n"%l)
    file.close()
    return True

def is_inside(v, shape):
    """
    Is coordinate inside the image.
    Tested
    """
    return ((v[0] >= 0) & (v[0] < shape[0]) &
            (v[1] >= 0) & (v[1] < shape[1]) &
            (v[2] >= 0) & (v[2] < shape[2]))

def get_mean(img,indexs):
    """
    Get mean intensity with in mask. 
    Tested
    """
    intensity = 0
    for c in indexs:
        #print c,img[tuple(c)]
        if is_inside(c,img.shape):
            intensity+= img[tuple(c)] 
    return intensity/len(indexs)

def get_mask(c_r,img):
    """
    Get cuboids region mask as spacial constraint.
    Tested
    """
    indexs = []
    mask = np.zeros_like(img)
    shape = mask.shape
    for x in np.arange(c_r[0],c_r[1]+1):
        for y in np.arange(c_r[2], c_r[3]+1):
            for z in np.arange(c_r[4], c_r[5]+1):    
                tmp = [x,y,z]
                inside = is_inside(tmp,shape)
                if inside :
                    indexs.append(tmp)
                    mask[x][y][z]=1
    return np.array(indexs),mask

def spherical_coordinates(x,y,z):
    """
    Rectangular coordinate to spherical coordinate.
    Tested
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    if z==0:
        theta = np.arctan(np.sqrt(x**2+y**2)/(z+0.000001))
    else:
        theta = np.arctan(np.sqrt(x**2+y**2)/z)
    if x==0:
        phi = np.arctan(float(y)/(x+0.000001))
    else:
        phi = np.arctan(float(y)/x)
       
    return r,theta,phi

def distance(c,t):
    """
    Compute the spacial distance.
    Tested
    """
    return np.sqrt((c[0]-t[0])**2+(c[1]-t[1])**2+(c[2]-t[2])**2)

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

if __name__=="__main__":
    main()
