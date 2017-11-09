"""
Prepare scannet Database
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from glob import glob

import os, sys
import numpy as np
from scipy import misc

sys.path.append('../../neuralnets')
from Dataset import load_seg
from Loader import read_pgm_xyz
from PIL import Image
#import matplotlib
#matplotlib.use('agg')
#from matplotlib import pyplot as plt
from skimage import io

from multiprocessing import Pool


def mask_imsave(filename, im):
    # used to save instance labels
    im = im.astype(np.uint32)
    io.imsave(filename, im)

def check_save(filename, img, method=pil_imsave):
    try:
        method(filename, img)
    except IOError:
        os.makedirs(os.path.dirname(img_full_path))
        method(filename, img)


def get_labels(out_s):
    semantic_id_bottles = 1
    unique_centroids, mask = np.unique(out_s.reshape(-1, out_s.shape[-1]), return_inverse=True, axis=0)
    mask = mask.reshape(out_s.shape[0], out_s.shape[1])
    bg_idx = np.where(np.apply_along_axis(lambda x: (np.all([xx == 0. for xx in x])), 1, unique_centroids))[0][0]
    bg_mask = (mask == bg_idx)
    replace_mask = (mask == 0)
    mask[bg_mask] = 0
    mask[replace_mask] = bg_idx
  
    return mask

def check_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def encode_classid(mask):
    # we only have one class (bottle) except background
    mask[mask > 0] += 1000
    return mask

blensor_result_path = '/mnt/disks/scannetdata/scannet'

def get_paths(i):
    ins_path = os.path.join(blensor_result_path, '%s/out' % i)
    ins_ = glob(os.path.join(ins_path, '*.color.jpg'))
    inst_ids_path = os.path.join(blensor_result_path, '%s/instance-filt' % i)
    inst_ids_ = glob(os.path.join(inst_ids_path, '*.png'))
    sem_ids_path = os.path.join(blensor_result_path, '%s/label-filt' % i)
    sem_ids_ = glob(os.path.join(sem_ids_path, '*.png'))
    return ins_, inst_ids_, sem_ids_


def init_pool(l):
    global idx
    idx = 0
    global lock
    lock = l

def worker(i, spl, img_dir, gt_dir, files):
    # i is like scene0034_01, sceneid_roomNb
    ins_path = os.path.join(blensor_result_path, '%s/out' % i)
    inst_ids_path = os.path.join(blensor_result_path, '%s/instance-filt' % i)
    sem_ids_path = os.path.join(blensor_result_path, '%s/label-filt' % i)
    ins_ = glob(os.path.join(ins_path, '*.color.jpg')) # .../out/frame-000000.color.jpg, frame_id = ss[-16:-10]
    #inst_ids_ = glob(os.path.join(inst_ids_path, '*.png'))
    #sem_ids_ = glob(os.path.join(sem_ids_path, '*.png'))

    for in_ in ins_:
        fid = int(ins[-16:-10])
        inst_id_ = os.path.join(inst_ids_path, "%d.png" % fid)
        sem_id_ = os.path.join(sem_ids_path, "%d.png" % fid)
        if not (os.path.exists(inst_id_) and os.path.exists(sem_id_)):
            continue
        # 'house' , 'roomNb' , 'frameNb' , 'type' , 'type2' , 'ext'
        img_path = "%s/%s/%s/%s_leftImg8bit.tiff" % (img_dir, spl, i, i)
        seg_path = "%s/%s/%d/%d_000000_000000_gtFine_instanceIds.png" % (gt_dir, spl, i, i)
        lock.acquire()
        line = "%d\t%s\t%s\n" % (idx, img_path, seg_path.replace('instanceIds', 'labelTrainIds'))
        idx += 1
        files[spl].write(line)
        lock.release()
        print(line)

        ########## dump images #################
        '''
        Need to change:

        scannet: 
        in_, inst_id_, sem_id_ are path to input image, instance id map, semantic id map correspondingly
        im_inp is load from in_, can be simply copied if shapes match, otherwise need to be resized
        inst_mask is instance id labels loaded from inst_id_
        sem_mask is sematice id labels loaded from sem_id_
        mask[i, j] = sem_mask[i, j] * 1000 + inst_mask[i, j]
        '''
        im_inp = read_pgm_xyz(inp)
        im_centroids = load_seg(outp)
        mask = get_labels(im_centroids)
        mask = encode_classid(mask)
        '''
        Need to change:
        '''
        img_full_path = '%s/%s' % (root_path, img_path)
        seg_full_path = '%s/%s' % (root_path, seg_path)
        check_save(img_full_path, im_inp, io.imsave)
        check_save(seg_full_path, mask, mask_imsave) 
        ########## dump images #################

def main():
    n_proc = 50
    #blensor_result_path = args.br_path
    root_path = 'data/scannet'
    gt_dir = "gtFine"  
    img_dir ="leftImg8bit"
    imglists = "imglists"  
    dirs = [gt_dir, img_dir, imglists]
    splits = [0.5, 0.3, 0.2] # train/val/test
    splits = np.cumsum(splits)
    splits_dict = {0: "train", 1: "val", 2: "test"}

    check_mkdir(root_path)
    for i in range(len(dirs)):
        d = os.path.join(root_path, dirs[i])
        check_mkdir(d)
        for spl in splits_dict.values():
            check_mkdir(os.path.join(d, spl))
    ids_ = os.listdir(blensor_result_path) # eg. scene0041_01

    l = multiprocessing.Lock()
    pool = Pool(n_proc, initializer=init_pool, initargs=(l,))



    files = {}
    for spl in splits_dict.values():
        files[spl] = open("%s/%s/%s.lst" % (root_path, imglists, spl), 'w')

    len_ids = len(ids_)
    workers = {}
    for spl in splits_dict.values():
        workers['train'] = lambda i: worker(i, spl, img_dir, gt_dir, files)
    pool.imap(worker, ids_[:splits[0] * len_ids])
    pool.imap(worker, ids_[splits[0] * len_ids: (splits[0] + splits[1])* len_ids])
    pool.imap(worker, ids_[(splits[0] + splits[1])* len_ids:])
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
