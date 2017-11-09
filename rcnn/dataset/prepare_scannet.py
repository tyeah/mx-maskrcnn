"""
Prepare bottles Database
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

def pil_imsave(filename, im):
    im = im.astype(np.uint32)
    '''
    if (im.shape) == 2:
        result = Image.fromarray(im, 'I')#.astype(np.int32)
    else:
        result = Image.fromarray(im, 'RGB')#.astype(np.int32)
    result.save('out.bmp')
    plt.imshow(im)
    plt.savefig(filename)
    '''
    io.imsave(filename, im)

def main():
    n_threads = 50
    #blensor_result_path = args.br_path
    blensor_result_path = '/mnt/disks/scannetdata/scannet'
    root_path = '/mnt/disks/scannetdata/maskrcnn'
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

    ins, inst_ids, sem_ids = [], [], []
    num_imgs = 0
    pool = Pool(n_threads)
    for i in ids_:
    #for i, data in enumerate(pool.imap(rotation_depth_map, ids_)):
        ins_path = os.path.join(blensor_result_path, '%s/out' % i)
        ins_ = glob(os.path.join(ins_path, '*.color.jpg'))
        inst_ids_path = os.path.join(blensor_result_path, '%s/instance-filt' % i)
        inst_ids_ = glob(os.path.join(inst_ids_path, '*.png'))
        sem_ids_path = os.path.join(blensor_result_path, '%s/label-filt' % i)
        sem_ids_ = glob(os.path.join(sem_ids_path, '*.png'))
        #print(ins_path, inst_ids_path, sem_ids_path)
        assert len(ins_) == len(inst_ids_) == len(sem_ids_)
        ins += ins_
        inst_ids += inst_ids_
        sem_ids += sem_ids_
        num_imgs += len(ins_)
        print(len(ins_))
    #num_imgs = len(ins)
    print(num_imgs, '-' * 80)
    exit()

    files = {}
    for spl in splits_dict.values():
        files[spl] = open("%s/%s/%s.lst" % (root_path, imglists, spl), 'w')
    for i, (idx, inp, outp) in enumerate(zip(ids, ins, outs)):
        spl = splits_dict[np.sum((splits - (i * 1.0 / num_imgs)) <= 0)]
        # 'city' , 'sequenceNb' , 'frameNb' , 'type' , 'type2' , 'ext'
        # 'house' , 'roomNb' , 'frameNb' , 'type' , 'type2' , 'ext'
        img_path = "%s/%s/%d/%d_000000_000000_gtFine_instanceIds.png" % (img_dir, spl, idx, idx)
        seg_path = "%s/%s/%d/%d_000000_000000_gtFine_instanceIds.png" % (gt_dir, spl, idx, idx)
        line = "%d\t%s\t%s\n" % (idx, img_path, seg_path.replace('instanceIds', 'labelTrainIds'))
        print(line)
        files[spl].write(line)

        ########## dump images #################
        im_inp = read_pgm_xyz(inp)
        im_centroids = load_seg(outp)
        mask = get_labels(im_centroids)
        mask = encode_classid(mask)
        img_full_path = '%s/%s' % (root_path, img_path)
        seg_full_path = '%s/%s' % (root_path, seg_path)
        try:
            #print("????:%s/%s" % (root_path, img_path))
            pil_imsave(img_full_path, im_inp)
            #print("????:%s/%s" % (root_path, img_path))
        except IOError:
            os.makedirs(os.path.dirname(img_full_path))
            pil_imsave(img_full_path, im_inp)
        try:
            pil_imsave(seg_full_path, mask)
        except IOError:
            os.makedirs(os.path.dirname(seg_full_path))
            pil_imsave(seg_full_path, mask)
        ########## dump images #################
    for v in files.values():
        v.close()



if __name__ == "__main__":
    main()
