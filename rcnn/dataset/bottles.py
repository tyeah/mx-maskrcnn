"""
Cityscape Database
"""

import cv2
import os
import numpy as np
import cPickle
import PIL.Image as Image
from imdb import IMDB
from ..processing.bbox_transform import bbox_overlaps
from .cityscape import Cityscape

class Bottles(Cityscape):
    def __init__(self, image_set, root_path, dataset_path):
        """
        fill basic information to initialize imdb
        :param image_set: train or val or trainval or test
        :param root_path: 'cache' and 'rpn_data'
        :param dataset_path: data and results
        :return: imdb object
        """
        super(Cityscape, self).__init__('bottles', image_set, root_path, dataset_path)
        self.image_set = image_set
        self.root_path = root_path
        self.data_path = dataset_path

        self.classes = ['__background__', 'bottle']
        self.class_id = [0, 1]
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images

