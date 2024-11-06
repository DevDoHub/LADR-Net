import json
import re
from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import logging
from config import cfg
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if cfg.DATASETS.NAMES == 'real2':
        img_path = osp.join(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.NAMES, img_path)

    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        if cfg.DATASETS.NAMES == 'real2':
            for _, _, pid, camid, trackid in data:
                pids += [pid]
                cams += [camid]
                tracks += [trackid]
            pids = set(pids)
            cams = set(cams)
            tracks = set(tracks)
            num_pids = len(pids)
            num_cams = len(cams)
            num_imgs = len(data)
            num_views = len(tracks)
            return num_pids, num_imgs, num_cams, num_views
        else:
            for _, pid, camid, trackid in data:
                pids += [pid]
                cams += [camid]
                tracks += [trackid]
            pids = set(pids)
            cams = set(cams)
            tracks = set(tracks)
            num_pids = len(pids)
            num_cams = len(cams)
            num_imgs = len(data)
            num_views = len(tracks)
            return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)
        logger = logging.getLogger("transreid.check")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # cameras")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        logger.info("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        logger.info("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        logger.info("  ----------------------------------------")

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, json_list=None, is_train = False):
        self.dataset = dataset
        self.transform = transform
        
        attr_file =  open(json_list, 'r', encoding='utf-8')
        self.attr_dict = json.load(attr_file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, instruct, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        # TODO 例子"images/2/3/track_26_n_1056_t_164240.jpg_533060.jpg"
        # img_path需要实际转换
        attr_item = self.attr_dict[instruct]
        attribute = pre_caption(attr_item, 50)

        return img, attribute, pid, camid, trackid, img_path
        #  return img, pid, camid, trackid,img_path.split('/')[-1]


def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')
    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption
