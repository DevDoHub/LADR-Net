# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle


class Real2(BaseImageDataset):
    """
    Real2
    https://github.com/Chenhaobin/COCAS-plus
    """
    dataset_dir = 'real2'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(Real2, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir,'datalist', 'gallery_attr.txt')
        self.query_dir = osp.join(self.dataset_dir,'datalist', 'query_attr.txt')
        self.gallery_dir = osp.join(self.dataset_dir,'datalist', 'gallery_attr.txt')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Real2 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        label_base = 0
        list_lines_all = [] 
        list_lines = []
        sub_dataset_pid = 0
        sub_dataset_pid_list = dict()

        with open(dir_path) as f:
            for line in f.readlines():
                info = line.strip('\n').split(" ")
                imgs = info[0]
                clothes = info[1]
                if info[2] not in sub_dataset_pid_list.keys():
                    pids = label_base + sub_dataset_pid
                    sub_dataset_pid_list[info[2]] = pids
                    sub_dataset_pid += 1
                else:
                    pids = sub_dataset_pid_list[info[2]]
                cids = info[3]
                if len(info) > 4:
                    cams = info[4]
                    # 将信息拆分为元组并添加到list_lines中
                    list_lines.append((imgs, clothes, pids, int(cams), int(cids)))
                else:
                    # 将信息拆分为元组并添加到list_lines中
                    list_lines.append((imgs, clothes, pids, cids))

        label_base = label_base + sub_dataset_pid # update label_base

        list_lines_all.extend(list_lines)

        return list_lines_all#TODO
        # img_paths = glob.glob(osp.join(dir_path, '**', '*.jpg'))
        # pattern = re.compile(r'([-\d]+)_c(\d)')
        # pid_container = set()
        # for img_path in sorted(img_paths):
        #     pid, _ = map(int, pattern.search(img_path).groups())
        #     if pid == -1: continue  # junk images are just ignored
        #     pid_container.add(pid)
        # pid2label = {pid: label for label, pid in enumerate(pid_container)}
        # dataset = []
        # for img_path in sorted(img_paths):
        #     pid, camid = map(int, pattern.search(img_path).groups())
        #     if pid == -1: continue  # junk images are just ignored
        #     assert 0 <= pid <= 1501  # pid == 0 means background
        #     assert 1 <= camid <= 6
        #     camid -= 1  # index starts from 0
        #     if relabel: pid = pid2label[pid]

        #     dataset.append((img_path, self.pid_begin + pid, camid, 1))
        # return dataset
