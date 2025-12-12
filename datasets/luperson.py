import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseImageDataset

import glob
import re
import json
import os.path as osp


class LUPerson(BaseImageDataset):

    dataset_dir = 'LUPerson'

    def __init__(self, root='', test_path = '', verbose=True):
        super(LUPerson, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'lup_lmdb')
        # self.img_test_dir = op.join('./data/market1501/imgs') 

        self.anno_path = op.join('./data/market1501', 'market1501_gpt_v1.json')
        self._check_before_run()

        self.anno_path_train = './data/luperson/merged_data_simple_normalizer.json'
        self._check_before_run()
        
        self.anno_test = self.anno_path

        self.train_annos, self._, self._ = self._split_anno(self.anno_path_train)
        self._, self.test_annos, self.val_annos = self._split_anno(self.anno_test)



        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            print("=> LUPerson loaded")
            self.print_dataset_statistics(self.train, self.test, self.val)



        self.num_train_pids, self.num_train_imgs, self.num_train_image_ids, self.num_train_captions = self.get_imagedata_info(self.train)
        self.num_test_pids,self.num_test_imgs, self.num_test_caption_ids, self.num_test_captions = self.get_imagedata_info(self.test)
        self.num_val_pids,self.num_val_imgs, self.num_val_caption_ids, self.num_val_captions = self.get_imagedata_info(self.val)

    def _split_anno(self, anno_path):
        train_annos, query_annos, gallery_annos = [], [], []
        if isinstance(anno_path, list):
            annos = []
            for path in anno_path:
                annos.extend(read_json(path))
        else:
            annos = read_json(anno_path)
        # try:
        if "luperson" in anno_path:
            for key, anno in annos.items():
                train_annos.append({key: anno})
        # except:
        if "market1501" in anno_path:
            for key, anno in annos.items():
                if 'query' in key :
                    query_annos.append({key: anno})
                elif 'bounding_box_test' in key:
                    gallery_annos.append({key: anno})

        return train_annos, query_annos, gallery_annos   

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        # 检查每个训练标注文件
        if isinstance(self.anno_path, list):
            for path in self.anno_path:
                if not op.exists(path):
                    raise RuntimeError("'{}' is not available".format(path))
        else:
            if not op.exists(self.anno_path):
                raise RuntimeError("'{}' is not available".format(self.anno_path))
        
    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                inner_anno = list(anno.values())[0]
                pid = int(inner_anno['id']-1) # make pid begin from 0nvidia-smi

                pid_container.add(pid)
                img_path = op.join(self.img_dir, list(anno.keys())[0])  # 使用外层字典的键作为图片路径
                captions = inner_anno['captain'] # caption list
                for caption in captions:
                    dataset.append((pid, image_id, img_path, caption))
                image_id += 1
            for idx, pid in enumerate(pid_container):
                # check pid begin from 0 and no break
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            print('----------------------------------'+str(len(dataset)))
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            pattern = re.compile(r'([-\d]+)_c([\d])')
            for anno in annos:
                pid, _ = map(int, pattern.search(str(anno.keys())).groups())
                # pid = int(anno['id'])
                pid_container.add(pid)
                img_path = next(iter(anno.keys()))
                img_paths.append(img_path)
                image_pids.append(pid)
                caption = next(iter(anno.values()))# caption list
                # for caption in caption_list:
                captions.append(caption)
                caption_pids.append(pid)
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container
