import json
import re
from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import logging
from config import cfg
# from utils.simple_tokenizer import SimpleTokenizer
from transformers import BertTokenizer

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
        if type(data) == list:
            pids, image_ids, image_paths, captions = [], [], [], []

            for pid, image_id, image_path, caption in data:
                pids += [pid]
                image_ids += [image_id]
                image_paths += [image_path]
                captions += [caption]
            pids = set(pids)
            image_ids = set(image_ids)
            image_paths = set(image_paths)
            # captions = set(captions)
            num_pids = len(pids)
            num_image_ids = len(image_ids)
            num_imgs = len(image_paths)
            num_captions = len(captions)
            return num_pids, num_imgs, num_image_ids, num_captions
        elif type(data) == dict:
            image_pids, img_paths, caption_pids, captions = [], [], [], []
            image_pids = data['image_pids']
            img_paths = data['img_paths']
            caption_pids = data['caption_pids']
            captions = data['captions']
            return len(set(image_pids)), len(set(img_paths)), len(set(caption_pids)), len(captions)

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, test, val):
        num_train_pids, num_train_imgs, num_train_image_ids, num_train_captions = self.get_imagedata_info(train)
        num_test_pids,num_test_imgs, num_test_caption_ids, num_test_captions = self.get_imagedata_info(test)
        num_val_pids,num_val_imgs, num_val_caption_ids, num_val_captions = self.get_imagedata_info(val)
        logger = logging.getLogger("transreid.check")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------------------------")
        logger.info("  subset   | # ids | # images | # caption ids | # caption ")
        logger.info("  ----------------------------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d} | {:9d}".format(num_train_pids, num_train_image_ids, num_train_pids, num_train_captions))
        logger.info("  test     | {:5d} | {:8d} | {:9d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_caption_ids, num_test_captions))
        logger.info("  val      | {:5d} | {:8d} | {:9d} | {:9d}".format(num_val_pids, num_val_imgs, num_val_caption_ids, num_val_captions))
        logger.info("  ----------------------------------------------------------")

class ImageTextDataset(Dataset):
    # def __init__(self, dataset, transform=None, json_list=None, is_train = False):
    def __init__(self, dataset, transform=None, is_train=True,text_length: int = 77,truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        # attr_file =  open(json_list, 'r', encoding='utf-8')
        # self.attr_dict = json.load(attr_file)
        self.text_length = cfg.MODEL.TEXT_LENGTH
        self.truncate = truncate
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
      
        pid, image_pid, img_path ,caption  = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        # TODO 例子"images/2/3/track_26_n_1056_t_164240.jpg_533060.jpg"
        # img_path需要实际转换
        # attr_item = self.attr_dict[instruct]
        # attribute = pre_caption(caption, 50)

        caption_tokens = self.tokenizer(caption, padding="max_length", truncation=self.truncate, max_length=self.text_length, return_tensors="pt")

        # mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(
        #     caption_tokens["input_ids"]
        # )
        return img, caption_tokens, pid, 0, 0, img_path
        #  return img, pid, camid, trackid,img_path.split('/')[-1]


    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        按照原始 BERT 论文中的概率，随机对语言模型任务中的部分 token 进行掩码（mask）。
        :param tokens: int 列表，已经分词的句子。
        :return: (int 列表, int 列表)，掩码后的 tokens 以及用于 MLM 预测的标签。
         """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability  后面会用这个随机数判断，当前 token 是否有 15% 的概率被选中做掩码（mask）。
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token 概率把 token 替换成 <|mask|>（即 tokens[i] = mask）
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token 10% 概率把 token 替换成一个随机 token（tokens[i] = random.choice(token_range)）
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)#剩下 10% 保持原 token 不变
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)

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


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        按照原始 BERT 论文中的概率，随机对语言模型任务中的部分 token 进行掩码（mask）。
        :param tokens: int 列表，已经分词的句子。
        :return: (int 列表, int 列表)，掩码后的 tokens 以及用于 MLM 预测的标签。
         """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability  后面会用这个随机数判断，当前 token 是否有 15% 的概率被选中做掩码（mask）。
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token 概率把 token 替换成 <|mask|>（即 tokens[i] = mask）
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token 10% 概率把 token 替换成一个随机 token（tokens[i] = random.choice(token_range)）
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)#剩下 10% 保持原 token 不变
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)

class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        # self.text_length = text_length
        # self.truncate = truncate
        self.text_length = cfg.MODEL.TEXT_LENGTH
        self.truncate = truncate
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = self.tokenizer(caption, padding="max_length", truncation=self.truncate, max_length=self.text_length, return_tensors="pt")

        return pid, caption
    
def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result