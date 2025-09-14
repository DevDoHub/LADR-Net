import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets.real2 import Real2

from .bases import ImageTextDataset, ImageDataset, TextDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler, RandomIdentitySampler_IdUniform
from .market1501 import Market1501
from .rstpreid import RSTPReid
from .cuhkpedes import cuhkpedes
from .msmt17 import MSMT17
from .cuhk03 import Cuhk03
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .mm import MM
from .sda import SDA
__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'mm': MM,
    'cuhk03': Cuhk03,
    'real2': Real2,
    'RSTPReid': RSTPReid,
    "cuhkpedes":cuhkpedes,
    'sda':SDA
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, instructs, pids, camids, viewids , imgs_path = zip(*batch)

    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    
    # 处理 instructs - 将字典列表转换为字典，每个key对应一个tensor
    instructs_batch = {}
    for key in instructs[0].keys():  # 获取所有keys
        # 先stack，然后压缩掉多余的维度
        stacked = torch.stack([inst[key] for inst in instructs], dim=0)
        if stacked.dim() == 3 and stacked.size(1) == 1:  # 如果是 (batch, 1, seq_len)
            stacked = stacked.squeeze(1)  # 压缩为 (batch, seq_len)
        instructs_batch[key] = stacked
    
    return torch.stack(imgs, dim=0), instructs_batch, pids, camids, viewids

def val_collate_fn(batch):
    imgs, instructs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    
    # 处理 instructs - 将字典列表转换为字典，每个key对应一个tensor
    instructs_batch = {}
    for key in instructs[0].keys():  # 获取所有keys
        # 先stack，然后压缩掉多余的维度
        stacked = torch.stack([inst[key] for inst in instructs], dim=0)
        if stacked.dim() == 3 and stacked.size(1) == 1:  # 如果是 (batch, 1, seq_len)
            stacked = stacked.squeeze(1)  # 压缩为 (batch, seq_len)
        instructs_batch[key] = stacked
    
    return torch.stack(imgs, dim=0), instructs_batch, pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    if cfg.DATASETS.NAMES == 'ourapi':
        dataset = OURAPI(root_train=cfg.DATASETS.ROOT_TRAIN_DIR, root_val=cfg.DATASETS.ROOT_VAL_DIR, config=cfg)
    elif cfg.DATASETS.NAMES == 'sda':
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, test_path = cfg.DATASETS.ROOT_TEST_DIR)
    else:
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    if cfg.DATASETS.NAMES == 'real2':
        train_set = ImageDataset(dataset.train, train_transforms, json_list=cfg.JSON_DIR, is_train=True)
        train_set_normal = ImageDataset(dataset.train, val_transforms, json_list=cfg.JSON_DIR, is_train=True)
    else:
        train_set = ImageTextDataset(dataset.train, train_transforms, is_train=True)
        train_set_normal = ImageTextDataset(dataset.train, val_transforms, is_train=True)

    num_classes = dataset.num_train_pids


    if cfg.DATALOADER.SAMPLER in ['softmax_triplet', 'img_triplet']:
        print('using img_triplet sampler')
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=False,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER in ['id_triplet', 'id']:
        print('using ID sampler')
        train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler_IdUniform(dataset.train, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn, drop_last = True,
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))
    
    # val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms, json_list=cfg.JSON_DIR, is_train=False)
    # val_set = ImageDataset(dataset.test, val_transforms, is_train=False)


    # val_loader = DataLoader(
    #     val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
    #     collate_fn=val_collate_fn
    # )
    ds = dataset.test
    val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                val_transforms)
    val_txt_set = TextDataset(ds['caption_pids'],
                                ds['captions'])

    val_img_loader = DataLoader(val_img_set,
                                batch_size=cfg.TEST.IMS_PER_BATCH,
                                shuffle=False,
                                num_workers=num_workers)
    val_txt_loader = DataLoader(val_txt_set,
                                batch_size=cfg.TEST.IMS_PER_BATCH,
                                shuffle=False,
                                num_workers=num_workers)

    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_img_loader, val_txt_loader, len(dataset.test), num_classes
