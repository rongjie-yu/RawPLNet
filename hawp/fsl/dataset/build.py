import torch
from .transforms import *
from . import train_dataset
from ..config.paths_catalog import DatasetCatalog
from . import test_dataset

def build_transform(cfg):
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                                           cfg.DATASETS.IMAGE.PIXEL_STD,
                                           cfg.DATASETS.IMAGE.TO_255)
        ]
    )

    if cfg.MODEL.NAME == "PointLine":
        transforms = Compose(
            [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                        cfg.DATASETS.IMAGE.WIDTH),
            ToTensor()
            ]
        )

    return transforms
def build_train_dataset(cfg, return_rgb=False, step_counter=None):
    assert len(cfg.DATASETS.TRAIN) == 1
    name = cfg.DATASETS.TRAIN[0]
    dargs = DatasetCatalog.get(name)

    factory = getattr(train_dataset,dargs['factory'])
    args = dargs['args']
    args['augmentation'] = cfg.DATASETS.AUGMENTATION
    args['raw_config'] = None if return_rgb else cfg.DATASETS.RAW
    args['return_rgb'] = return_rgb
    args['step_counter'] = step_counter
    args['transform'] = Compose(
                                [Resize(cfg.DATASETS.IMAGE.HEIGHT,
                                        cfg.DATASETS.IMAGE.WIDTH,
                                        cfg.DATASETS.TARGET.HEIGHT,
                                        cfg.DATASETS.TARGET.WIDTH),
                                 ToTensor(),
                                 Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                                           cfg.DATASETS.IMAGE.PIXEL_STD,
                                           cfg.DATASETS.IMAGE.TO_255)])

    if cfg.MODEL.NAME == "PointLine":
        args['transform'] = Compose(
                                    [Resize(cfg.DATASETS.IMAGE.HEIGHT,
                                            cfg.DATASETS.IMAGE.WIDTH,
                                            cfg.DATASETS.TARGET.HEIGHT,
                                            cfg.DATASETS.TARGET.WIDTH),
                                    ToTensor()])


    dataset = factory(**args)
    
    dataset = torch.utils.data.DataLoader(dataset,
                                          batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                          collate_fn=train_dataset.collate_fn,
                                          shuffle = True,
                                          num_workers = cfg.DATALOADER.NUM_WORKERS)
    return dataset

def build_test_dataset(cfg):
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                                           cfg.DATASETS.IMAGE.PIXEL_STD,
                                           cfg.DATASETS.IMAGE.TO_255)
        ]
    )

    if cfg.MODEL.NAME == "PointLine":
        transforms = Compose(
            [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                        cfg.DATASETS.IMAGE.WIDTH),
            ToTensor()
            ]
        )

    datasets = []
    for name in cfg.DATASETS.TEST:
        dargs = DatasetCatalog.get(name)
        factory = getattr(test_dataset,dargs['factory'])
        args = dargs['args']
        args['transform'] = transforms
        dataset = factory(**args)
        dataset = torch.utils.data.DataLoader(
            dataset,  batch_size = 1,
            collate_fn = dataset.collate_fn,
            num_workers = cfg.DATALOADER.NUM_WORKERS,
        )
        datasets.append((name,dataset))
    return datasets
