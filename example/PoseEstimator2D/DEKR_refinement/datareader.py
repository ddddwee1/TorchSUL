from config import cfg
from config import update_config
from dataset import make_dataloader

def get_train_dataloader():
    update_config(cfg)
    train_loader, sampler = make_dataloader(cfg, is_train=True, distributed=True)
    return train_loader, sampler
