import os
import argparse

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.plugins import DDPPlugin

import sys
from ipdb import set_trace
# 当运行任意一个python模块(代码文件)时, __file__变量储存的是从当前(执行python命令的目录)到该模块/文件的相对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../shape_assembly'))
from config import get_cfg_defaults
from datasets.baseline.dataloader import ShapeAssemblyDataset
from models.baseline.network_vnn import ShapeAssemblyNet_vnn

def main(cfg):
    # 把整个yaml文件看成一个cfg
    # Initialize model
    data_features = ['src_pc', 'src_rot', 'src_trans', 'tgt_pc', 'tgt_rot', 'tgt_trans', 'category_name', 'cut_name', 'shape_id', 'result_id']
    if cfg.exp.name == 'vnn':
        model = ShapeAssemblyNet_vnn(cfg=cfg, data_features=data_features).cuda()
    elif cfg.exp.name == 'baseline':
        model = ShapeAssemblyNet_WRH(cfg=cfg, data_features=data_features).cuda()
    else:
        raise NotImplementedError
    # Initialize train dataloader
    train_data = ShapeAssemblyDataset(
        data_root_dir=cfg.data.root_dir,
        data_csv_file=cfg.data.train_csv_file,
        data_features=data_features,
        num_points=cfg.data.num_pc_points
    )
    train_data.load_data()

    print('Len of Train Data: ', len(train_data))
    print('Distribution of Train Data\n', str(train_data))
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.exp.batch_size,
        num_workers=cfg.exp.num_workers,
        # persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        drop_last=False
    )

    # Initialize val dataloader
    val_data = ShapeAssemblyDataset(
        data_root_dir=cfg.data.root_dir,
        # data_csv_file=cfg.data.val_csv_file,
        data_csv_file = cfg.data.train_csv_file,
        data_features=data_features,
        num_points=cfg.data.num_pc_points
    )
    val_data.load_data()

    print('Len of Val Data: ', len(val_data))
    print('Distribution of Val Data\n', str(val_data))
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=cfg.exp.batch_size,
        num_workers=cfg.exp.num_workers,
        # persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        drop_last=False
    )


    all_gpus = list(cfg.gpus)
    if len(all_gpus) == 1:
        torch.cuda.set_device(all_gpus[0])

    # Create checkpoint directory
    if not os.path.exists(cfg.exp.log_dir):
        os.makedirs(cfg.exp.log_dir)
    checkpoint_dir = os.path.join(cfg.exp.log_dir, cfg.exp.checkpoint_dir, cfg.exp.name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/total loss",
        mode='min',
        save_top_k=5,
        # period=2,
        dirpath=cfg.exp.log_dir,
        filename=os.path.join('weights', 'best'),
        every_n_epochs=1,
        save_last=True,
        save_weights_only=False
        # filepath=os.path.join(cfg.exp.log_dir, 'weights', 'best')
    )

    # tb_dir = os.path.join(cfg.exp.log_dir, "tb_logs")
    # if not os.path.exists(tb_dir):
    #     os.makedirs(tb_dir)
    logger = TensorBoardLogger(cfg.exp.log_dir, name="tb_logs")

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=list(cfg.exp.gpus),
        # accelerator='ddp',
        # plugins=DDPPlugin(find_unused_parameters=False),
        strategy='ddp_find_unused_parameters_false',
        max_epochs=cfg.exp.num_epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        default_root_dir=checkpoint_dir,
        # distributed_backend="ddp"
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)

    print("Done training...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--cfg_file', default='', type=str)
    parser.add_argument('--gpus', nargs='+', default=-1, type=int)

    args = parser.parse_args()
    # args.cfg_file = './config/train.yml'
    args.cfg_file = os.path.join('./config', args.cfg_file)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    # if args.gpus == -1:
    #     args.gpus = [0, 1, 2, 3]
    cfg.gpus = cfg.exp.gpus

    cfg.freeze()
    print(cfg)
    main(cfg)
