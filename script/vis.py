import os
import sys
import copy
import argparse
import importlib
import trimesh
import trimesh
import numpy as np
from tqdm import tqdm
import open3d as o3d
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from pdb import set_trace
from torch.utils.data import DataLoader
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../shape_assembly'))
from config import get_cfg_defaults
from datasets.baseline.dataloader_wrh import ShapeAssemblyDataset
# from models.baseline.network_vnn import ShapeAssemblyNet_vnn
from models.baseline.network_wrh import ShapeAssemblyNet_WRH

@torch.no_grad()
def visualize(cfg):
    # Initialize model
    data_features = ['src_pc', 'src_rot', 'src_trans', 'tgt_pc', 'tgt_rot', 'tgt_trans', 'category_name', 'cut_name', 'shape_id', 'result_id', 'src_mesh', 'tgt_mesh']

    # model = ShapeAssemblyNet_vnn(cfg=cfg, data_features=data_features).cuda()
    model = ShapeAssemblyNet_WRH(cfg=cfg, data_features=data_features).cuda()
    if cfg.exp.weight_file is not None:
        ckp = torch.load(cfg.exp.weight_file, map_location='cpu')
        model.load_state_dict(ckp['state_dict'])
    # Initialize dataloaders
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
        shuffle=False,
        drop_last=False
    )
    save_dir = os.path.join(
        os.path.dirname(cfg.exp.weight_file), 'vis')
    # save some predictions for visualization
    vis_lst, loss_lst = [], []
    for batch in tqdm(train_loader):
        for i in range(10): #! 存10帧
            q1 = torch.tensor(R.from_euler('xyz', [0, 0, i*36], degrees=True).as_matrix()).unsqueeze(0).repeat(batch['src_pc'].shape[0], 1, 1)
            q2 = torch.tensor(R.from_euler('xyz', [0, i*36, 0 ], degrees=True).as_matrix()).unsqueeze(0).repeat(batch['src_pc'].shape[0], 1, 1)
            batch['src_pc'] = torch.bmm(q1.float(), batch['src_pc'].float() )
            batch['tgt_pc'] = torch.bmm(q2.float(),batch['tgt_pc'].float())
            out_dict = model.forward(batch['src_pc'].cuda(), batch['tgt_pc'].cuda())
            #out_dict: 'src_rot' 'src_trans' 'tgt_rot''tgt_trans'
            cur_save_dir = os.path.join(save_dir, batch['category_name'][0],batch['shape_id'][0], batch['cut_name'][0], batch['result_id'][0])
            if not os.path.exists(cur_save_dir):
                os.makedirs(cur_save_dir)

            src_mesh = trimesh.load(batch['src_mesh'][0])
            src_verts = np.array(src_mesh.vertices).reshape(-1, 3)
            tgt_mesh = trimesh.load(batch['tgt_mesh'][0])
            tgt_verts = np.array(tgt_mesh.vertices).reshape(-1, 3)
            src_mesh.export(os.path.join(cur_save_dir, f'partA_{i}.obj'))
            tgt_mesh.export(os.path.join(cur_save_dir, f'partB_{i}.obj'))
            # meshes[1].vertices = o3d.utility.Vector3dVector(
                        #             vertices)
            gt_src_rot = model.recover_R_from_6d(batch['src_rot'][0]).squeeze().cpu().numpy() # 3x3
            gt_tgt_rot = model.recover_R_from_6d(batch['tgt_rot'][0]).squeeze().cpu().numpy() # 3x3
            gt_src_trans = batch['src_trans'][0].squeeze().cpu().numpy() # 3
            gt_tgt_trans = batch['tgt_trans'][0].squeeze().cpu().numpy() # 3

            pred_src_rot = model.recover_R_from_6d(out_dict['src_rot'][0]).squeeze().cpu().numpy() # 3x3
            pred_tgt_rot = model.recover_R_from_6d(out_dict['tgt_rot'][0]).squeeze().cpu().numpy() # 3x3
            pred_src_trans = out_dict['src_trans'][0].squeeze().cpu().numpy() # 3
            pred_tgt_trans = out_dict['tgt_trans'][0].squeeze().cpu().numpy() # 3

            input_vert_src = ((src_verts + gt_src_trans) @ gt_src_rot ) @ np.array(q1[0])
            input_vert_tgt = ((tgt_verts + gt_tgt_trans) @ gt_tgt_rot ) @ np.array(q2[0])
            src_mesh.vertices = o3d.utility.Vector3dVector(input_vert_src)
            tgt_mesh.vertices = o3d.utility.Vector3dVector(input_vert_tgt)
            src_mesh.export(os.path.join(cur_save_dir, f'input_partA_{i}.obj'))
            tgt_mesh.export(os.path.join(cur_save_dir, f'input_partB_{i}.obj'))
            
            pred_vert_src = (input_vert_src @ pred_src_rot.T) -  pred_src_trans
            pred_vert_tgt = (input_vert_tgt @ pred_tgt_rot.T) -  pred_tgt_trans
            src_mesh.vertices = o3d.utility.Vector3dVector(pred_vert_src)
            tgt_mesh.vertices = o3d.utility.Vector3dVector(pred_vert_tgt)
            src_mesh.export(os.path.join(cur_save_dir, f'pred_partA_{i}.obj'))
            tgt_mesh.export(os.path.join(cur_save_dir, f'pred_partB_{i}.obj'))
        set_trace()






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization script')
    parser.add_argument('--cfg_file', required=True, type=str, help='.py')
    parser.add_argument('--category', type=str, default='', help='data subset')
    parser.add_argument('--min_num_part', type=int, default=-1)
    parser.add_argument('--gpus', nargs='+', default=[3], type=int)
    parser.add_argument('--max_num_part', type=int, default=-1)
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--vis', type=int, default=-1, help='visualization')
    args = parser.parse_args()

    args.cfg_file = os.path.join('./config', args.cfg_file)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)
    cfg.exp.gpus = args.gpus
    # manually increase batch_size according to the number of GPUs in DP
    # not necessary in DDP because it's already per-GPU batch size
    if len(cfg.exp.gpus) > 1 and parallel_strategy == 'dp':
        cfg.exp.batch_size *= len(cfg.exp.gpus)
        cfg.exp.num_workers *= len(cfg.exp.gpus)
    if args.category:
        cfg.data.category = args.category
    if args.min_num_part > 0:
        cfg.data.min_num_part = args.min_num_part
    if args.max_num_part > 0:
        cfg.data.max_num_part = args.max_num_part
    if args.weight:
        cfg.exp.weight_file = args.weight
    else:
        assert cfg.exp.weight_file, 'Please provide weight to test'

    cfg_backup = copy.deepcopy(cfg)
    cfg.freeze()
    print(cfg)

    if not args.category:
        args.category = 'all'
    visualize(cfg)
