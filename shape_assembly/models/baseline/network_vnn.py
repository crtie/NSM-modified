import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import pytorch_lightning as pl

from pytorch3d.transforms import quaternion_to_matrix
from transforms3d.quaternions import quat2mat
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D

import os
import sys
import copy
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../'))
from models.encoder.pointnet import PointNet
from models.encoder.dgcnn import DGCNN
from models.encoder.vn_dgcnn import VN_DGCNN, VN_DGCNN_corr, VN_DGCNN_New
from models.baseline.regressor_wrh import Regressor_WRH, Regressor_6d, VN_Regressor_6d
import utils
from pdb import set_trace


def bgs(d6s):
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)


class ShapeAssemblyNet_vnn(pl.LightningModule):

    def __init__(self, cfg, data_features):
        super().__init__()
        self.cfg = cfg
        self.encoder = self.init_encoder()
        self.pose_predictor_rot = self.init_pose_predictor_rot()
        self.pose_predictor_trans = self.init_pose_predictor_trans()
        self.data_features = data_features
        self.iter_counts = 0
        self.close_eps = 0.1
        self.L2 = nn.MSELoss() 
        self.R = torch.tensor([[0.26726124, -0.57735027,  0.77151675],
                  [0.53452248, -0.57735027, -0.6172134],
                  [0.80178373,  0.57735027,  0.15430335]], dtype=torch.float64).unsqueeze(0)

    def init_encoder(self):
        if self.cfg.model.encoder == 'dgcnn':
            encoder = DGCNN(feat_dim=self.cfg.model.pc_feat_dim)
        elif self.cfg.model.encoder == 'vn_dgcnn':
            encoder = VN_DGCNN_New(feat_dim=self.cfg.model.pc_feat_dim)
        elif self.cfg.model.encoder == 'pointnet':
            encoder = PointNet(feat_dim=self.cfg.model.pc_feat_dim)
        return encoder

    def init_pose_predictor_rot(self):
        if self.cfg.model.encoder == 'vn_dgcnn':
            pc_feat_dim = self.cfg.model.pc_feat_dim * 2 * 3
        if self.cfg.model.pose_predictor_rot == 'original':
            pose_predictor_rot = Regressor_WRH(pc_feat_dim= pc_feat_dim, out_dim=6)
        elif self.cfg.model.pose_predictor_rot == 'vn':
            pose_predictor_rot = VN_equ_Regressor(pc_feat_dim= pc_feat_dim/3, out_dim=6)

        return pose_predictor_rot

    def init_pose_predictor_trans(self):
        if self.cfg.model.encoder == 'vn_dgcnn':
            pc_feat_dim = self.cfg.model.pc_feat_dim * 2 * 3
        if self.cfg.model.pose_predictor_trans == 'original':
            pose_predictor_trans = Regressor_WRH(pc_feat_dim=pc_feat_dim, out_dim=3)
        elif self.cfg.model.pose_predictor_trans == 'vn':
            pose_predictor_trans = VN_inv_Regressor(pc_feat_dim=pc_feat_dim/3, out_dim=3)
        return pose_predictor_trans

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        return optimizer


    def check_equiv(self, x, R, xR, name):
        mean_diff = torch.mean(torch.abs(torch.matmul(x, R) - xR))
        if mean_diff > self.close_eps:
            print(f'---[Equiv check]--- {name}: {mean_diff}')
        return
    
    def check_inv(self, x, R, xR, name):
        mean_diff = torch.mean(torch.abs(x - xR))
        if mean_diff > self.close_eps:
            print(f'---[Equiv check]--- {name}: {mean_diff}')
        return

    def check_network_property(self, gt_data, pred_data):
        with torch.no_grad():
            B, _, N = gt_data['src_pc'].shape
            R = self.R.float().repeat(B, 1, 1).to(gt_data['src_pc'].device)
            pcs_R = torch.matmul(gt_data['src_pc'].permute(0, 2, 1), R).permute(0, 2, 1)
            pred_data_R = self.forward(pcs_R, gt_data['tgt_pc'])
            
            #! check pointcloud equivariance (should be naturally equivariant)
            # self.check_equiv(gt_data['src_pc'].reshape(B, N, 3), R.permute(0, 2, 1), pcs_R.reshape(B, N, 3), 'pcs')

            #! check equivariance of equivariant features
            equiv_feats = pred_data['Fa']
            equiv_feats_R = pred_data_R['Fa']
            self.check_equiv(equiv_feats, R, equiv_feats_R, 'equiv_feats')

            #! check invariance of invariant features
            inv_feats = pred_data['Ga']
            inv_feats_R = pred_data_R['Ga']
            self.check_inv(inv_feats, R, inv_feats_R, 'inv_feats')

            # if self.cfg.model.pose_predictor_trans == 'vn':
            #     self.check_inv(pred_data['src_trans'], R, pred_data_R['src_trans'], 'trans')

            if self.cfg.model.pose_predictor_rot == 'vn':
                rot = bgs(pred_data['src_rot'].reshape(-1, 2, 3).permute(0, 2, 1)).permute(0, 2, 1)
                rot_R = bgs(pred_data_R['src_rot'].reshape(-1, 2, 3).permute(0, 2, 1)).permute(0, 2, 1)
                # rot = pred_data['src_rot'].reshape(-1, 2, 3)
                # rot_R = pred_data_R['src_rot'].reshape(-1, 2, 3)
                self.check_equiv(rot, R, rot_R, 'rot')
        return


    def forward(self, src_pc, tgt_pc):
        batch_size = src_pc.shape[0]
        num_points = src_pc.shape[2]
        # print(src_pc.shape)
        # print(tgt_pc.shape)
        if self.cfg.model.encoder == 'dgcnn':
            src_point_feat = self.encoder(src_pc)  # (batch_size, pc_feat_dim(512), num_point(1024))
            tgt_point_feat = self.encoder(tgt_pc)  # (batch_size, pc_feat_dim(512), num_point(1024))

            src_feat = torch.mean(src_point_feat, dim=2)
            tgt_feat = torch.mean(tgt_point_feat, dim=2)

        if self.cfg.model.encoder == 'vn_dgcnn':
            Fa, Ga = self.encoder(src_pc)
            Fb, Gb = self.encoder(tgt_pc)
            # set_trace()
            if self.cfg.model.corr_module == 'yes':
                src_feat = torch.bmm(Gb, Fa)
                tgt_feat = torch.bmm(Ga, Fb)
            else:
                src_feat = Fa
                tgt_feat = Fb
        if self.cfg.model.pose_predictor_rot == 'original':
            src_rot = self.pose_predictor_rot(src_feat.reshape(batch_size, -1))
            tgt_rot = self.pose_predictor_rot(tgt_feat.reshape(batch_size, -1))
        else:
            src_rot = self.pose_predictor_rot(src_feat)
            tgt_rot = self.pose_predictor_rot(tgt_feat)
        
        if self.cfg.model.pose_predictor_trans == 'original':
            src_trans = self.pose_predictor_trans(src_feat.reshape(batch_size, -1))
            tgt_trans = self.pose_predictor_trans(tgt_feat.reshape(batch_size, -1))
        else:
            src_trans = self.pose_predictor_trans(src_feat)
            tgt_trans = self.pose_predictor_trans(tgt_feat)
        # 最后的返回值
        pred_dict = {
            'src_rot': src_rot,
            'src_trans': src_trans,
            'tgt_rot': tgt_rot,
            'tgt_trans': tgt_trans,
        }
        if self.cfg.model.encoder == 'vn_dgcnn':
            pred_dict['Fa'] = Fa
            pred_dict['Ga'] = Ga
        return pred_dict

    def compute_point_loss(self, batch_data, pred_data):
        # Point clouds
        src_pc = batch_data['src_pc'].float()  # batch x 3 x 1024
        tgt_pc = batch_data['tgt_pc'].float()  # batch x 3 x 1024

        # Ground truths
        src_quat_gt = batch_data['src_quat'].float()
        tgt_quat_gt = batch_data['tgt_quat'].float()
        src_trans_gt = batch_data['src_trans'].float()  # batch x 3 x 1
        tgt_trans_gt = batch_data['tgt_trans'].float()  # batch x 3 x 1
        # print('src_trans_gt: ', src_trans_gt.shape)

        # Model predictions
        # src_quat_pred = pred_data['src_quat']
        # tgt_quat_pred = pred_data['tgt_quat']
        src_trans_pred = pred_data['src_trans']
        tgt_trans_pred = pred_data['tgt_trans']

        # Source point loss
        transformed_src_pc_pred = quaternion_to_matrix(src_quat_pred) @ src_pc + src_trans_pred  # batch x 3 x 1024
        with torch.no_grad():
            transformed_src_pc_gt = quaternion_to_matrix(src_quat_gt) @ src_pc + src_trans_gt  # batch x 3 x 1024
        # transformed_src_pc_pred = self.batch_quat2mat(src_quat_pred) @ src_pc + src_trans_pred    # batch x 3 x 1024
        # transformed_src_pc_gt = self.batch_quat2mat(src_quat_gt) @ src_pc + src_trans_gt
        src_point_loss = torch.mean(torch.sum((transformed_src_pc_pred - transformed_src_pc_gt) ** 2, axis=1))

        # Target point loss
        transformed_tgt_pc_pred = quaternion_to_matrix(tgt_quat_pred) @ tgt_pc + tgt_trans_pred  # batch x 3 x 1024
        with torch.no_grad():
            transformed_tgt_pc_gt = quaternion_to_matrix(tgt_quat_gt) @ tgt_pc + tgt_trans_gt  # batch x 3 x 1024
        # transformed_tgt_pc_pred = self.batch_quat2mat(tgt_quat_pred) @ tgt_pc + tgt_trans_pred # batch x 3 x 1024
        # transformed_tgt_pc_gt   = self.batch_quat2mat(tgt_quat_gt) @ tgt_pc + tgt_trans_gt     # batch x 3 x 1024
        tgt_point_loss = torch.mean(torch.sum((transformed_tgt_pc_pred - transformed_tgt_pc_gt) ** 2, axis=1))

        # Point loss
        point_loss = (src_point_loss + tgt_point_loss) / 2.0
        return point_loss

    def compute_trans_loss(self, batch_data, pred_data):
        # Ground truths
        src_trans_gt = batch_data['src_trans'].float()  # batch x 3 x 1
        tgt_trans_gt = batch_data['tgt_trans'].float()  # batch x 3 x 1

        # Model predictions
        src_trans_pred = pred_data['src_trans']  # batch x 3 x 1
        tgt_trans_pred = pred_data['tgt_trans']  # batch x 3 x 1

        src_trans_pred = src_trans_pred.unsqueeze(dim=2)
        tgt_trans_pred = tgt_trans_pred.unsqueeze(dim=2)

        # Compute loss
        src_trans_loss = F.l1_loss(src_trans_pred, src_trans_gt)
        tgt_trans_loss = F.l1_loss(tgt_trans_pred, tgt_trans_gt)
        trans_loss = (src_trans_loss + tgt_trans_loss) / 2.0
        return trans_loss

    def compute_rot_loss(self, batch_data, pred_data):
        # Ground truths
        src_R_6d = batch_data['src_rot']
        tgt_R_6d = batch_data['tgt_rot']

        src_R_6d_pred = pred_data['src_rot']
        tgt_R_6d_pred = pred_data['tgt_rot']

        src_rot_loss = torch.mean(utils.get_6d_rot_loss(src_R_6d, src_R_6d_pred))
        tgt_rot_loss = torch.mean(utils.get_6d_rot_loss(tgt_R_6d, tgt_R_6d_pred))
        rot_loss = (src_rot_loss + tgt_rot_loss) / 2.0
        # print(rot_loss)
        return rot_loss

    def recover_R_from_6d(self, R_6d):
        # R_6d is batch * 6
        R = utils.bgs(R_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        # R is batch * 3 * 3
        return R

    def quat_to_eular(self, quat):
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])

        r = R.from_quat(quat)
        euler0 = r.as_euler('xyz', degrees=True)

        return euler0

    def training_step(self, batch_data, batch_idx):
        self.iter_counts += 1
        total_loss, point_loss, rot_loss, trans_loss = self.forward_pass(batch_data, mode='train')
        return total_loss
        # return {'loss': total_loss, 'log': tensorboard_logs}

    def validation_step(self, batch_data, batch_idx):
        total_loss, point_loss, rot_loss, trans_loss = self.forward_pass(batch_data, mode='val')
        return total_loss
        # return {'loss': total_loss, 'log': tensorboard_logs}

    def test_step(self, batch_data, batch_idx):
        total_loss, point_loss, rot_loss, trans_loss = self.forward_pass(batch_data, mode='train')
        return total_loss
        # return {'loss': total_loss, 'log': tensorboard_logs}
    
    def recon_pts(self, batch_data, pred_data):
        pts_src = batch_data['src_pc']
        pts_tgt = batch_data['tgt_pc']
        gt_R_src = self.recover_R_from_6d(batch_data['src_rot'])
        gt_R_tgt = self.recover_R_from_6d(batch_data['tgt_rot'])
        gt_t_src = batch_data['src_trans'].float().squeeze()
        gt_t_tgt = batch_data['tgt_trans'].float().squeeze()

        pred_R_src = self.recover_R_from_6d(pred_data['src_rot'])
        pred_R_tgt = self.recover_R_from_6d(pred_data['tgt_rot'])
        pred_t_src = pred_data['src_trans']
        pred_t_tgt = pred_data['tgt_trans']
        

    def calculate_metrics(self, batch_data, pred_data, mode):
        GD = self.compute_rot_loss(batch_data, pred_data)
        pts_src = batch_data['src_pc']
        pts_tgt = batch_data['tgt_pc']
        gt_R_src = self.recover_R_from_6d(batch_data['src_rot'])
        gt_R_tgt = self.recover_R_from_6d(batch_data['tgt_rot'])
        gt_t_src = batch_data['src_trans'].float().squeeze()
        gt_t_tgt = batch_data['tgt_trans'].float().squeeze()

        pred_R_src = self.recover_R_from_6d(pred_data['src_rot'])
        pred_R_tgt = self.recover_R_from_6d(pred_data['tgt_rot'])
        pred_t_src = pred_data['src_trans']
        pred_t_tgt = pred_data['tgt_trans']
        RMSE_T_src = self.L2(gt_t_src, pred_t_src)
        RMSE_T_tgt = self.L2(gt_t_tgt, pred_t_tgt)
        RMSE_T = (RMSE_T_src + RMSE_T_tgt) / 2.0


        gt_transformed_pts_src = pts_src.permute(0, 2, 1) @ gt_R_src + gt_t_src
        gt_transformed_pts_tgt = pts_tgt.permute(0, 2, 1) @ gt_R_tgt + gt_t_tgt
        pred_transformed_pts_src = pts_src.permute(0, 2, 1) @ pred_R_src + pred_t_src
        pred_transformed_pts_tgt = pts_tgt.permute(0, 2, 1) @ pred_R_tgt + pred_t_tgt
        point_loss = (self.L2(gt_transformed_pts_src, pred_transformed_pts_src) + self.L2(gt_transformed_pts_tgt, pred_transformed_pts_tgt)) / 2.0
        
        
        self.log("{}/GD".format(mode), GD.item(), on_step=True, on_epoch=True, prog_bar=True,
                logger=True, sync_dist=True)
        self.log("{}/RMSE_T".format(mode), RMSE_T.item(), on_step=True, on_epoch=True, prog_bar=True,
                logger=True, sync_dist=True)
        self.log("{}/point_loss".format(mode), point_loss.item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        return

    def forward_pass(self, batch_data, mode):

        pred_data = self.forward(batch_data['src_pc'].float(), batch_data['tgt_pc'].float())
        self.check_network_property(batch_data, pred_data)
        point_loss = 0.0

        # Pose loss
        rot_loss = self.compute_rot_loss(batch_data, pred_data)
        trans_loss = self.compute_trans_loss(batch_data, pred_data)
        # if mode == 'val':
        #     self.calculate_metrics(batch_data, pred_data, mode)
        # Total loss
        total_loss = point_loss + rot_loss + 10 * trans_loss
        print(mode)
        self.log("{}/total loss".format(mode), total_loss.item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        # self.log("{}/point loss".format(mode), point_loss.item(), on_step=True, on_epoch=False, prog_bar=True,
        #          logger=True, sync_dist=True)
        self.log("{}/rot loss".format(mode), rot_loss.item(), on_step=True, on_epoch=False, prog_bar=True,
                 logger=True, sync_dist=True)
        self.log("{}/trans loss".format(mode), trans_loss.item(), on_step=True, on_epoch=False, prog_bar=True,
                 logger=True, sync_dist=True)
        return total_loss, point_loss, rot_loss, trans_loss

