import torch
import torch.nn as nn

import pytorch_lightning as pl
from models.encoder.vn_layers import *
from pdb import set_trace


class Regressor_WRH(pl.LightningModule):
    def __init__(self, pc_feat_dim, out_dim):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(pc_feat_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU()
        )

        self.head = nn.Linear(128, out_dim)

    def forward(self, x):
        f = self.fc_layers(x)
        output = self.head(f)
        return output


# 这个函数还没改过
class Regressor_6d(pl.LightningModule):
    def __init__(self, pc_feat_dim):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(2*pc_feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )

        # Rotation prediction head
        self.rot_head = nn.Linear(128, 6)

        # Translation prediction head
        self.trans_head = nn.Linear(128, 3)

    def forward(self, x):
        f = self.fc_layers(x)
        quat = self.rot_head(f)
        quat = quat / torch.norm(quat, p=2, dim=1, keepdim=True)
        trans  = self.trans_head(f)
        trans  = torch.unsqueeze(trans, dim=2)
        return quat, trans


# 这个函数还没改过
class VN_Regressor(pl.LightningModule):
    def __init__(self, pc_feat_dim, out_dim):
        super().__init__()
        self.fc_layers = nn.Sequential(
            VNLinear(2*pc_feat_dim, 256),
            nn.BatchNorm1d(256),
            VNNewLeakyReLU(in_channels=256, negative_slope=0.2),
            VNLinear(256, 128),
            nn.BatchNorm1d(128),
            VNNewLeakyReLU(in_channels=128, negative_slope=0.2)
        )

        # Rotation prediction head
        self.head = VNLinear(128, out_dim)

    def forward(self, x):
        f = self.fc_layers(x)
        output = self.head(f)
        return output


# 这个函数还没改过
class VN_Regressor_6d(pl.LightningModule):
    def __init__(self, pc_feat_dim):
        super().__init__()
        self.fc_layers = nn.Sequential(
            VNLinear(1024, 256),
            # VNBatchNorm(256),
            nn.BatchNorm1d(256),
            VNLeakyReLU(in_channels=256, negative_slope=0.2),
            VNLinear(256, 128),
            # VNBatchNorm(128),
            nn.BatchNorm1d(128),
            VNLeakyReLU(in_channels=128, negative_slope=0.2)
        )

        # Rotation prediction head
        self.rot_head = VNLinear(128, 2)

        # Translation prediction head
        self.trans_head = nn.Linear(128*3, 3)

    def forward(self, x):
        f = self.fc_layers(x)

        rot = self.rot_head(f)
        rot = rot / torch.norm(rot, p=2, dim=1, keepdim=True)
        trans  = self.trans_head(f.reshape(-1, 128*3))
        trans  = torch.unsqueeze(trans, dim=2)
        return rot, trans