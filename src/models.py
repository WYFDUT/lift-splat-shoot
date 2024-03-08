"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D # D = 41
        self.C = C # C = 64

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        # TODO: depthnet明显可以改进，BEVDepth的作者就关注了这一点
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        # 估计深度方向的概率分布并输出特征图每个位置的语义特征 (用64维的特征表示），整个过程用1x1卷积层实现
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D, :, :])
        # 利用得到的深度方向的概率密度和语义特征通过外积运算构建图像特征点云
        # (bs*N,1,41,H/16,W/16) x (bs*N,64,1,H/16,W/16)
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C), :, :].unsqueeze(2)
        # new_x (bs*N,41,64,H/16,W/16)
        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # TODO: Learn to use efficient net
        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        # endpoints['reduction_5'] (bs*N,112,H/16,W/16) endpoints['reduction_4'] (bs*N,320,H/32,W/32)
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        # x (bs*N,320+112,H/32,W/32)
        return x

    def forward(self, x):
        # x bs*6 x C x H x W
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        '''
        xbound=[-50.0, 50.0, 0.5],
        ybound=[-50.0, 50.0, 0.5],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[4.0, 45.0, 1.0],
        '''

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64

        self.frustum = self.create_frustum() # D x H x W x 3  41 x 8 x 22 x 3 

        self.D, _, _, _ = self.frustum.shape # self.D = 41
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    def create_frustum(self):
        # make grid in image plane
        # 创建视锥
        '''
        生成视锥 
        a) 生成视锥 需要注意的是，生成的锥点，其位置是基于图像坐标系的，同时锥点是图像特征上每个单元格映射回原始图像的位置。生成方式如下：
        '''
        ogfH, ogfW = self.data_aug_conf['final_dim'] # 128 352
        fH, fW = ogfH // self.downsample, ogfW // self.downsample # 8 22
        # ds.shape: 41*8*22
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape # D = 41
        # xs.shape 41*8*22
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        # ys.shape 41*8*22
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        # 41*8*22*3
        frustum = torch.stack((xs, ys, ds), -1)
        # frustum[i,j,k,0]就是(i,j)位置，深度为k的像素的宽度方向上的栅格坐标xs
        return nn.Parameter(frustum, requires_grad=False) # Learnable Parameter

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        # 锥点由图像坐标系向自车坐标系进行坐标转化
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        # trans：由相机坐标系->车身坐标系的平移矩阵，trans=(bs, N, 3)
        B, N, _ = trans.shape 

        # undo post-transformation
        # B x N x D x H x W x 3
        # self.frustum D x H x W x 3
        # post_trans：由图像增强引起的平移矩阵，post_trans=(bs, N, 3)
        # 这边代码利用了张量的广播机制，points.size->B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化   R-1*(C-T)  R-1*((xs,ys,ds)-(x0,y0,z0))
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        # Z[u,v,1] = KP
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        # TODO: Why the format is different from  P=RPw+T
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        '''
        points = points - trans.view(B, N, 1, 1, 1, 3, 1)
        combine = torch.inverse(rots).matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        '''
        # 每个batch中的每个环视相机图像特征点，其在不同深度下位置对应在ego坐标系下的坐标
        # points->B x N x D x H x W x 3
        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape
        # bs*6 x C x H x W
        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)
        # x (B, N, D, H/16, W/16, C)
        return x

    def voxel_pooling(self, geom_feats, x):
        """
        Attention
        这里有一个新的概念叫体素 voxel
        """
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        # x (B*N*D*H*W, C) 图像特征点云
        # geom_feats (B x N x D x H x W x 3) ego坐标系的点
        x = x.reshape(Nprime, C)

        # flatten indices
        # 世界坐标转化为体素坐标
        # 转化的公式：图像坐标 = 世界坐标系下的（位置 - 原点位置）/ 像素间隔
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        # geom_feats (B*N*D*H*W, 3) 
        geom_feats = geom_feats.view(Nprime, 3)
        '''
        torch.full([2,3],2.0)
        tensor([[2., 2., 2.,
                 2., 2., 2.]])
        '''
        # batch_ix (B*N*D*H*W, 1) 每个点对应于哪个batch
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        # geom_feats (B*N*D*H*W, 4)
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        # 给B*N*D*H*W个体素点编了rank的码
        # 确保临近点rank邻近
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        # 这样就把属于相同BEV pillar的体素放在相邻位置，得到点云在体素中的索引。
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        # trick detail LSS解读 - 高毅鹏的文章 - 知乎 https://zhuanlan.zhihu.com/p/667012159
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        # final:bsx64x1x200x200
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        # 将对应点的点云特征代入
        # 需要注意的是，geom_feats[:, 3]索引不会超过bs-1，geom_feats[:, 2]索引不会超过0，
        # geom_feats[:, 0]、geom_feats[:, 1]同理
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z 消除z dim
        # final:bsx64x200x200
        # torch.squeeze(final, dim=2)
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        # geom (B x N x D x H x W x 3)
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        # x (B, N, D, H/16, W/16, C)
        x = self.get_cam_feats(x)
        
        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)