import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
# from utils.model_utils import *
from model_utils import *
from models.pcn import PCN_encoder
import open3d as o3d
import os
from train_utils import *


device = 'cuda'
# proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
# import pointnet2_utils as pn2

# from utils.mm3d_pn2 import three_interpolate, furthest_point_sample, gather_points, grouping_operation
# from ..utils import three_interpolate, furthest_point_sample, gather_points, grouping_operation
sys.path.append("../utils")
from mmcv.ops import three_interpolate, furthest_point_sample, gather_points, grouping_operation



class SA_module(nn.Module):
    def __init__(self, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, k=16):
        super(SA_module, self).__init__()
        self.share_planes = share_planes
        self.k = k
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, mid_planes, kernel_size=1)

        self.conv_w = nn.Sequential(nn.ReLU(inplace=False),
                                    nn.Conv2d(rel_planes * (k + 1), mid_planes // share_planes, kernel_size=1,
                                              bias=False),
                                    nn.ReLU(inplace=False),
                                    nn.Conv2d(mid_planes // share_planes, k * mid_planes // share_planes,
                                              kernel_size=1))
        self.activation_fn = nn.ReLU(inplace=False)

        self.conv_out = nn.Conv2d(mid_planes, out_planes, kernel_size=1)

    def forward(self, input):
        x, idx = input
        batch_size, _, _, num_points = x.size()
        identity = x  # B C 1 N
        x = self.activation_fn(x)
        xn = get_edge_features(x, idx)  # B C K N
        x1, x2, x3 = self.conv1(x), self.conv2(xn), self.conv3(xn)

        x2 = x2.view(batch_size, -1, 1, num_points).contiguous()  # B kC 1 N
        w = self.conv_w(torch.cat([x1, x2], 1)).view(batch_size, -1, self.k, num_points)
        w = w.repeat(1, self.share_planes, 1, 1)
        out = w * x3
        out = torch.sum(out, dim=2, keepdim=True)

        out = self.activation_fn(out)
        out = self.conv_out(out)  # B C 1 N
        out += identity
        return [out, idx]


class Folding(nn.Module):
    def __init__(self, input_size, output_size, step_ratio, global_feature_size=1024, num_models=1):
        super(Folding, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.step_ratio = step_ratio
        self.num_models = num_models

        self.conv = nn.Conv1d(input_size + global_feature_size + 2, output_size, 1, bias=True)

        sqrted = int(math.sqrt(step_ratio)) + 1
        for i in range(1, sqrted + 1).__reversed__():
            if (step_ratio % i) == 0:
                num_x = i
                num_y = step_ratio // i
                break

        grid_x = torch.linspace(-0.2, 0.2, steps=num_x)
        grid_y = torch.linspace(-0.2, 0.2, steps=num_y)

        x, y = torch.meshgrid(grid_x, grid_y)  # x, y shape: (2, 1)
        self.grid = torch.stack([x, y], dim=-1).view(-1, 2)  # (2, 2)

    def forward(self, point_feat, global_feat):
        batch_size, num_features, num_points = point_feat.size()
        point_feat = point_feat.transpose(1, 2).contiguous().unsqueeze(2).repeat(1, 1, self.step_ratio, 1).view(
            batch_size,
            -1, num_features).transpose(1, 2).contiguous()
        global_feat = global_feat.unsqueeze(2).repeat(1, 1, num_points * self.step_ratio).repeat(self.num_models, 1, 1)
        grid_feat = self.grid.unsqueeze(0).repeat(batch_size, num_points, 1).transpose(1, 2).contiguous().to(device)
        features = torch.cat([global_feat, point_feat, grid_feat], axis=1)
        features = F.relu(self.conv(features))
        return features


class Linear_ResBlock(nn.Module):
    def __init__(self, input_size=1024, output_size=256):
        super(Linear_ResBlock, self).__init__()
        self.conv1 = nn.Linear(input_size, input_size)
        self.conv2 = nn.Linear(input_size, output_size)
        self.conv_res = nn.Linear(input_size, output_size)

        self.af = nn.ReLU(inplace=False)

    def forward(self, feature):
        return self.conv2(self.af(self.conv1(self.af(feature)))) + self.conv_res(feature)


class SK_SA_module(nn.Module):
    def __init__(self, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, k=[10, 20], r=2, L=32):
        super(SK_SA_module, self).__init__()

        self.num_kernels = len(k)
        d = max(int(out_planes / r), L)

        self.sams = nn.ModuleList([])

        for i in range(len(k)):
            self.sams.append(SA_module(in_planes, rel_planes, mid_planes, out_planes, share_planes, k[i]))

        self.fc = nn.Linear(out_planes, d)
        self.fcs = nn.ModuleList([])

        for i in range(len(k)):
            self.fcs.append(nn.Linear(d, out_planes))

        self.softmax = nn.Softmax(dim=1)
        self.af = nn.ReLU(inplace=False)

    def forward(self, input):
        x, idxs = input
        assert (self.num_kernels == len(idxs))
        for i, sam in enumerate(self.sams):
            fea, _ = sam([x, idxs[i]])
            fea = self.af(fea)
            fea = fea.unsqueeze(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)

        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)

        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return [fea_v, idxs]


class SKN_Res_unit(nn.Module):
    def __init__(self, input_size, output_size, k=[10, 20], layers=1):
        super(SKN_Res_unit, self).__init__()
        self.conv1 = nn.Conv2d(input_size, output_size, 1, bias=False)
        self.sam = self._make_layer(output_size, output_size // 16, output_size // 4, output_size, int(layers), 8, k=k)
        self.conv2 = nn.Conv2d(output_size, output_size, 1, bias=False)
        self.conv_res = nn.Conv2d(input_size, output_size, 1, bias=False)
        self.af = nn.ReLU(inplace=False)

    def _make_layer(self, in_planes, rel_planes, mid_planes, out_planes, blocks, share_planes=8, k=16):
        layers = []
        for _ in range(0, blocks):
            layers.append(SK_SA_module(in_planes, rel_planes, mid_planes, out_planes, share_planes, k))
        return nn.Sequential(*layers)

    def forward(self, feat, idx):
        x, _ = self.sam([self.conv1(feat), idx])
        x = self.conv2(self.af(x))
        return x + self.conv_res(feat)


class SA_SKN_Res_encoder(nn.Module):
    def __init__(self, input_size=3, k=[10, 20], pk=16, output_size=64, layers=[2, 2, 2, 2],
                 pts_num=[3072, 1536, 768, 384]):
        super(SA_SKN_Res_encoder, self).__init__()
        self.init_channel = 64

        c1 = self.init_channel
        self.sam_res1 = SKN_Res_unit(input_size, c1, k, int(layers[0]))

        c2 = c1 * 2
        self.sam_res2 = SKN_Res_unit(c2, c2, k, int(layers[1]))

        c3 = c2 * 2
        self.sam_res3 = SKN_Res_unit(c3, c3, k, int(layers[2]))

        c4 = c3 * 2
        self.sam_res4 = SKN_Res_unit(c4, c4, k, int(layers[3]))

        self.conv5 = nn.Conv2d(c4, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1024)

        self.conv6 = nn.Conv2d(c4 + 1024, c4, 1)
        self.conv7 = nn.Conv2d(c3 + c4, c3, 1)
        self.conv8 = nn.Conv2d(c2 + c3, c2, 1)
        self.conv9 = nn.Conv2d(c1 + c2, c1, 1)

        self.conv_out = nn.Conv2d(c1, output_size, 1)
        self.dropout = nn.Dropout()
        self.af = nn.ReLU(inplace=False)
        self.k = k
        self.pk = pk
        self.rate = 2

        self.pts_num = pts_num

    def _make_layer(self, in_planes, rel_planes, mid_planes, out_planes, blocks, share_planes=8, k=16):
        layers = []
        for _ in range(0, blocks):
            layers.append(SK_SA_module(in_planes, rel_planes, mid_planes, out_planes, share_planes, k))
        return nn.Sequential(*layers)

    def _edge_pooling(self, features, points, rate=2, k=16, sample_num=None):
        features = features.squeeze(2)

        if sample_num is None:
            input_points_num = int(features.size()[2])
            sample_num = input_points_num // rate

        ds_features, p_idx, pn_idx, ds_points = edge_preserve_sampling(features, points, sample_num, k)
        ds_features = ds_features.unsqueeze(2)
        return ds_features, p_idx, pn_idx, ds_points

    def _edge_unpooling(self, features, src_pts, tgt_pts):
        features = features.squeeze(2)
        idx, weight = three_nn_upsampling(tgt_pts, src_pts)
        features = three_interpolate(features, idx, weight)
        features = features.unsqueeze(2)
        return features

    def forward(self, features):
        batch_size, _, num_points = features.size()
        pt1 = features[:, 0:3, :]

        idx1 = []
        for i in range(len(self.k)):
            idx = knn(pt1, self.k[i])
            idx1.append(idx)

        pt1 = pt1.transpose(1, 2).contiguous()

        x = features.unsqueeze(2)
        x = self.sam_res1(x, idx1)
        x1 = self.af(x)

        x, _, _, pt2 = self._edge_pooling(x1, pt1, self.rate, self.pk, self.pts_num[1])
        idx2 = []
        for i in range(len(self.k)):
            idx = knn(pt2.transpose(1, 2).contiguous(), self.k[i])
            idx2.append(idx)

        x = self.sam_res2(x, idx2)
        x2 = self.af(x)

        x, _, _, pt3 = self._edge_pooling(x2, pt2, self.rate, self.pk, self.pts_num[2])
        idx3 = []
        for i in range(len(self.k)):
            idx = knn(pt3.transpose(1, 2).contiguous(), self.k[i])
            idx3.append(idx)

        x = self.sam_res3(x, idx3)
        x3 = self.af(x)

        x, _, _, pt4 = self._edge_pooling(x3, pt3, self.rate, self.pk, self.pts_num[3])
        idx4 = []
        for i in range(len(self.k)):
            idx = knn(pt4.transpose(1, 2).contiguous(), self.k[i])
            idx4.append(idx)

        x = self.sam_res4(x, idx4)
        x4 = self.af(x)
        x = self.conv5(x4)
        x, _ = torch.max(x, -1)
        x = x.view(batch_size, -1)
        x = self.dropout(self.af(self.fc2(self.dropout(self.af(self.fc1(x))))))

        x = x.unsqueeze(2).repeat(1, 1, self.pts_num[3]).unsqueeze(2)
        x = self.af(self.conv6(torch.cat([x, x4], 1)))
        x = self._edge_unpooling(x, pt4, pt3)
        x = self.af(self.conv7(torch.cat([x, x3], 1)))
        x = self._edge_unpooling(x, pt3, pt2)
        x = self.af(self.conv8(torch.cat([x, x2], 1)))
        x = self._edge_unpooling(x, pt2, pt1)
        x = self.af(self.conv9(torch.cat([x, x1], 1)))
        x = self.conv_out(x)
        x = x.squeeze(2)
        return x


class MSAP_SKN_decoder(nn.Module):
    def __init__(self, num_coarse_raw, num_fps, num_coarse, num_fine, layers=[2, 2, 2, 2], knn_list=[10, 20], pk=10,
                 points_label=False, local_folding=False):
        super(MSAP_SKN_decoder, self).__init__()
        self.num_coarse_raw = num_coarse_raw
        self.num_fps = num_fps
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.points_label = points_label
        self.local_folding = local_folding

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse_raw * 3)

        self.dense_feature_size = 256
        self.expand_feature_size = 64

        if points_label:
            self.input_size = 6
        else:
            self.input_size = 3

        self.encoder = SA_SKN_Res_encoder(input_size=self.input_size, k=knn_list, pk=pk,
                                          output_size=self.dense_feature_size, layers=layers)

        self.up_scale = int(np.ceil(num_fine / (num_coarse_raw + 2048)))

        if self.up_scale >= 2:
            self.expansion1 = EF_expansion(input_size=self.dense_feature_size, output_size=self.expand_feature_size,
                                           step_ratio=self.up_scale, k=4)
            self.conv_cup1 = nn.Conv1d(self.expand_feature_size, self.expand_feature_size, 1)
        else:
            self.expansion1 = None
            self.conv_cup1 = nn.Conv1d(self.dense_feature_size, self.expand_feature_size, 1)

        self.conv_cup2 = nn.Conv1d(self.expand_feature_size, 3, 1, bias=True)

        self.conv_s1 = nn.Conv1d(self.expand_feature_size, 16, 1, bias=True)
        self.conv_s2 = nn.Conv1d(16, 8, 1, bias=True)
        self.conv_s3 = nn.Conv1d(8, 1, 1, bias=True)

        if self.local_folding:
            self.expansion2 = Folding(input_size=self.expand_feature_size, output_size=self.dense_feature_size,
                                      step_ratio=(num_fine // num_coarse))
        else:
            self.expansion2 = EF_expansion(input_size=self.expand_feature_size, output_size=self.dense_feature_size,
                                           step_ratio=(num_fine // num_coarse), k=4)

        self.conv_f1 = nn.Conv1d(self.dense_feature_size, self.expand_feature_size, 1)
        self.conv_f2 = nn.Conv1d(self.expand_feature_size, 3, 1)

        self.af = nn.ReLU(inplace=False)

    def forward(self, global_feat, point_input, use_Xray_segm, Xray_segm):
        batch_size = global_feat.size()[0]

        coarse_raw = self.fc3(self.af(self.fc2(self.af(self.fc1(global_feat))))).view(batch_size, 3,
                                                                                      self.num_coarse_raw)

        input_points_num = point_input.size()[2]
        org_points_input = point_input
        org_Xray_segm = Xray_segm


        if self.points_label:
            id0_coarse = torch.zeros(coarse_raw.shape[0], 1, coarse_raw.shape[2]).to(device).contiguous()
            id1_coarse = torch.ones(coarse_raw.shape[0], 1, coarse_raw.shape[2]).to(device).contiguous()
            coarse_input = torch.cat((coarse_raw, id1_coarse, id0_coarse, id0_coarse), 1)


            id0_orig = torch.zeros(org_points_input.shape[0], 1, org_points_input.shape[2]).to(device).contiguous()
            id1_orig = torch.ones(org_points_input.shape[0], 1, org_points_input.shape[2]).to(device).contiguous()
            org_points_input = torch.cat((org_points_input, id0_orig, id1_orig, id0_orig), 1)

            if(use_Xray_segm):
                id0_Xraysegm = torch.zeros(org_Xray_segm.shape[0], 1, org_Xray_segm.shape[2]).to(device).contiguous()
                id1_Xraysegm = torch.ones(org_Xray_segm.shape[0], 1, org_Xray_segm.shape[2]).to(device).contiguous()
                org_Xray_segm = torch.cat((org_Xray_segm, id0_Xraysegm, id0_Xraysegm, id1_Xraysegm), 1)

        else:
            coarse_input = coarse_raw

        if use_Xray_segm:
            points = torch.cat((coarse_input, org_points_input, org_Xray_segm), 2)
        else:
            points = torch.cat((coarse_input, org_points_input),2)

        dense_feat = self.encoder(points)

        if self.up_scale >= 2:
            dense_feat = self.expansion1(dense_feat)

        coarse_features = self.af(self.conv_cup1(dense_feat))
        coarse_high = self.conv_cup2(coarse_features)

        if coarse_high.size()[2] > self.num_fps:
            idx_fps = furthest_point_sample(coarse_high.transpose(1, 2).contiguous(), self.num_fps)
            coarse_fps = gather_points(coarse_high, idx_fps)
            coarse_features = gather_points(coarse_features, idx_fps)
        else:
            coarse_fps = coarse_high

        if coarse_fps.size()[2] > self.num_coarse:
            scores = F.softplus(self.conv_s3(self.af(self.conv_s2(self.af(self.conv_s1(coarse_features))))))
            idx_scores = scores.topk(k=self.num_coarse, dim=2)[1].view(batch_size, -1).int()
            coarse = gather_points(coarse_fps, idx_scores)
            coarse_features = gather_points(coarse_features, idx_scores)
        else:
            coarse = coarse_fps

        if coarse.size()[2] < self.num_fine:
            if self.local_folding:
                up_features = self.expansion2(coarse_features, global_feat)
                center = coarse.transpose(2, 1).contiguous().unsqueeze(2).repeat(1, 1, self.num_fine // self.num_coarse,
                                                                                 1).view(batch_size, self.num_fine,
                                                                                         3).transpose(2, 1).contiguous()
                fine = self.conv_f2(self.af(self.conv_f1(up_features))) + center
            else:
                up_features = self.expansion2(coarse_features)
                fine = self.conv_f2(self.af(self.conv_f1(up_features)))
        else:
            assert (coarse.size()[2] == self.num_fine)
            fine = coarse

        return coarse_raw, coarse_high, coarse, fine

class MLP_feature_selection(nn.Module):
    def __init__(self,global_feature_size=2048):
        super(MLP_feature_selection, self).__init__()
        self.fc1 = nn.Linear(global_feature_size , global_feature_size)  # Adjust the input size based on your label map dimensions
        self.fc2 = nn.Linear(global_feature_size, global_feature_size//2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class Model(nn.Module):
    def __init__(self, args, size_z=128, global_feature_size=1024):
        super(Model, self).__init__()

        layers = [int(i) for i in args.layers.split(',')]
        knn_list = [int(i) for i in args.knn_list.split(',')]

        self.num_points = args.num_points
        self.size_z = size_z
        self.distribution_loss = args.distribution_loss
        self.train_loss = args.loss
        self.eval_emd = args.eval_emd
        self.align = args.align
        self.encoder_pcd = PCN_encoder(output_size=global_feature_size)
        self.encoder_labelmap = PCN_encoder(output_size=global_feature_size)

        self.posterior_infer1 = Linear_ResBlock(input_size=global_feature_size, output_size=global_feature_size)
        self.posterior_infer2 = Linear_ResBlock(input_size=global_feature_size, output_size=size_z * 2)
        self.prior_infer = Linear_ResBlock(input_size=global_feature_size, output_size=size_z * 2)
        self.generator = Linear_ResBlock(input_size=size_z, output_size=global_feature_size)
        self.decoder = MSAP_SKN_decoder(num_fps=args.num_fps, num_fine=args.num_points, num_coarse=args.num_coarse,
                                        num_coarse_raw=args.num_coarse_raw, layers=layers, knn_list=knn_list,
                                        pk=args.pk, local_folding=args.local_folding, points_label=args.points_label)
        self.feature_selector = MLP_feature_selection(global_feature_size*2)
        self.use_labelmaps_in_PMNET = args.use_labelmaps_in_PMNET
        self.use_labelmaps_in_RENet = args.use_labelmaps_in_RENet


    def compute_kernel(self, x, y):
        x_size = x.size()[0]
        y_size = y.size()[0]
        dim = x.size()[1]

        tiled_x = x.unsqueeze(1).repeat(1, y_size, 1)
        tiled_y = y.unsqueeze(0).repeat(x_size, 1, 1)
        return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / float(dim))

    def mmd_loss(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def forward(self, x_pcd, x_labelmap=None, gt=None, prefix="train", mean_feature=None, alpha=None):

        if prefix == "train":
            y = gather_points(gt.transpose(1, 2).contiguous(), furthest_point_sample(gt, self.num_points))

            gt = torch.cat([gt, gt], dim=0)
            points = torch.cat([x_pcd, y], dim=0)
            x_pcd = torch.cat([x_pcd, x_pcd], dim=0)
            x_labelmap_for_RENet = torch.cat([x_labelmap, x_labelmap], dim=0)

        else:
            points = x_pcd
            x_labelmap_for_RENet = x_labelmap

        feat_pcd = self.encoder_pcd(points)
        if(self.use_labelmaps_in_PMNET):
            feat_x_labelmap = self.encoder_labelmap(x_labelmap)

        if prefix == "train":
            # for training we have both a reconstruction path and a completion path
            feat_x_pcd, feat_y = feat_pcd.chunk(2)

            ############ rough completion path ############
            if (self.use_labelmaps_in_PMNET):
                # concatenate the feature vectors from the partial pcd and from the labelmaps
                feat_concat = torch.cat([feat_x_pcd,feat_x_labelmap],dim=1)

                # pass through one layer to get 1024 features again
                feat_combined = self.feature_selector(feat_concat)
                o_x = self.posterior_infer2(self.posterior_infer1(feat_combined))

            # feat_x_pcd comes = partial point cloud encoding, q is the distribution from partial
            else:
                o_x = self.posterior_infer2(self.posterior_infer1(feat_x_pcd))

            q_mu, q_std = torch.split(o_x, self.size_z, dim=1)

            q_std = F.softplus(q_std)
            q_distribution = torch.distributions.Normal(q_mu, q_std)

            z_q = q_distribution.rsample()
            ############ rough completion path ############

            ############ reconstruction path ############
            o_y = self.prior_infer(feat_y)
            p_mu, p_std = torch.split(o_y, self.size_z, dim=1)
            p_std = F.softplus(p_std)

            p_distribution = torch.distributions.Normal(p_mu, p_std)
            p_distribution_fix = torch.distributions.Normal(p_mu.detach(), p_std.detach())
            m_distribution = torch.distributions.Normal(torch.zeros_like(p_mu), torch.ones_like(p_std))
            z_p = p_distribution.rsample()
            ############ reconstruction path ############

            z = torch.cat([z_q, z_p], dim=0)
            feat_pcd = torch.cat([feat_x_pcd, feat_x_pcd], dim=0)


        else:
            # concatenate the features from the partial pcd with the ones from the labelmap
            if(self.use_labelmaps_in_PMNET):

                feat_concat = torch.cat([feat_pcd, feat_x_labelmap], dim=1)
                # pass through one layer to get 1024 features again
                feat_combined = self.feature_selector(feat_concat)

                o_x = self.posterior_infer2(self.posterior_infer1(feat_combined))
            else:
                o_x = self.posterior_infer2(self.posterior_infer1(feat_pcd))

            q_mu, q_std = torch.split(o_x, self.size_z, dim=1)
            q_std = F.softplus(q_std)
            q_distribution = torch.distributions.Normal(q_mu, q_std)
            p_distribution = q_distribution
            p_distribution_fix = p_distribution
            m_distribution = p_distribution
            z = q_distribution.rsample()

        feat_pcd += self.generator(z)

        coarse_raw, coarse_high, coarse, fine = self.decoder(feat_pcd, x_pcd, self.use_labelmaps_in_RENet, x_labelmap_for_RENet)
        coarse_raw = coarse_raw.transpose(1, 2).contiguous()
        coarse_high = coarse_high.transpose(1, 2).contiguous()
        coarse = coarse.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()

        if prefix == "train":
            if self.distribution_loss == 'MMD':
                z_m = m_distribution.rsample()
                z_q = q_distribution.rsample()
                z_p = p_distribution.rsample()
                z_p_fix = p_distribution_fix.rsample()
                dl_rec = self.mmd_loss(z_m, z_p)
                dl_g = self.mmd_loss2(z_q, z_p_fix)
            elif self.distribution_loss == 'KLD':
                dl_rec = torch.distributions.kl_divergence(m_distribution, p_distribution)
                dl_g = torch.distributions.kl_divergence(p_distribution_fix, q_distribution)
            else:
                raise NotImplementedError('Distribution loss is either MMD or KLD')

            if self.train_loss == 'cd':
                loss1, _ = calc_cd(coarse_raw, gt)
                loss2, _ = calc_cd(coarse_high, gt)
                loss3, _ = calc_cd(coarse, gt)
                loss4, _ = calc_cd(fine, gt)

            total_train_loss = loss1.mean() * 10 + loss2.mean() * 0.5 + loss3.mean() + loss4.mean() * alpha

            total_train_loss += (dl_rec.mean() + dl_g.mean()) * 20

            return fine, loss4, total_train_loss
        elif prefix == "val" or prefix == "test":
            fine_cpu = fine.cpu().numpy()
            gt_cpu = gt.cpu().numpy()
            if self.align:
                gts_aligned = []
                inputs_aligned = []
                # iterate over all point clouds in batch size
                input_cpu = x_pcd.transpose(1, 2).contiguous().cpu().numpy()
                for pcd_idx in range(fine_cpu.shape[0]):
                    # create point cloud from completion
                    completion_pcd = o3d.geometry.PointCloud()
                    completion_pcd.points = o3d.utility.Vector3dVector(fine_cpu[pcd_idx])

                    # create point cloud from GT
                    GT_pcd = o3d.geometry.PointCloud()
                    GT_pcd.points = o3d.utility.Vector3dVector(gt_cpu[pcd_idx])

                    # perform the registration
                    reg_p2p = o3d.pipelines.registration.registration_icp(GT_pcd, completion_pcd, 0.02)

                    # apply trafo on the GT pcd
                    GT_pcd.transform(reg_p2p.transformation)
                    gts_aligned.append(np.asarray(GT_pcd.points))

                    # create pcd from input
                    input_pcd = o3d.geometry.PointCloud()
                    input_pcd.points = o3d.utility.Vector3dVector(input_cpu[pcd_idx])

                    # apply trafo on the input pcd
                    input_pcd.transform(reg_p2p.transformation)
                    inputs_aligned.append(np.asarray(input_pcd.points))

                # stack them back
                gt = np.stack(gts_aligned)
                gt = torch.tensor(gt).float().to(device)

            cds_p_arch = []
            cds_t_arch = []
            f1s_arch = []
            emds_arch = []

            cds_p_body = []
            cds_t_body = []
            f1s_body = []
            emds_body = []

            # fine_cpu.shape[0] --> number of shapes in the current batch
            nr_shapes = fine_cpu.shape[0]

            # compute the center of mass of the GT shapes
            centers_of_mass = gt.mean(dim=1)

            # get the vert arch of the GT
            # Get the second coordinate (y-coordinate) of the centers of mass
            centers_of_mass_y = centers_of_mass[:, 1].unsqueeze(1)

            # Compare each point's y-coordinate with the center of mass y-coordinate
            # shapes[:, :, 1] gives all y-coordinates of all points
            # The comparison results in a boolean mask
            mask_gt_arch = gt[:, :, 1] > centers_of_mass_y
            mask_completed_arch = fine[:, :, 1] > centers_of_mass_y

            # For the vertebral body
            mask_gt_body = gt[:, :, 1] <= centers_of_mass_y
            mask_completed_body = fine[:, :, 1] <= centers_of_mass_y

            for pcd_idx in range(nr_shapes):
                gt_arch = gt[pcd_idx][mask_gt_arch[pcd_idx]]
                completion_arch = fine[pcd_idx][mask_completed_arch[pcd_idx]]

                gt_body = gt[pcd_idx][mask_gt_body[pcd_idx]]
                completion_body = fine[pcd_idx][mask_completed_body[pcd_idx]]

                """temp for testing"""

                def save_point_cloud(points, filename):
                    points_cpu = points.cpu().numpy()
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points_cpu)
                    o3d.io.write_point_cloud(filename, pcd)
                    print(f"Point cloud saved to {filename}")
                
                
                #save_point_cloud(gt_arch, f"gt_arch_{pcd_idx}.pcd")
                #save_point_cloud(completion_arch, f"completion_arch_{pcd_idx}.pcd")
                #save_point_cloud(gt_body, f"gt_body_{pcd_idx}.pcd")
                #save_point_cloud(completion_body, f"completion_body_{pcd_idx}.pcd")


                """temp for testing"""

                # add batch size dimension
                completion_arch = completion_arch[np.newaxis, :, :]
                gt_arch = gt_arch[np.newaxis, :, :]


                completion_body = completion_body[np.newaxis, :, :]
                gt_body = gt_body[np.newaxis, :, :]

                # compute here the metrics only for one shape
                cd_p_arch, cd_t_arch, f1_arch = calc_cd(completion_arch, gt_arch, calc_f1=True)
                cds_p_arch.append(cd_p_arch)
                cds_t_arch.append(cd_t_arch)
                f1s_arch.append(f1_arch)

                cd_p_body, cd_t_body, f1_body = calc_cd(completion_body, gt_body, calc_f1=True)
                cds_p_body.append(cd_p_body)
                cds_t_body.append(cd_t_body)
                f1s_body.append(f1_body)

                if self.eval_emd:
                    # we can only compare point clouds with the same number of points through EMD

                    # we can only compare point clouds with the same number of points through EMD
                    if completion_arch.shape[1] != 1024 or gt_arch.shape[1] != 1024:
                        # shape of the point clouds need to be B x N x 3
                        gt_arch = gather_points(gt_arch.transpose(1, 2).contiguous(), furthest_point_sample(gt_arch, 1024)).transpose(1, 2)
                        completion_arch = gather_points(completion_arch.transpose(1, 2).contiguous(), furthest_point_sample(completion_arch, 1024)).transpose(1, 2)

                    emd_arch = calc_emd(completion_arch, gt_arch, eps=0.004, iterations=3000)
                    emds_arch.append(emd_arch)

                    if completion_body.shape[1] != 1024 or gt_body.shape[1] != 1024:
                        gt_body = gather_points(gt_body.transpose(1, 2).contiguous(), furthest_point_sample(gt_body, 1024)).transpose(1, 2)
                        completion_body = gather_points(completion_body.transpose(1, 2).contiguous(), furthest_point_sample(completion_body, 1024)).transpose(1, 2)

                    emd_body = calc_emd(completion_body,gt_body, eps=0.004, iterations=3000)
                    emds_body.append(emd_body)


            cd_p_arch = torch.stack(cds_p_arch)
            cd_p_arch = torch.reshape(cd_p_arch, (fine_cpu.shape[0],))
            cd_t_arch = torch.stack(cds_t_arch)
            cd_t_arch = torch.reshape(cd_t_arch, (fine_cpu.shape[0],))
            f1_arch = torch.stack(f1s_arch)
            f1_arch = torch.reshape(f1_arch, (fine_cpu.shape[0],))

            cd_p_body = torch.stack(cds_p_body)
            cd_p_body = torch.reshape(cd_p_body, (fine_cpu.shape[0],))
            cd_t_body = torch.stack(cds_t_body)
            cd_t_body = torch.reshape(cd_t_body, (fine_cpu.shape[0],))
            f1_body = torch.stack(f1s_body)
            f1_body = torch.reshape(f1_body, (fine_cpu.shape[0],))

            if self.eval_emd:
                emd_arch = torch.stack(emds_arch)
                emd_arch = torch.reshape(emd_arch, (fine_cpu.shape[0],))

                emd_body = torch.stack(emds_body)
                emd_body = torch.reshape(emd_body, (fine_cpu.shape[0],))
            else:
                emd_arch = 0
                emd_body = 0



            if self.eval_emd:
                emd = calc_emd(fine, gt, eps=0.004, iterations=3000)
            else:
                emd = 0

            # compute the metrics for the whole vertebral shape
            cd_p, cd_t, f1 = calc_cd(fine, gt, calc_f1=True)

            return {
                'out1': coarse_raw,
                'result': fine,
                'gt': gt,
                'inputs': x_pcd,
                'emd': emd,
                'cd_p': cd_p,
                'cd_t': cd_t,
                'f1': f1,
                'emd_arch': emd_arch,
                'cd_p_arch': cd_p_arch,
                'cd_t_arch': cd_t_arch,
                'f1_arch': f1_arch,
                'emd_body': emd_body,
                'cd_p_body': cd_p_body,
                'cd_t_body': cd_t_body,
                'f1_body': f1_body
            }


