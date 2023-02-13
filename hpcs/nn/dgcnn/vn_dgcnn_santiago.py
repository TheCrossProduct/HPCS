import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from hpcs.nn.dgcnn.utils.vn_layers import *
from hpcs.nn.dgcnn.utils.vn_dgcnn_util import get_graph_feature

class ConstrainedConv1d(nn.Conv1d):
    def forward(self, input):
        out = nn.functional.conv1d(input, torch.nn.functional.normalize(self.weight), self.bias, self.stride,self.padding, self.dilation, self.groups)
        return out

class VN_DGCNN_santiago(nn.Module):
    def __init__(self, in_channels, out_features, k, dropout, num_class,normalize_classification=True):
        super(VN_DGCNN_santiago, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.k = k
        self.dropout = dropout
        self.num_class = num_class
        self.pooling='max'
        self.normalize_classification=normalize_classification

        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)
        self.bn11 = nn.BatchNorm1d(self.out_features)
        if self.pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)

        elif self.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool

        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, 1024 // 3, dim=4, share_nonlinearity=True)
        self.std_feature = VNStdFeature(1024 // 3 * 2, dim=4, normalize_frame=False)


        self.conv8 = nn.Sequential(nn.Conv1d(2046, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=self.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=self.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Sequential(nn.Conv1d(128, self.out_features, kernel_size=1, bias=False),
                                   self.bn11)
        if normalize_classification:
            self.classification = ConstrainedConv1d(self.out_features, self.num_class, kernel_size=1, bias=False)
            #Force to have norm 1 the initialited weight
            with torch.no_grad():
                self.classification.weight= torch.nn.Parameter(torch.nn.functional.normalize(self.classification.weight,eps=1e-5))
        else:
            self.classification = nn.Conv1d(self.out_features, self.num_class, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x3 = self.pool3(x)

        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, z0 = self.std_feature(x)
        x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(batch_size, -1, num_points)
        x = x.view(batch_size, -1, num_points)

        #MLP
        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        x = self.dp2(x)
        x = self.conv10(x)
        features = self.conv11(x)
        return self.classification(features),features
