import torch.nn as nn
import torch.utils.data
from hpcs.nn.dgcnn.utils.vn_layers import *
from hpcs.nn.dgcnn.utils.vn_dgcnn_util import get_graph_feature


class VN_DGCNN_partseg_class(nn.Module):
    def __init__(self, in_channels, out_features, k, dropout, pooling):
        super(VN_DGCNN_partseg_class, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.k = k
        self.dropout = dropout
        self.pooling = pooling

        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = VNLinearLeakyReLU(2, 128 // 3)
        self.conv2 = VNLinearLeakyReLU(128 // 3, 128 // 3)
        self.conv3 = VNLinearLeakyReLU(128 // 3, 128 // 3)
        self.conv4 = VNLinearLeakyReLU(128 // 3 * 2, 128 // 3)
        self.conv5 = VNLinearLeakyReLU(128 // 3, 128 // 3)
        self.conv6 = VNLinearLeakyReLU(128 // 3, 128 // 3)
        self.conv7 = VNLinearLeakyReLU(128 // 3 * 2, 128 // 3)
        self.conv8 = VNLinearLeakyReLU(128 // 3, 128 // 3)

        self.conv9 = VNLinearLeakyReLU(64 // 3 * 2, 1024 // 3, dim=4, share_nonlinearity=True)
        self.std_feature = VNStdFeature(1024 // 3 * 2, dim=4, normalize_frame=False)

        self.conv10 = nn.Sequential(nn.Conv1d(2046, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.dp1 = nn.Dropout(p=0.5)
        self.conv11 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv12 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv13 = nn.Conv1d(128, self.out_features, kernel_size=1, bias=False)

        if self.pooling == 'max':
            self.pool1 = VNMaxPool(128 // 3)
            self.pool2 = VNMaxPool(128 // 3)
            self.pool3 = VNMaxPool(128 // 3)
        elif self.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv7(x)
        x = self.conv8(x)
        x3 = self.pool3(x)

        x = self.conv9(x3)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, z0 = self.std_feature(x)
        # x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(batch_size, -1, num_points)
        x = x.view(batch_size, -1, num_points)

        x = self.conv10(x)
        x = self.dp1(x)
        x = self.conv11(x)
        x = self.dp2(x)
        x = self.conv12(x)
        x = self.conv13(x)

        return x.transpose(1, 2)