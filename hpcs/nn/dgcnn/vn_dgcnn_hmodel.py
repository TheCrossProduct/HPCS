import torch.nn as nn
import torch.utils.data
from hpcs.nn.dgcnn.utils.vn_layers import *
from hpcs.nn.dgcnn.utils.vn_dgcnn_util import get_graph_feature
from hpcs.nn.dgcnn.utils.poincareball import PoincareBall
from hpcs.nn.dgcnn.utils.manifold_layers import MobiusLayer


class VN_DGCNN_hmodel(nn.Module):
    def __init__(self, in_channels, out_features, k, dropout, pooling, num_class):
        super(VN_DGCNN_hmodel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.k = k
        self.dropout = dropout
        self.pooling = pooling
        self.num_class = num_class

        self.manifold = PoincareBall(c=1, dim=2299)
        self.manifold1 = PoincareBall(c=1, dim=1024)
        self.manifold2 = PoincareBall(c=1, dim=512)
        self.manifold3 = PoincareBall(c=1, dim=256)

        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        if self.pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
        elif self.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, 1024 // 3, dim=4, share_nonlinearity=True)
        self.std_feature = VNStdFeature(1024 // 3 * 2, dim=4, normalize_frame=False)
        self.conv7 = nn.Sequential(nn.Conv1d(self.num_class, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.emb = MobiusLayer(2299, 1024, self.manifold)
        self.emb1 = MobiusLayer(1024, 512, self.manifold1)
        self.emb2 = MobiusLayer(512, 256, self.manifold2)
        self.mlr = MobiusLayer(256, self.out_features, self.manifold3)


    def forward(self, x, l):
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
        x = x.max(dim=-1, keepdim=True)[0]

        l = l.view(batch_size, -1, 1)
        l = self.conv7(l)

        x = torch.cat((x, l), dim=1)
        x = x.repeat(1, 1, num_points)

        x = torch.cat((x, x123), dim=1)
        x = x.transpose(1, 2)

        x = self.manifold.expmap0(x)
        x = self.emb(x)
        x = self.emb1(x)
        x = self.emb2(x)
        x = self.mlr(x)

        return x