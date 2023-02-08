from typing import Optional
import torch
from torch.nn import functional as F
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations

from hpcs.models.base_hyp_hc import BaseSimilarityHypHC
from hpcs.utils.data import to_categorical


class ShapeNetHypHC(BaseSimilarityHypHC):
    def __init__(self, nn_feat: torch.nn.Module,
                 nn_emb: Optional[torch.nn.Module],
                 lr: float = 1e-3,
                 embedding: int = 6,
                 margin: float = 0.5,
                 t_per_anchor: int = 50,
                 fraction: float = 1.2,
                 temperature: float = 0.05,
                 anneal_factor: float = 0.5,
                 anneal_step: int = 0,
                 num_class: int = 4,
                 trade_off: float = 0.1,
                 plot_inference: bool = False,
                 use_hc_loss: bool = True,
                 radius: float = 1.0,
                 train_rotation: str = 'so3',
                 test_rotation: str = 'so3',
                 class_vector: bool = False):
        super(ShapeNetHypHC, self).__init__(nn_feat=nn_feat,
                                            nn_emb=nn_emb,
                                            lr=lr,
                                            embedding=embedding,
                                            margin=margin,
                                            t_per_anchor=t_per_anchor,
                                            fraction=fraction,
                                            temperature=temperature,
                                            anneal_factor=anneal_factor,
                                            anneal_step=anneal_step,
                                            num_class=num_class,
                                            trade_off=trade_off,
                                            plot_inference=plot_inference,
                                            use_hc_loss=use_hc_loss,
                                            radius=radius)

        self.train_rotation = train_rotation
        self.test_rotation = test_rotation
        self.class_vector = class_vector

    def _forward(self, batch, testing):
        points, label, targets = batch

        if testing:
            rot = self.test_rotation
        else:
            rot = self.train_rotation

        trot = None
        if rot == 'z':
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True)
        elif rot == 'so3':
            trot = Rotate(R=random_rotations(points.shape[0]))
        if trot is not None:
            points = trot.transform_points(points.cpu())

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        points, label, targets = points.float().to(device), label.long().to(device), targets
        points = points.transpose(2, 1)

        if self.class_vector:
            num_parts = self.num_class
            batch_class_vector = []
            for object in targets:
                parts = F.one_hot(torch.unique(object), num_parts)
                class_vector = parts.sum(dim=0).float()
                batch_class_vector.append(class_vector)
            decode_vector = torch.stack(batch_class_vector)
        else:
            num_categories = self.num_class
            decode_vector = to_categorical(label, num_categories)

        x_euclidean = self.nn_feat(points, decode_vector)
        if self.nn_emb:
            x_poincare = self.nn_emb(x_euclidean)
            x_poincare = x_poincare.contiguous().view(-1, self.embedding)
        else:
            x_poincare = None

        return points, x_euclidean, x_poincare, targets