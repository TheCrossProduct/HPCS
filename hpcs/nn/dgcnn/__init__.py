__all__ = [
    'DGCNN_simple',
    'DGCNN_partseg',
    'VN_DGCNN_partseg',
    'vn_dgcnn_partseg_encoder',
]

from .dgcnn import DGCNN_simple
from .dgcnn_partseg import DGCNN_partseg
from .vn_dgcnn_partseg import VN_DGCNN_partseg
from .vn_dgcnn_partseg_encoder import VN_DGCNN_partseg_encoder
