import pytorch_lightning as pl

from hpcs.models import ShapeNetHypHC, PartNetHypHC

from hpcs.nn.dgcnn import DGCNN_partseg, VN_DGCNN_partseg

from hpcs.nn.pointnet import POINTNET_partseg, VN_POINTNET_partseg
from hpcs.nn.hyperbolic import ExpMap, MLPExpMap


def configure_feature_extractor(model_name, num_class, out_features, num_categories, k, dropout, pretrained):
    if model_name == 'dgcnn_partseg':
        nn = DGCNN_partseg(in_channels=3, out_features=num_class, k=k, dropout=dropout, num_categories=num_categories)
    elif model_name == 'vn_dgcnn_partseg':
        nn = VN_DGCNN_partseg(in_channels=3, out_features=out_features, k=k, dropout=dropout, pooling='mean', num_categories=num_categories)
    elif model_name == 'pointnet_partseg':
        nn = POINTNET_partseg(num_part=num_class, normal_channel=False)
    elif model_name == 'vn_pointnet_partseg':
        nn = VN_POINTNET_partseg(num_part=num_class, normal_channel=True, k=k, pooling='mean')
    else:
        raise ValueError(f"Not implemented for model_name {model_name}")
    return nn

def configure_hyperbolic_embedder(input_features: int, output_features: int):
    if input_features == output_features:
        return ExpMap()
    else:
        return MLPExpMap(input_feat=input_features, out_feat=output_features)


def configure(config):
    dataset = config['dataset']['value']
    model_name = config['model']['value']
    train_rotation = config['train_rotation']['value']
    test_rotation = config['test_rotation']['value']
    eucl_embedding = config['eucl_embedding']['value']
    hyp_embedding = config['hyp_embedding']['value']
    k = config['k']['value']
    margin = config['margin']['value']
    t_per_anchor = config['t_per_anchor']['value']
    temperature = config['temperature']['value']
    lr = config['lr']['value']
    accelerator = 'cpu'
    anneal_factor = config['anneal_factor']['value']
    anneal_step = config['anneal_step']['value']
    num_class = config['num_class']['value']
    pretrained = config['pretrained']['value']
    dropout = config['dropout']['value']
    fraction = config['fraction']['value']
    class_vector = config['class_vector']['value']
    trade_off = config['trade_off']['value']
    miner = config['miner']['value']
    cosface = config['cosface']['value']
    plot_inference = config['plot_inference']['value']
    hierarchical = config['hierarchical']['value']
    hierarchy_list = config['hierarchy_list']['value']


    if dataset == 'shapenet':
        num_categories = 16
    else:
        num_categories = 1

    nn_feat = configure_feature_extractor(model_name=model_name,
                                          num_class=num_class,
                                          out_features=eucl_embedding,
                                          num_categories=num_categories,
                                          k=k,
                                          dropout=dropout,
                                          pretrained=pretrained)

    nn_emb = configure_hyperbolic_embedder(input_features=eucl_embedding, output_features=hyp_embedding)

    if dataset == 'shapenet':
        model = ShapeNetHypHC(nn_feat=nn_feat,
                              nn_emb=nn_emb,
                              euclidean_size=eucl_embedding,
                              hyp_size=hyp_embedding,
                              train_rotation=train_rotation,
                              test_rotation=test_rotation,
                              lr=lr,
                              margin=margin,
                              t_per_anchor=t_per_anchor,
                              fraction=fraction,
                              temperature=temperature,
                              anneal_factor=anneal_factor,
                              anneal_step=anneal_step,
                              num_class=num_class,
                              class_vector=class_vector,
                              trade_off=trade_off,
                              miner=miner,
                              cosface=cosface,
                              plot_inference=plot_inference)
    elif dataset == 'partnet':
        model = PartNetHypHC(nn_feat=nn_feat,
                             nn_emb=nn_emb,
                             euclidean_size=eucl_embedding,
                             hyp_size=hyp_embedding,
                             train_rotation=train_rotation,
                             test_rotation=test_rotation,
                             lr=lr,
                             margin=margin,
                             t_per_anchor=t_per_anchor,
                             fraction=fraction,
                             temperature=temperature,
                             anneal_factor=anneal_factor,
                             anneal_step=anneal_step,
                             num_class=num_class,
                             class_vector=class_vector,
                             trade_off=trade_off,
                             miner=miner,
                             cosface=cosface,
                             hierarchical=hierarchical,
                             hierarchy_list=hierarchy_list,
                             plot_inference=plot_inference)
    else:
        raise KeyError(f"Not available implementation for dataset: {dataset}")

    trainer = pl.Trainer(accelerator=accelerator,
                         max_epochs=-1,
                         limit_test_batches=10
                         )

    return model, trainer