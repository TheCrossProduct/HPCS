from typing import Union, List


def get_sampler_and_backend(gpu: Union[List[Union[str, int]], str], distr: bool):
    # training with multiple gpu-s
    if isinstance(gpu, list) and len(gpu) > 1:
        # check if training on a cluster or not
        distributed_backend = 'ddp' if distr else 'dp'
        replace_sampler_ddp = False if distr else True
    else:
        distributed_backend = None
        replace_sampler_ddp = True

    return replace_sampler_ddp, distributed_backend
