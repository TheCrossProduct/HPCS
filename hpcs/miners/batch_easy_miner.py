from .batch_easy_hard_miner import BatchEasyHardMiner


class BatchEasyMiner(BatchEasyHardMiner):
    def __init__(self, **kwargs):
        super().__init__(
            pos_strategy=BatchEasyHardMiner.EASY,
            neg_strategy=BatchEasyHardMiner.EASY,
            **kwargs
        )

    def mine(self, *args, **kwargs):
        a1, p, a2, n = super().mine(*args, **kwargs)
        return a1, p, n