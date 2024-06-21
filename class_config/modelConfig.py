from omegaconf import DictConfig, OmegaConf


class ModelConfig:
    def __init__(self, cfg: DictConfig):
        self.T = OmegaConf.select(cfg, "model.T", default=1)


class GPT2Config:
    EMBED_DIM = 512
    N_HEAD = 8
    DROPOUT = 0.1
    N_BLOCK_GPT = 3
    BATCH_FIRST = True
    BATCH_SIZE = 64
    MAX_GEN_LEN = 128
    learning_rate = 0.1
