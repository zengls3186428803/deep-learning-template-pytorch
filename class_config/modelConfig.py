from omegaconf import DictConfig, OmegaConf


class ModelConfig:
    def __init__(self, cfg: DictConfig = None):
        self.T = OmegaConf.select(cfg, "model.T", default=1)


class GPT2Config:
    def __init__(self, cfg: DictConfig):
        self.embed_dim = OmegaConf.select(cfg, "model.embed_dim", default=10)
        self.n_head = OmegaConf.select(cfg, "model.n_head", default=1)
        self.dropout = OmegaConf.select(cfg, "model.dropout", default=0.1)
        self.n_block_gpt = OmegaConf.select(cfg, "model.n_block_gpt", default=1)
        self.batch_first = OmegaConf.select(cfg, "model.batch_first", default=True)
        self.batch_size = OmegaConf.select(cfg, "model.batch_size", default=64)
        self.max_gen_len = OmegaConf.select(cfg, "model.max_gen_len", default=128)
        self.max_pos = OmegaConf.select(cfg, "model.max_pos", default=5000)