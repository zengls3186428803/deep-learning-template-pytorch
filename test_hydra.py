from omegaconf import OmegaConf, DictConfig
import argparse
import hydra
import os

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig = None):
    print(cfg)
    parser = argparse.ArgumentParser(prog="test_my_parser", description="test argparse.")
    parser.add_argument("-a", default="23", type=int)
    agrs = parser.parse_args()
    print(agrs)
    print(agrs.a)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="test_my_parser", description="test argparse.")
    parser.add_argument("-a", default="23", type=int)
    agrs = parser.parse_args()
    print(agrs)
    print(agrs.a)
    pass
    main()
