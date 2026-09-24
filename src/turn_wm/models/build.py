from hydra.utils import instantiate


def build_model(cfg):
    # The encoder is nested in the model config and built recursively.
    return instantiate(cfg.model)
