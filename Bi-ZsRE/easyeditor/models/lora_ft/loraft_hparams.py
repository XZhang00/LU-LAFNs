from dataclasses import dataclass
from typing import List
import yaml

from ...util.hparams import HyperParams


@dataclass
class LORAFTHyperParams(HyperParams):
    # Method
    alg_name: str
    device: int
    model_name: str
    lora_model_name: str
    max_length: int = 128
    eval_bsz: int = 4

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        return cls(**config)
