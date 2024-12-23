from ..models.rome import ROMEHyperParams, apply_rome_to_model, apply_mrome_to_model
from ..models.memit import MEMITHyperParams, apply_memit_to_model
from ..models.kn import KNHyperParams, apply_kn_to_model
from ..models.ft import FTHyperParams, apply_ft_to_model
from ..models.serac import SERACHparams, SeracRewriteExecutor
from ..dataset import ZsreDataset, CounterFactDataset
# from ..models.ike import IKEHyperParams, apply_ike_to_model
from ..models.key import (KEYHyperParams, apply_key_to_model, 
                apply_multikey_to_model,
                apply_multikey_to_model_m,
                apply_multikey_to_model_batch,
                apply_multikey_to_model_v2,
                apply_multikey_to_model_v3,
                apply_multikey_to_model_v4,
                apply_multikey_to_model_v5,
                apply_multikey_to_model_v6,
                apply_multivalue_to_model)
from ..models.pmet import PMETHyperParams, apply_pmet_to_model
from ..models.lora_ft import LORAFTHyperParams
from ..models.remake import REMAKEHyperParams


ALG_DICT = {
    'ROME': apply_rome_to_model,
    'MROME': apply_mrome_to_model,
    'MEMIT': apply_memit_to_model,
    'MMEMIT': apply_memit_to_model,
    'PMET': apply_pmet_to_model,
    'MPMET': apply_pmet_to_model,
    "FT": apply_ft_to_model,
    'KN': apply_kn_to_model,
    'SERAC': SeracRewriteExecutor().apply_to_model,
    'KEY': apply_key_to_model,
    'MKEY': apply_multikey_to_model_batch,
    'MKEYPG': apply_multikey_to_model_v4,
    'MVALUE': apply_multivalue_to_model,
    'LORAFT': None,
    'REMAKE': None,
}

DS_DICT = {
    "cf": CounterFactDataset,
    "zsre": ZsreDataset
}