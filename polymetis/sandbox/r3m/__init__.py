import omegaconf
import hydra
import os
import torch
import copy
from r3m.models import R3M


VALID_ARGS = ["_target_", "device", "lr", "hidden_dim", "size", "l2weight", "l1weight", "langweight", "tcnweight", "l2dist", "bs"]
device = "cuda" if torch.cuda.is_available() else "cpu"


def remove_language_head(state_dict):
    keys = state_dict.keys()
    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    for key in list(keys):
        if ("lang_enc" in key) or ("lang_rew" in key):
            del state_dict[key]
    return state_dict


def cleanup_config(cfg):
    config = copy.deepcopy(cfg)
    keys = config.agent.keys()
    for key in list(keys):
        if key not in VALID_ARGS:
            del config.agent[key]
    config.agent["_target_"] = "r3m.R3M"
    config["device"] = device
    
    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    config.agent["langweight"] = 0
    return config.agent


def load_r3m(modelid):
    assert modelid in ["resnet50"]
    configpath = os.path.join(os.path.dirname(__file__), "checkpoints", f"r3m_{modelid}_config.yml")
    modelpath = os.path.join(os.path.dirname(__file__), "checkpoints", f"r3m_{modelid}_model.pt")
    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    rep = torch.nn.DataParallel(rep)
    r3m_state_dict = remove_language_head(torch.load(modelpath, map_location=torch.device(device))['r3m'])
    rep.load_state_dict(r3m_state_dict)
    return rep
