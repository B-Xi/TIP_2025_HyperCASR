import json
import torch
import random
import os
import numpy as np
import pytorch_lightning as pl
from .pyExt import Dict2Obj


def getDatasetInfo(dataset):
    PATH = "/mnt/HDD/data/zwj/model_1/my_model/datasets/" + dataset + "/config.json"
    with open(PATH, "r") as f:
        info = json.load(f)

    return info


def seed_torch(seed):
    pl.seed_everything(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
