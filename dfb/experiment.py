from random import sample
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader

from dfb.download import *
from dfb.databuilder import *
from dfb.dataset import *
from dfb.model.aannet import AAnNet
from dfb.model.dcn import DCN
from dfb.model.srdcnn import SRDCNN
from dfb.model.wdcnnrnn import WDCNNRNN
from dfb.model.lnrnet import LNRNet
from dfb.model.catlnrnet import *
from dfb.trainmodule import *
from dfb.processing import *
from dfb.paramsampler import *

from dfb.model.ticnn import *
from dfb.model.wdcnn import *
from dfb.model.stimcnn import *
from dfb.model.stftcnn import *
from dfb.model.clformer import *
from dfb.model.qlnrnet import *
from dfb.model.lnrnext import *


sampler_info = {
    "sgd": {
        "optimizer": torch.optim.SGD,
        "n_params": 1,
        "param_names": ["lr"],
        "lb": [-4],
        "ub": [0],
        "reversed": [False]
    },
    "momentum": {
        "optimizer": torch.optim.SGD,
        "n_params": 2,
        "param_names": ["lr", "momentum"],
        "lb": [-4, -3],
        "ub": [0, 0],
        "reversed": [False, True]
    },
    "rmsprop": {
        "optimizer": torch.optim.RMSprop,
        "n_params": 4,
        "param_names": ["lr", "momentum", "alpha", "eps"],
        "lb": [-4, -3, -3, -10],
        "ub": [-1, 0, 0, 0],
        "reversed": [False, True, True, False]
    },
    "adam": {
        "optimizer": torch.optim.Adam,
        "n_params": 4,
        "param_names": ["lr", "beta1", "beta2", "eps"],
        "lb": [-4, -3, -4, -10],
        "ub": [-1, 0, -1, 0],
        "reversed": [False, True, True, False] 
    }
}

model_info = {
    "stimcnn": {
        "model": STIMCNN,
        "sample_length": 784,
        "tf": [NpToTensor(), ToImage(28, 28, 1)]
    },
    "stftcnn": {
        "model": STFTCNN,
        "sample_length": 512,
        "tf": [STFT(window_length=128, noverlap=120, nfft=128),
               Resize(64, 64),
               NpToTensor(),
               ToImage(64, 64, 1)]
    },
    "wdcnn": {
        "model": WDCNN,
        "sample_length": 2048, 
        "tf": [MinMaxScaling(), NpToTensor(), ToSignal()]  # 기존
        # "tf": [AWGN(0), MinMaxScaling(), NpToTensor(), ToSignal()]  # 노이즈 강건성을 위한 코드
    },
    "lnrnet": {
        "model": LNRNet,
        "sample_length": 2048,
        "tf": [NpToTensor(), ToSignal()]
    },
    "maxlnrnet": {
        "model": LNRNet,
        "sample_length": 2048,
        "tf": [Normalize(norm="max"), NpToTensor(), ToSignal()],
        "shape": (1, 2048)
    },
    "wdcnnrnn": {
        "model": WDCNNRNN,
        "sample_length": 4096,
        "tf": [NpToTensor(), ToSignal()]
    },
    "ticnn": {
        "model": TICNN,
        "sample_length": 2048,
        "tf": [NpToTensor(), ToSignal()]
    },
    "dcn": {
        "model": DCN,
        "sample_length": 784,
        "tf": [NpToTensor(), ToSignal()]
    },
    "srdcnn": {
        "model": SRDCNN,
        "sample_length": 1024,
        "tf": [NpToTensor(), ToSignal()]
    },
    "aannet": {
        "model": AAnNet,
        "sample_length": 1670,
        "tf": [NpToTensor(), ToSignal()]
    },
    "clformer": {
        "model": CLFormer,
        "sample_length": 1024,
        "tf": [NpToTensor(), ToSignal()]
    }
}

data_info = {
    "cwru": {
        "n_classes": 10
    },
    "cwru_48k": {
        "n_classes": 10
    },
    "mfpt": {
        "n_classes": 3
    },
    "ottawa": {
        "n_classes": 5
    }
}

def get_model_list():
    return list(model_info.keys())

def get_data_list():
    return list(data_info.keys())

def get_optimizer_list():
    return list(sampler_info.keys())

def get_optimizer(name: str):
    return sampler_info[name]["optimizer"] 

def get_data_array(data: str,
                   sample_length: int):
    
    train_npz = np.load(f"./final_dataset/{data}/train.npz")
    val_npz = np.load(f"./final_dataset/{data}/val.npz")
    test_npz = np.load(f"./final_dataset/{data}/test.npz")

    train_data = {}
    val_data = {}
    test_data = {}

    train_data['data'] = train_npz['data'][:, :sample_length]
    train_data['label'] = train_npz['label']
    val_data['data'] = val_npz['data'][:, :sample_length]
    val_data['label'] = val_npz['label']
    test_data['data'] = test_npz['data'][:, :sample_length]
    test_data['label'] = test_npz['label']

    return (train_data, val_data, test_data)

def get_datamodule(data: str,
                   sample_length: int,
                   transform_data: transforms.transforms.Compose,
                   transform_label: transforms.transforms.Compose,
                   batch_size: int,
                   num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_array, val_array, test_array = get_data_array(data=data, sample_length=sample_length)

    train_loader = get_dataloader(train_array['data'],
                                  train_array['label'],
                                  transform_data,
                                  transform_label,
                                  True,
                                  batch_size,
                                  num_workers)
    
    val_loader = get_dataloader(val_array['data'],
                                val_array['label'],
                                transform_data,
                                transform_label,
                                False,
                                batch_size,
                                num_workers)
    
    test_loader = get_dataloader(test_array['data'],
                                 test_array['label'],
                                 transform_data,
                                 transform_label,
                                 False,
                                 batch_size,
                                 num_workers)

    return (train_loader, val_loader, test_loader)


def get_sampler(optimizer: str,
                n_exp: int) -> LogScaleSampler:
    if optimizer not in list(sampler_info.keys()):
        raise ValueError(f"optimizer must be {list(sampler_info.keys())}")

    return LogScaleSampler(n_params=sampler_info[optimizer]["n_params"],
                           param_names=sampler_info[optimizer]["param_names"],
                           lb=sampler_info[optimizer]["lb"],
                           ub=sampler_info[optimizer]["ub"],
                           reversed=sampler_info[optimizer]["reversed"],
                           n_exps=n_exp)


def get_sample_length(model: str) -> int:
    model_list = get_model_list()
    if model not in model_list:
        raise ValueError(f"### model argument must be {model_list}")

    return model_info[model]["sample_length"]


def get_num_classes(data: str) -> int:
    data_list = get_data_list()

    if data not in data_list:
        raise ValueError(f"data argument must be {data_list}")

    return data_info[data]["n_classes"]


def get_model(data: str, model: str, **kwargs):
    model_list = get_model_list()
    data_list = get_data_list()

    if model not in model_list:
        raise ValueError(f"### model argument must be {model_list}")
    if data not in data_list:
        raise ValueError(f"data argument must be {data_list}")
    
    n_classes = get_num_classes(data=data)

    return model_info[model]["model"](n_classes=n_classes, **kwargs)

def get_transform(model: str):
    model_list = get_model_list()
    if model not in model_list:
        raise ValueError(f"model argument must be {model_list}")
    
    return model_info[model]["tf"]
