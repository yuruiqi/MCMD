import torch
import numpy as np


Config_base = {
    # Experiment
    'NAME': 'Base',

    # Path
    'TRAIN': '/homes/rqyu/Projects/MCMD/Data/Split Temp/train.json',
    'VAL': '/homes/rqyu/Projects/MCMD/Data/Split Temp/val.json',
    'TEST': '/homes/rqyu/Projects/MCMD/Data/Split Temp/test.json',

    'SAVE': r'/homes/clwang/Data/MCMD/Results',

    # Train
    'EPOCH': 1000,
    'BATCH': 32,
    'PATIENCE': 200,
    'LR': 0.01,
    'POS WEIGHT': None,

    # Model
    'OUT MODE': 'origin',
    'LOSS MODE': 'dicebce',
    'UNCERTAINTY WEIGHT': 1,
    'ATTENTION': False,
    'SHAPE': [64,64],
    'GROUP': 1,
    'TWO CONV': True,
    'FILTERS': [16, 32, 64, 128, 256]
}


Config0 = {
    'NAME': '0.Unet',
}

Config1 = {
    'NAME': '1.Mgnet',
    'GROUP':4,
}

Config2 = {
    'NAME': '2.Uncertainty',
    'LOSS MODE': 'uncertainty',
    'GROUP': 4,
}

Config3 = {
    'NAME': '3.Uncertainty_aug',
    'LOSS MODE': 'uncertainty',
    'AUG MODE': 'conservative',
    'GROUP': 4,
}


Config4 = {
    'NAME': '4.Uncertainty_aug_weight',
    'LOSS MODE': 'uncertainty',
    'AUG MODE': 'conservative',
    'UNCERTAINTY WEIGHT': 100,
    'GROUP': 4,
}


Config5 = {
    'NAME': '5.attention',
    'GROUP':4,
    'ATTENTION':True
}


Config6 = {
    'NAME': '6. Uncertantiy_aug_weight1000',
    'LOSS MODE': 'uncertainty',
    'AUG MODE': 'conservative',
    'UNCERTAINTY WEIGHT': 1000,
    'GROUP': 4,
}


configs = [Config0, Config1, Config2, Config3, Config4, Config5, Config6]
