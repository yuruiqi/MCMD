import torch
import numpy as np


Config_base = {
    # Experiment
    'NAME': 'Base',

    # Path
    'TRAIN': '/homes/rqyu/Projects/MCMD/Data/Split Temp/train.json',
    'VAL': '/homes/rqyu/Projects/MCMD/Data/Split Temp/val.json',
    'TEST': '/homes/rqyu/Projects/MCMD/Data/Split Temp/test.json',

    'SAVE': r'/homes/rqyu/Projects/MCMD/Results',

    # Train
    'EPOCH': 1000,
    'BATCH': 32,
    'PATIENCE': 200,
    'LR': 0.01,
    'POS WEIGHT': None,

    # Model
    'UNCERTAINTY': None,
    'OUT MODE': 'origin',
    'SHAPE': [64,64,32],
    'GROUP': 1,
    'TWO CONV': False,
    'FILTERS': [16, 32, 64, 128, 256]
}

Config0 = {
    'Name':'Unet',
    'TWO CONV': True,
}

Config1 = {
    'NAME': 'Mgnet',
    'GROUP':4
}

Config2 = {
    'NAME': 'Uncertainty',
    'UNCERTAINTY': 1,
    'GROUP': 4,
}

Config3 = {
    'NAME': 'Uncertainty_aug',
    'UNCERTAINTY': 1,
    'AUG MODE': 'conservative',
    'GROUP': 4,
}

Config4 = {
    'NAME': 'Unet_2d',
    'TWO CONV': True,
    'SHAPE': [64,64]
}

Config5 = {
    'NAME': 'Mgnet_2d',
    'GROUP':4,
    'SHAPE': [64, 64]
}

Config6 = {
    'NAME': 'Uncertainty_2d',
    'UNCERTAINTY': 1,
    'GROUP': 4,
    'SHAPE': [64, 64]
}

Config7 = {
    'NAME': 'Uncertainty_aug_2d',
    'UNCERTAINTY': 1,
    'AUG MODE': 'conservative',
    'GROUP': 4,
    'SHAPE': [64, 64]
}

Config8 = {
    'NAME': 'Uncertainty_aug_2d_l1',
    'UNCERTAINTY': 1,
    'AUG MODE': 'conservative',
    'GROUP': 4,
    'SHAPE': [64, 64]
}


configs = [Config0, Config1, Config2, Config3, Config4, Config5, Config6, Config7, Config8]
