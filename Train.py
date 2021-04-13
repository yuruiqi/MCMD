import os
import argparse
import sys
import json

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(r'/homes/rqyu/PycharmProjects/MyUtils')

from Data.Preprocess import join_path
from TrainUtils.Recorder import LossRecorder, SegRecorder
from Network.MG_Net import MGNet
from Network.Loss import DiceBCELoss
from Visualization.Image import show_multi_images

from Run import run
from Utils.Dataset import MyDataset
from Config import configs, Config_base
from Utils.UncertaintyLoss import AllLoss


def train(config):
    filters = [x*config['GROUP'] for x in config['FILTERS']]
    shape = config['SHAPE']
    group = config['GROUP']
    device = config['DEVICE']
    if config['PRELOAD'] == 0:
        train_preload = True
        val_preload = True
    elif config['PRELOAD'] == 1:
        train_preload = True
        val_preload = False
    else:
        train_preload = False
        val_preload = False

    model_save_dir = join_path(config['SAVE'], config['NAME'])
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    train_loader = DataLoader(MyDataset(config['TRAIN'], config, augment=True, preload=train_preload),
                              batch_size=config['BATCH'], shuffle=True, num_workers=5)
    val_loader = DataLoader(MyDataset(config['VAL'], config, preload=val_preload),
                            batch_size=config['BATCH'], shuffle=False, num_workers=5)

    if len(shape)==3:
        mode = '3d'
    else:
        mode = '2d'

    model = MGNet(group, filters, group, mode=mode, two_conv=config['TWO CONV'])
    model.to(device)

    if config['UNCERTAINTY'] is not None:
        criterion = AllLoss(mode=config['UNCERTAINTY'],
                            pos_weight=torch.ones(shape, device=device)*config['POS WEIGHT']
                            if config['POS WEIGHT'] is not None else None)
    else:
        criterion = DiceBCELoss(weight=torch.ones(shape, device=device) * config['POS WEIGHT']
        if config['POS WEIGHT'] is not None else None)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['LR'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    # Recorder

    train_loss_recorder = LossRecorder('train', save_dir=model_save_dir)
    val_loss_recorder = LossRecorder('val', patience=config['PATIENCE'], save_dir=model_save_dir)

    train_result_recorder = SegRecorder('train', save_dir=model_save_dir, n_classes=1)
    val_result_recorder = SegRecorder('val', save_dir=model_save_dir, n_classes=1)

    for epoch in range(config['EPOCH']):
        print('Epoch {}'.format(epoch))
        run(train_loader, model, criterion, optimizer, 'train', train_loss_recorder, train_result_recorder, config)

        with torch.no_grad():
            run(val_loader, model, criterion, optimizer, 'inference', val_loss_recorder, val_result_recorder, config)

        scheduler.step()

        # Recorder
        train_loss_recorder.new_epoch()
        train_loss_recorder.print_result()

        train_result_recorder.new_epoch()
        train_result_recorder.print_result()

        val_loss_recorder.new_epoch()
        val_loss_recorder.print_result()

        val_result_recorder.new_epoch()
        val_result_recorder.print_result()

        if epoch%5 == 0:
            img = val_result_recorder.data_all[-2]['img'][10][0]
            seg = val_result_recorder.results[-2]['label_merge'][10]
            pred = val_result_recorder.results[-2]['pred_merge'][10]
            var = val_result_recorder.data_all[-2]['var'][10]
            score = val_result_recorder.data_all[-2]['score'][10]

            var_dir = join_path(model_save_dir, 'var')
            if not os.path.exists(var_dir):
                os.mkdir(var_dir)

            show_multi_images([{'name':'label', 'img':img, 'roi':seg},
                               {'name':'pred', 'img':img, 'roi':pred},
                               {'name': 'var', 'img': var},
                               {'name':'var2', 'img':img, 'roi':var}],
                              arrangement=[2,2],
                              save_path=join_path(var_dir, f'{epoch}.png'),
                              title=score.item())

        train_result_recorder.clear()
        val_result_recorder.clear()

        # save
        save, finish = val_loss_recorder.judge(key='loss')
        if save:
            print('Saving.')
            best_epoch = epoch
            torch.save(model.state_dict(), join_path(model_save_dir, config['NAME'] + '.pkl'))
        print('')
        if finish:
            break

    print('********** Best Epoch: {} **********'.format(best_epoch))
    train_loss_recorder.print_result(index=best_epoch, keys=['loss', 'dice', 'bce'])
    val_loss_recorder.print_result(index=best_epoch, keys=['loss', 'dice', 'bce'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=int, default=0, help='the id of the config')
    parser.add_argument('--preload', type=int, default=0, help='0:all, 1:train, 2:none')
    parser.add_argument('--device', type=int, default=2, help='cuda id')
    parser.add_argument('--batch', type=int, default=32, help='batch size')

    config_id = parser.parse_args().config
    config = Config_base.copy()
    config.update(configs[config_id])

    config['PRELOAD'] = parser.parse_args().preload
    config['DEVICE'] = torch.device(f'cuda:{parser.parse_args().device}')
    config['BATCH'] = parser.parse_args().batch

    # config = Config_base.copy()
    # config.update(configs[6])
    # config['PRELOAD'] = 2
    # config['DEVICE'] = 0
    # config['BATCH'] = 32

    train(config)