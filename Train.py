import os
import time
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


def save_var_train(recorder, i, epoch, model_save_dir):
    img = recorder.data_all[-2]['img'][i][0]
    seg = recorder.results[-2]['label_merge'][i]
    pred = recorder.results[-2]['pred_merge'][i]
    var = recorder.data_all[-2]['var'][i]
    score = recorder.data_all[-2]['score'][i]

    var_dir = join_path(model_save_dir, 'var')
    if not os.path.exists(var_dir):
        os.mkdir(var_dir)

    show_multi_images([{'name': 'label', 'img': img, 'roi': seg},
                       {'name': 'pred', 'img': img, 'roi': pred},
                       {'name': 'var', 'img': var},
                       {'name': 'var2', 'img': img, 'roi': var}],
                      arrangement=[2, 2],
                      save_path=join_path(var_dir, f'{epoch}.png'),
                      title=score.item())


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

    model_save_dir = out_dir
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    train_loader = DataLoader(MyDataset(config['TRAIN'], config, augment=True, preload=train_preload),
                              batch_size=config['BATCH'], shuffle=True, num_workers=5)
    val_loader = DataLoader(MyDataset(config['VAL'], config, preload=val_preload),
                            batch_size=config['BATCH'], shuffle=False, num_workers=5)

    mode = '3d' if len(shape)==3 else '2d'

    model = MGNet(group, filters, group, mode=mode, two_conv=config['TWO CONV'], attention=config['ATTENTION'])
    model.to(device)

    criterion = AllLoss(mode=config['LOSS MODE'], weight=config['UNCERTAINTY WEIGHT'])

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

        if group>1 and epoch%5 == 0:
            save_var_train(val_result_recorder, 10, epoch, model_save_dir)

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

    train_loss_recorder.draw(save=True)
    val_loss_recorder.draw(save=True)

    print('********** Best Epoch: {} **********'.format(best_epoch))
    train_loss_recorder.print_result(index=best_epoch, keys=['loss', 'dice', 'bce'])
    val_loss_recorder.print_result(index=best_epoch, keys=['loss', 'dice', 'bce'])
    train_result_recorder.print_result(index=best_epoch)
    val_result_recorder.print_result(index=best_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=int, default=0, help='the id of the config')
    parser.add_argument('--preload', type=int, default=0, help='0:all, 1:train, 2:none')
    parser.add_argument('--device', type=int, default=0, help='cuda id')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    args = parser.parse_args()
    
    config_id = args.config
    config = Config_base.copy()
    config.update(configs[config_id])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    config['PRELOAD'] = args.preload
    config['DEVICE'] = torch.device(f'cuda:{args.device}')
    config['BATCH'] = args.batch

    arguments = vars(args)
    arguments.update(config)
    arguments.pop('DEVICE')
    
    out_dir = join_path(config['SAVE'], config['NAME'], time.strftime("%m%d_%H%M"))
    os.makedirs(out_dir, exist_ok=True)
    with open(join_path(out_dir, 'params.json'), 'w') as f:
        json.dump(arguments, f, indent=2)

    train(config)
