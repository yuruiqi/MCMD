import torch
from Run import run
from torch.utils.data import DataLoader
import json
import os
import matplotlib.pyplot as plt

from Data.Preprocess import join_path
from TrainUtils.Recorder import LossRecorder, SegRecorder
from Network.MG_Net import MGNet
from Network.Loss import DiceBCELoss
from ImageProcess.Analysis import get_biggest_slice
from ImageProcess.Operations import normalize01
from Visualization.Image import show_multi_images

from Utils.Dataset import MyDataset
from Config import configs, Config_base
from Utils.UncertaintyLoss import AllLoss


def show_3d(recorder):
    for i in range(3):
        img = recorder.data_all[0]['img'][i][0]
        seg = recorder.results[0]['label_merge'][i]
        pred = recorder.results[0]['pred_merge'][i]

        slice = get_biggest_slice(seg)

        for j in range(slice-1, slice+2):
            plt.imshow(img[:,:,j], cmap='gray')
            plt.contour(seg[:,:,j])
            plt.title('label')
            plt.show()

            plt.imshow(img[:,:,j], cmap='gray')
            plt.contour(pred[:,:,j])
            plt.title('pred')
            plt.show()


def show_2d_uncertainty(recorder, casenames, save_dir):
    for i in range(len(recorder.data_all[-2]['img'])):
        img = recorder.data_all[-2]['img'][i][0]
        seg = recorder.results[-2]['label_merge'][i]
        pred = recorder.results[-2]['pred_merge'][i]
        var = recorder.data_all[-2]['var'][i]
        score = recorder.data_all[-2]['score'][i]
        group0 = recorder.data_all[-2]['group'][i][0]
        group1 = recorder.data_all[-2]['group'][i][1]
        group2 = recorder.data_all[-2]['group'][i][2]
        group3 = recorder.data_all[-2]['group'][i][3]

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        show_multi_images([{'name': 'label', 'img': img, 'roi': seg},
                           {'name': 'pred', 'img': img, 'roi': pred},
                           {'name': 'var', 'img': var},
                           {'name': 'var roi', 'img': img, 'contour':normalize01(var)},
                           {'name': 'group0', 'img': group0},
                           {'name': 'group1', 'img': group1},
                           {'name': 'group2', 'img': group2},
                           {'name': 'group3', 'img': group3}],
                          arrangement=[2, 4],
                          save_path=join_path(save_dir, f'{i}.png'),
                          title=f'{casenames[i]} {score.item()}')


def show_2d(recorder, casenames, save_dir):
    for i in range(len(recorder.data_all[-2]['img'])):
        img = recorder.data_all[-2]['img'][i][0]
        seg = recorder.results[-2]['label_merge'][i]
        pred = recorder.results[-2]['pred_merge'][i]
        score = recorder.data_all[-2]['score'][i]

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        show_multi_images([{'name': 'label', 'img': img, 'roi': seg},
                           {'name': 'pred', 'img': img, 'roi': pred}],
                          arrangement=[1,2],
                          save_path=join_path(save_dir, f'{i}.png'),
                          title=f'{casenames[i]} {score.item()}')


def inference(config):
    filters = [x * config['GROUP'] for x in config['FILTERS']]
    shape = config['SHAPE']
    group = config['GROUP']
    device = config['DEVICE']
    model_save_dir = join_path(config['SAVE'], config['NAME'])

    print(config['NAME'])

    test_loader = DataLoader(MyDataset(config['TEST'], config, augment=False, preload=False),
                             batch_size=config['BATCH'], shuffle=False, num_workers=5)

    casenames = [x['image'] for x in test_loader.dataset.datalist]

    if len(shape) == 3:
        mode = '3d'
    else:
        mode = '2d'

    model = MGNet(group, filters, group, mode=mode, two_conv=config['TWO CONV'], attention=config['ATTENTION'])
    model.load_state_dict(torch.load(join_path(model_save_dir, config['NAME'] + '.pkl')))
    model.to(device)

    criterion = AllLoss(mode=config['LOSS MODE'], weight=config['UNCERTAINTY WEIGHT'])

    # Recorder

    test_loss_recorder = LossRecorder('test', save_dir=model_save_dir)
    test_result_recorder = SegRecorder('test', save_dir=model_save_dir, n_classes=1)

    with torch.no_grad():
        run(test_loader, model, criterion, None, 'inference', test_loss_recorder, test_result_recorder, config)

    # Recorder
    test_loss_recorder.new_epoch()
    test_loss_recorder.print_result()

    test_result_recorder.new_epoch()
    test_result_recorder.print_result()

    # show_3d(test_result_recorder)
    # if config['GROUP'] > 1:
    #     show_2d_uncertainty(test_result_recorder, casenames, join_path(model_save_dir, 'test_var'))
    # else:
    #     show_2d(test_result_recorder, casenames, join_path(model_save_dir, 'test_var'))


if __name__ == '__main__':
    config = Config_base.copy()
    config.update(configs[10])
    config['PRELOAD'] = 2
    config['DEVICE'] = 0
    config['BATCH'] = 32

    inference(config)
