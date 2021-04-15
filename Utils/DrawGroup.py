import os

from Data.Preprocess import join_path
from ImageProcess.Operations import normalize01
from Visualization.Image import show_multi_images


def drawgroup(recorder, mode, config, save_dir, casenames=None, epoch=None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if config['GROUP']>1:
        group = True
    else:
        group = False

    if mode == 'train':
        drawuncertainty(recorder, 10, join_path(save_dir, f'{epoch}.png'), group, casenames)
    else:
        for i in range(len(recorder.data_all[-2]['img'])):
            drawuncertainty(recorder, i, join_path(save_dir, f'{i}.png'), group)


def drawuncertainty(recorder, i, save_path, group, casenames=None):
    img = recorder.data_all[-2]['img'][i][0]
    seg = recorder.results[-2]['label_merge'][i]
    pred = recorder.results[-2]['pred_merge'][i]
    var = recorder.data_all[-2]['var'][i]
    score = recorder.data_all[-2]['score'][i]

    casename = 'Unnamed' if casenames is None else casenames[i]

    if group:
        group0 = recorder.data_all[-2]['group'][i][0]
        group1 = recorder.data_all[-2]['group'][i][1]
        group2 = recorder.data_all[-2]['group'][i][2]
        group3 = recorder.data_all[-2]['group'][i][3]

        show_multi_images([{'name': 'label', 'img': img, 'roi': seg},
                           {'name': 'pred', 'img': img, 'roi': pred},
                           {'name': 'var', 'img': var},
                           {'name': 'var roi', 'img': img, 'contour':normalize01(var)},
                           {'name': 'group0', 'img': group0},
                           {'name': 'group1', 'img': group1},
                           {'name': 'group2', 'img': group2},
                           {'name': 'group3', 'img': group3}],
                          arrangement=[2, 4],
                          save_path=save_path,
                          title=f'{casename} {score.item()}')
    else:
        show_multi_images([{'name': 'label', 'img': img, 'roi': seg},
                           {'name': 'pred', 'img': img, 'roi': pred}],
                          arrangement=[1,2],
                          save_path=save_path,
                          title=f'{casename} {score.item()}')
