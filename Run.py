import torch
import torch.nn as nn
from ImageProcess.Operations import seg_to_mask
from ImageProcess.Analysis import get_biggest_slice
from Network.Utils import batch_slice
import matplotlib.pyplot as plt


def run(data_loader, model, criterion, optimizer, mode, loss_recorder, result_recorder, config):
    if mode == 'train':
        model.train()
    elif mode == 'inference':
        model.eval()

    device = config['DEVICE']

    for img, seg, score in data_loader:
        # ids.append(id)
        img = img.to(torch.float32).to(device)  # (batch, n, h, w, d) if AUG MODE, else (batch, 1, h, w, d)
        # TODO: now only (batch, 1, h, w, d). ignore AUG MODE='ALL'
        seg = seg.to(torch.float32).to(device)  # (batch, n, h, w, d) if AUG MODE, else (batch, 1, h, w, d)

        prediction = model(img)  # (batch, n, h, w, d) before sigmoid

        # Optimize
        if prediction.shape[1] != seg.shape[1]:

            img_show = img[0,0].detach().cpu().numpy()
            seg_show = seg[0,0].detach().cpu().numpy()
            prediction_show = prediction[0].detach().cpu().numpy()

            plt.imshow(img_show, cmap='gray')
            plt.show()
            plt.imshow(prediction_show[0], cmap='gray')
            plt.show()
            plt.imshow(prediction_show[1], cmap='gray')
            plt.show()
            plt.imshow(prediction_show[2], cmap='gray')
            plt.show()
            plt.imshow(prediction_show[3], cmap='gray')
            plt.show()
            plt.imshow(seg_show, cmap='gray')
            plt.show()

            prediction_mean = prediction.mean(dim=1, keepdim=True)
            prediction_var = prediction.var(dim=1)
            if config['UNCERTAINTY'] is None:
                losses = criterion(prediction_mean, seg)
            else:
                losses = criterion(prediction, seg)
        else:
            losses = criterion(prediction, seg)

        if len(losses) == 3:
            loss, dice, bce = losses
        else:
            loss, dice, bce, uncertainty = losses
            loss_recorder.record(uncertainty.detach().item(), 'uncertainty')

        loss_recorder.record(loss.detach().item(), 'loss')
        loss_recorder.record(dice.detach().item(), 'dice')
        loss_recorder.record(bce.detach().item(), 'bce')

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if result_recorder is not None:
            if prediction.shape[1] != seg.shape[1]:
                result_recorder.record(img.detach().cpu().numpy(), 'img')  # (batch, 1, h, w, d)
                result_recorder.record(seg.detach().cpu().numpy(), 'label')  # (batch, 1, h, w, d)
                result_recorder.record(torch.sigmoid(prediction_mean).detach().cpu().numpy(), 'pred')  # (batch, 1, h, w, d)
                result_recorder.record(torch.sigmoid(prediction_var).detach().cpu().numpy(), 'var')  # (batch, h, w, d)
                result_recorder.record(score.detach().cpu().numpy(), 'score')
            else:
                shape = seg.shape
                new_shape = (shape[0] * shape[1], 1) + shape[2:]
                result_recorder.record(img.reshape(new_shape).detach().cpu().numpy(), 'img')  # (batch*n, h, w, d)
                result_recorder.record(seg.reshape(new_shape).detach().cpu().numpy(), 'label')  # (batch*n, 1, h, w, d)
                prediction = torch.sigmoid(prediction).reshape(new_shape).detach().cpu().numpy()  # (batch*n, 1, h, w, d)
                result_recorder.record(prediction, 'pred')
                result_recorder.record(score.detach().cpu().numpy(), 'score')
