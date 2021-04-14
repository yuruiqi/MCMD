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
        seg = seg.to(torch.float32).to(device)  # (batch, n, h, w, d) if AUG MODE, else (batch, 1, h, w, d)

        prediction = model(img)  # (batch, n, h, w, d) before sigmoid

        # Optimize
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
            if prediction.shape[1] > 1:
                mean = prediction.mean(dim=1, keepdim=True).detach().cpu().numpy()
                var = torch.sigmoid(prediction.var(dim=1)).detach().cpu().numpy()

                result_recorder.record(img.detach().cpu().numpy(), 'img')  # (batch, 1, h, w, d)
                result_recorder.record(seg[:,0:1].detach().cpu().numpy(), 'label')  # (batch, 1, h, w, d)
                result_recorder.record(mean, 'pred')  # (batch, 1, h, w, d)
                result_recorder.record(var, 'var')  # (batch, h, w, d)
                result_recorder.record(score.detach().cpu().numpy(), 'score')
                result_recorder.record(prediction.detach().cpu().numpy(), 'group')  # (batch, n, h, w, d)
            else:
                result_recorder.record(img.detach().cpu().numpy(), 'img')  # (batch, 1, h, w, d)
                result_recorder.record(seg.detach().cpu().numpy(), 'label')  # (batch, 1, h, w, d)
                result_recorder.record(prediction.detach().cpu().numpy(), 'pred')  # (batch, 1, h, w, d)
                result_recorder.record(score.detach().cpu().numpy(), 'score')
